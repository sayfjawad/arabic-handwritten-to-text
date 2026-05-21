import os
import io
import uuid
import secrets
import threading
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from captcha.image import ImageCaptcha
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

_image_captcha = ImageCaptcha(width=220, height=72)
_CAPTCHA_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _gen_captcha_text(length: int = 5) -> str:
    return "".join(secrets.choice(_CAPTCHA_CHARS) for _ in range(length))

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
MODEL_NAME = "sherif1313/Arabic-English-handwritten-OCR-v3"

model = None
processor = None
model_lock = threading.Lock()
model_loading = False
model_error = None


def _force_lm_head_to_embeddings(m):
    input_emb = m.get_input_embeddings()
    output_emb = m.get_output_embeddings()
    if input_emb.weight.shape != output_emb.weight.shape:
        raise RuntimeError("Shape mismatch between input and output embeddings")
    output_emb.weight = input_emb.weight
    return m


def load_model_background():
    global model, processor, model_loading, model_error
    try:
        print(f"Loading model {MODEL_NAME} ...")
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        m = _force_lm_head_to_embeddings(m)
        p = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        m.eval()
        with model_lock:
            model = m
            processor = p
        print("Model loaded and ready.")
    except Exception as e:
        with model_lock:
            model_error = str(e)
        print(f"Model loading failed: {e}")
    finally:
        model_loading = False


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Resize, denoise and enhance contrast — mirrors preprocess_images.py."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    MAX_SIDE = 1000
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def run_ocr(processed: np.ndarray) -> str:
    pil_image = Image.fromarray(processed)

    # Save to a temp file because process_vision_info expects a file path or URL
    tmp_path = os.path.join(UPLOAD_DIR, f"_tmp_{uuid.uuid4().hex}.png")
    try:
        pil_image.save(tmp_path)

        prompt = (
            "اقرأ النص العربي الموجود في الصورة واستخرج النص فقط. "
            "لا تشرح. لا تترجم. لا تضف أي شيء غير النص المكتوب."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        result = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return result.strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/captcha")
def captcha():
    text = _gen_captcha_text()
    session["captcha"] = text
    img_data = _image_captcha.generate(text)
    response = send_file(img_data, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/status")
def status():
    with model_lock:
        if model_error:
            return jsonify({"status": "error", "message": model_error})
        if model is None:
            return jsonify({"status": "loading"})
        return jsonify({"status": "ready"})


@app.route("/process", methods=["POST"])
@limiter.limit("20 per minute")
def process():
    user_answer = request.form.get("captcha", "").strip().upper()
    expected = session.pop("captcha", None)
    if not expected or user_answer != expected:
        return jsonify({"error": "Invalid CAPTCHA. Please try again.", "captcha_error": True}), 400

    with model_lock:
        if model_error:
            return jsonify({"error": f"Model failed to load: {model_error}"}), 500
        if model is None:
            return jsonify({"error": "Model is still loading, please wait."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    img_bytes = file.read()

    try:
        processed = preprocess_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {e}"}), 422

    try:
        text = run_ocr(processed)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    return jsonify({"text": text})


@app.route("/download", methods=["POST"])
def download():
    text = request.form.get("text", "")
    buf = io.BytesIO(text.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="arabic_ocr_result.txt",
    )


model_loading = True
_t = threading.Thread(target=load_model_background, daemon=True)
_t.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
