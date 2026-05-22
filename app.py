import io
import json
import os
import secrets
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from captcha.image import ImageCaptcha
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "archive")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

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


def preprocess_image(img_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns (processed_array, (original_h, original_w))."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    orig_h, orig_w = img.shape[:2]

    MAX_SIDE = 1000
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, (orig_h, orig_w)


def run_ocr(processed: np.ndarray) -> tuple[str, int]:
    """Returns (extracted_text, tokens_generated)."""
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
        tokens_generated = int(generated_ids.shape[1] - input_len)
        result = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return result.strip(), tokens_generated
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _write_archive(
    job_id: str,
    meta: dict,
    input_bytes: bytes | None,
    processed: np.ndarray | None,
):
    """Write job artifacts to disk in a background thread."""
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        job_dir = os.path.join(ARCHIVE_DIR, date_str, job_id)
        os.makedirs(job_dir, exist_ok=True)

        if input_bytes is not None:
            ext = meta.get("file_ext", ".bin")
            with open(os.path.join(job_dir, f"input{ext}"), "wb") as f:
                f.write(input_bytes)

        if processed is not None:
            cv2.imwrite(os.path.join(job_dir, "processed.png"), processed)

        with open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        print(f"[archive] Failed to write job {job_id}:\n{traceback.format_exc()}")


def _archive_async(job_id, meta, input_bytes, processed):
    t = threading.Thread(
        target=_write_archive,
        args=(job_id, meta, input_bytes, processed),
        daemon=True,
    )
    t.start()


def _find_job_meta_path(job_id: str) -> str | None:
    if not os.path.isdir(ARCHIVE_DIR):
        return None
    for date_dir in sorted(os.listdir(ARCHIVE_DIR), reverse=True)[:3]:
        path = os.path.join(ARCHIVE_DIR, date_dir, job_id, "meta.json")
        if os.path.exists(path):
            return path
    return None


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
    job_id = uuid.uuid4().hex
    t_start = time.monotonic()

    meta = {
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "original_filename": None,
        "file_size_bytes": None,
        "file_ext": None,
        "client_ip": get_remote_address(),
        "status": "error",
        "error_stage": None,
        "error": None,
        "result": None,
        "tokens_generated": None,
        "input_image_shape": None,
        "processed_image_shape": None,
        "duration_preprocess_ms": None,
        "duration_ocr_ms": None,
        "duration_total_ms": None,
    }
    input_bytes = None
    processed = None

    # --- CAPTCHA check ---
    user_answer = request.form.get("captcha", "").strip().upper()
    expected = session.pop("captcha", None)
    if not expected or user_answer != expected:
        meta["error_stage"] = "captcha"
        meta["error"] = "Invalid CAPTCHA"
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": "Invalid CAPTCHA. Please try again.", "captcha_error": True}), 400

    # --- Model readiness check ---
    with model_lock:
        if model_error:
            meta["error_stage"] = "model"
            meta["error"] = model_error
            meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
            _archive_async(job_id, meta, input_bytes, processed)
            return jsonify({"error": f"Model failed to load: {model_error}"}), 500
        if model is None:
            meta["error_stage"] = "model"
            meta["error"] = "Model not yet loaded"
            meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
            _archive_async(job_id, meta, input_bytes, processed)
            return jsonify({"error": "Model is still loading, please wait."}), 503

    # --- File validation ---
    if "image" not in request.files:
        meta["error_stage"] = "validation"
        meta["error"] = "No image file provided"
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        meta["error_stage"] = "validation"
        meta["error"] = "No file selected"
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        meta["error_stage"] = "validation"
        meta["error"] = f"Unsupported file type: {ext}"
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    input_bytes = file.read()
    meta["original_filename"] = file.filename
    meta["file_size_bytes"] = len(input_bytes)
    meta["file_ext"] = ext

    # --- Preprocessing ---
    t_pre = time.monotonic()
    try:
        processed, orig_shape = preprocess_image(input_bytes)
        meta["duration_preprocess_ms"] = round((time.monotonic() - t_pre) * 1000)
        meta["input_image_shape"] = list(orig_shape)
        meta["processed_image_shape"] = list(processed.shape[:2])
    except Exception as e:
        meta["error_stage"] = "preprocessing"
        meta["error"] = str(e)
        meta["duration_preprocess_ms"] = round((time.monotonic() - t_pre) * 1000)
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": f"Image preprocessing failed: {e}"}), 422

    # --- OCR ---
    t_ocr = time.monotonic()
    try:
        text, tokens_generated = run_ocr(processed)
        meta["duration_ocr_ms"] = round((time.monotonic() - t_ocr) * 1000)
        meta["result"] = text
        meta["tokens_generated"] = tokens_generated
        meta["status"] = "success"
    except Exception as e:
        meta["error_stage"] = "ocr"
        meta["error"] = str(e)
        meta["duration_ocr_ms"] = round((time.monotonic() - t_ocr) * 1000)
        meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
        _archive_async(job_id, meta, input_bytes, processed)
        return jsonify({"error": f"OCR failed: {e}"}), 500

    meta["duration_total_ms"] = round((time.monotonic() - t_start) * 1000)
    _archive_async(job_id, meta, input_bytes, processed)
    return jsonify({"text": text, "job_id": job_id})


@app.route("/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback():
    job_id = request.form.get("job_id", "").strip()
    rating = request.form.get("rating", "").strip()
    corrected_text = request.form.get("corrected_text", "")

    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    if rating not in ("up", "down"):
        return jsonify({"error": "Invalid rating"}), 400

    meta_path = _find_job_meta_path(job_id)
    if meta_path is None:
        return jsonify({"error": "Job not found"}), 404

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        predicted = meta.get("result", "")
        is_correction = bool(corrected_text) and corrected_text != predicted

        meta["feedback_rating"] = rating
        meta["feedback_corrected_text"] = corrected_text if is_correction else None
        meta["feedback_is_correction"] = is_correction
        meta["feedback_timestamp"] = datetime.now(timezone.utc).isoformat()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if is_correction:
            correction_path = os.path.join(os.path.dirname(meta_path), "corrected.txt")
            with open(correction_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["POST"])
def download():
    text = request.form.get("text", "")
    buf = io.BytesIO(text.encode("utf-8-sig"))
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
