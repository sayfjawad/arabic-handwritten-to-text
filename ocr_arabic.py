import os
import sys
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_NAME = "sherif1313/Arabic-English-handwritten-OCR-v3"

#IMAGE_DIR = "images"
IMAGE_DIR = "processed_images"
OUTPUT_FILE = "output.txt"


def force_lm_head_to_embeddings(model):
    print("Forcing lm_head.weight to use input embedding weights...")

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    print("Input embedding weight shape:", tuple(input_emb.weight.shape))
    print("Output embedding weight shape:", tuple(output_emb.weight.shape))

    if input_emb.weight.shape != output_emb.weight.shape:
        raise RuntimeError(
            f"Shape mismatch: input_emb={tuple(input_emb.weight.shape)}, "
            f"output_emb={tuple(output_emb.weight.shape)}"
        )

    # Important fix:
    # The checkpoint is missing lm_head.weight.
    # We manually tie lm_head.weight to the token embedding matrix.
    output_emb.weight = input_emb.weight

    same_storage = input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
    print("lm_head tied to input embeddings:", same_storage)

    if not same_storage:
        raise RuntimeError("Failed to tie lm_head to input embeddings")

    return model


def single_gpu_device_config():
    """Return (device_map, max_memory) that keeps the full model on one device.

    device_map="auto" without memory limits splits the model across all visible
    GPUs, which breaks Qwen2.5-VL inference because visual and text tokens end
    up on different devices.  We restrict GPU allocation to the GPU with the
    most free VRAM; any overflow spills to CPU RAM via accelerate's offload.
    """
    if not torch.cuda.is_available():
        return "cpu", None

    n = torch.cuda.device_count()
    free_gb = [torch.cuda.mem_get_info(i)[0] / 1024 ** 3 for i in range(n)]
    best = max(range(n), key=lambda i: free_gb[i])
    free = free_gb[best]

    # Only grant memory to the chosen GPU; other GPUs get nothing.
    # accelerate will spill to CPU if the model doesn't fit.
    max_memory = {i: "0GiB" for i in range(n)}
    max_memory[best] = f"{int(free * 0.9)}GiB"
    max_memory["cpu"] = "32GiB"

    print(f"Using GPU {best}: {torch.cuda.get_device_name(best)} ({free:.1f} GiB free)")
    return "auto", max_memory


def load_model():
    print("=" * 80)
    print(f"Loading model: {MODEL_NAME}")
    print("=" * 80)

    device_map, max_memory = single_gpu_device_config()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="eager",  # flash_attn not installed; use standard attention
    )

    model = force_lm_head_to_embeddings(model)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model.eval()
    return model, processor


def extract_text_from_image(model, processor, image_path):
    prompt = (
        "اقرأ النص العربي الموجود في الصورة واستخرج النص فقط. "
        "لا تشرح. لا تترجم. لا تضف أي شيء غير النص المكتوب."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    device = next(model.parameters()).device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

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

    output_text = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def find_images():
    if not os.path.isdir(IMAGE_DIR):
        print(f"Image directory does not exist: {IMAGE_DIR}")
        sys.exit(1)

    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")

    return sorted(
        os.path.join(IMAGE_DIR, filename)
        for filename in os.listdir(IMAGE_DIR)
        if filename.lower().endswith(image_extensions)
    )


def main():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is not available. This will be slow.")

    image_files = find_images()

    if not image_files:
        print(f"No images found in: {IMAGE_DIR}")
        return

    model, processor = load_model()

    results = []

    for image_path in image_files:
        print("=" * 80)
        print(f"Processing: {image_path}")

        try:
            text = extract_text_from_image(model, processor, image_path)
        except Exception as e:
            text = f"[ERROR while processing {image_path}: {e}]"

        print("\nOCR result:")
        print(text)

        results.append(f"===== {image_path} =====\n{text}\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print("=" * 80)
    print(f"Done. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
