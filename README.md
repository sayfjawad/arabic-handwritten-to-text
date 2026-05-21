# Arabic Handwritten OCR

A local OCR service for extracting text from Arabic handwritten (and printed) images. It uses the [`sherif1313/Arabic-English-handwritten-OCR-v3`](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3) model, a Qwen2.5-VL fine-tune, and ships both a Flask web UI and a CLI batch processor.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 and drivers installed on the host — CPU inference is impractically slow
- **24 GB VRAM minimum (safe floor)** — see the VRAM breakdown below
- Docker + NVIDIA Container Toolkit (only if using the Dev Container setup)

### VRAM requirements

The model (`sherif1313/Arabic-English-handwritten-OCR-v3`) is a Qwen2.5-VL fine-tune with roughly **8.3 B total parameters** (language model + vision encoder). It is loaded in **float16** with no quantization.

| Budget item | Approximate size |
|---|---|
| Model weights (float16) | ~16.6 GB |
| KV cache (GQA, typical sequence) | ~0.2 GB |
| Activations + CUDA overhead | ~1.5–2 GB |
| **Total at inference** | **~18–20 GB** |

What this means in practice:

- **< 16 GB — will not load.** The weights alone exceed 16 GB, so an 8 GB or 16 GB card cannot fit the model at all.
- **16–20 GB — likely OOM.** The model fits in theory only if nothing else shares the GPU, and even then activations during a forward pass can push usage over the edge.
- **24 GB — safe minimum.** Leaves ~4–6 GB headroom for activations and the OS/CUDA context. Cards in this tier: RTX 3090 / 3090 Ti, RTX 4090, RTX 6000 Ada, A5000, L4 24 GB.
- **32 GB+ — comfortable.** Tested configuration (V100 32 GB). Other options: A100 40 GB, A100 80 GB, H100.

## Installation

### Option 1: Local setup (Linux/Ubuntu)

Run the included setup script from inside the project directory. It creates a `.venv`, installs the CUDA 12.1 PyTorch wheel, and all Python dependencies:

```bash
bash setup_arabic_ocr_env.sh
```

To target a different directory:

```bash
bash setup_arabic_ocr_env.sh /path/to/arabic-ocr
```

The script ends with an environment check that prints PyTorch version, OpenCV version, and whether CUDA was detected.

#### Manual install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes the CUDA 12.1 index URL so PyTorch is fetched with GPU support automatically.

### Option 2: VS Code Dev Container

1. Open the project folder in VS Code.
2. When prompted, click **Reopen in Container** (or press `F1` → `Dev Containers: Reopen in Container`).
3. The container builds automatically and installs all dependencies. The first build downloads the PyTorch GPU base image which may take a few minutes.

The Dev Container `requirements.txt` omits the PyTorch packages because they are already present in the `pytorch/pytorch` base image.

## Running the web app

```bash
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python app.py
```

The server starts on `http://0.0.0.0:5000`. The model loads in the background — the UI shows a pulsing amber dot while loading and a green dot once ready (typically 30–90 seconds on first run; subsequent runs are faster after the model is cached by HuggingFace).

### Web UI features

- Drag-and-drop or browse to upload an image
- Live image preview before processing
- RTL Arabic text output area
- One-click copy to clipboard
- Download result as `.txt`

### Supported formats

PNG, JPG/JPEG, TIFF, WEBP, BMP — up to 32 MB.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/status` | Model loading state (`loading` / `ready` / `error`) |
| POST | `/process` | Run OCR. Multipart form field: `image` (file). Returns `{"text": "..."}` |
| POST | `/download` | Download text as `.txt`. Form field: `text` (string) |

Example curl call:

```bash
curl -X POST http://localhost:5000/process \
  -F "image=@my_image.jpg" | jq .text
```

## Running the CLI batch processor

Place input images in the `images/` directory, then optionally pre-process them:

```bash
source .venv/bin/activate

# Optional — resize, denoise and enhance contrast before OCR
python preprocess_images.py          # reads images/, writes processed_images/

# Run OCR
python ocr_arabic.py                 # reads processed_images/, writes output.txt
```

`ocr_arabic.py` defaults to `processed_images/` as its input directory. To use raw images directly, change `IMAGE_DIR = "images"` at the top of the file.

Results are written to `output.txt` with one section per image.

## Image preprocessing

Both `app.py` and `preprocess_images.py` apply the same pipeline before feeding images to the model:

1. Resize so the longest side is at most 1000 px (keeps visual token count manageable)
2. Convert to grayscale
3. NLM denoising (`h=5`)
4. CLAHE contrast enhancement (`clipLimit=1.5`, `tileGridSize=8×8`)

## Running as a systemd service

Copy the unit file and enable it:

```bash
sudo cp arabic-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arabic-ocr
```

The service runs as the `hermes` user, uses the `.venv` Python interpreter, and restarts automatically on failure. Logs go to the system journal:

```bash
journalctl -u arabic-ocr -f
```

To stop or restart:

```bash
sudo systemctl stop arabic-ocr
sudo systemctl restart arabic-ocr
```

## Troubleshooting

**CUDA not available**: Verify NVIDIA drivers and CUDA toolkit are installed. In Docker, ensure `nvidia-container-toolkit` is installed on the host.

**Out of memory (OOM)**: The model is large. Set the allocator environment variable and close other GPU-intensive applications:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Project structure

```
arabic-ocr/
├── app.py                  # Flask web app + background model loader
├── ocr_arabic.py           # CLI batch processor
├── preprocess_images.py    # Standalone image preprocessing script
├── requirements.txt        # Python dependencies (CUDA 12.1 PyTorch)
├── setup_arabic_ocr_env.sh # One-shot environment setup script
├── arabic-ocr.service      # systemd unit file
├── templates/
│   └── index.html          # Web UI (RTL, dark theme)
├── images/                 # Drop raw input images here (not tracked)
└── processed_images/       # Output of preprocess_images.py (not tracked)
```

## Model notes

The model checkpoint (`sherif1313/Arabic-English-handwritten-OCR-v3`) is missing `lm_head.weight` in its saved state. Both `app.py` and `ocr_arabic.py` work around this by tying the output embedding weights to the input embedding matrix after loading — this is intentional and safe for this checkpoint.

The prompt sent to the model instructs it to extract only the written text without translation or explanation:

> اقرأ النص العربي الموجود في الصورة واستخرج النص فقط. لا تشرح. لا تترجم. لا تضف أي شيء غير النص المكتوب.
