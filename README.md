# Arabic Handwritten OCR

A self-hosted OCR service for extracting text from Arabic handwritten (and printed) images. It uses [`sherif1313/Arabic-English-handwritten-OCR-v3`](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3), a Qwen2.5-VL fine-tune, and ships a Flask web UI, a REST API, and a CLI batch processor.

---

## Hardware requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA, ≥ 12 GB VRAM | RTX 2080 Ti 22 GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free (model cache + deps) | — |
| CPU | Any x86-64 | — |

**GPU is required.** CPU inference is viable for testing but takes several minutes per image instead of a few seconds.

### VRAM breakdown (observed, float16, no quantization)

| Budget item | Approximate size |
|---|---|
| Model weights (float16) | ~8.5 GB |
| KV cache + activations | ~0.5 GB |
| CUDA context + overhead | ~0.5 GB |
| **Total at inference** | **~9–10 GB** |

Cards known to work: RTX 2080 Ti (22 GB), RTX 3090, RTX 4090, A5000, V100 32 GB, A100.  
Cards that are too small: anything with less than 12 GB VRAM (the model alone requires ~8.5 GB).

---

## Software requirements (host machine)

| Requirement | Version |
|---|---|
| OS | Linux (Ubuntu 22.04+ recommended) |
| NVIDIA driver | ≥ 525 (supports CUDA 12.x) |
| Python | 3.10 – 3.12 (3.13 not supported by PyTorch) |
| uv | Any recent version (installed by `setup.sh` if missing) |
| CUDA toolkit | Not required on host — PyTorch ships its own CUDA runtime |

Verify your driver version:

```bash
nvidia-smi
```

The `CUDA Version` shown must be 12.0 or higher. If the driver is older, update it via your distro's package manager or from [nvidia.com](https://www.nvidia.com/drivers).

---

## Installation

### Option 1 — Local setup (recommended)

Clone the repo and run the setup script. It creates a `.venv`, installs PyTorch 2.6.0 with CUDA 12.4 support, and all other dependencies:

```bash
git clone git@github.com:sayfjawad/arabic-handwritten-to-text.git
cd arabic-handwritten-to-text
bash setup.sh
```

The script ends with a verification step that prints PyTorch version, OpenCV version, and confirms CUDA is detected. A line like `CUDA True` and `GPU NVIDIA ...` means it worked.

> **Why the CUDA wheel is pinned:** `requirements.txt` sets the PyTorch CUDA index as the primary source. Without this, `pip`/`uv` picks the CPU-only build from PyPI, which is lighter but runs ~30× slower.

#### Manual install (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 \
            --extra-index-url https://pypi.org/simple \
            torch==2.6.0+cu124 torchvision==0.21.0+cu124
pip install -r requirements.txt
```

### Option 2 — VS Code Dev Container

Requirements: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host.

1. Open the repo folder in VS Code.
2. When prompted, click **Reopen in Container** (or `F1` → `Dev Containers: Reopen in Container`).
3. The container builds automatically. The first build downloads the PyTorch base image (~5 GB) and all Python dependencies — allow 10–15 minutes.

The container uses `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` as its base and passes `--gpus=all` so the GPU is available inside. After the build, a post-create command confirms CUDA is detected.

---

## Running

### Development (foreground)

```bash
source .venv/bin/activate
python app.py
```

The server starts at `http://localhost:5000`. The model loads in the background — the UI shows an amber dot while loading (~30–90 s on the first run; faster on subsequent runs once HuggingFace has cached the weights locally). A green dot means the model is ready.

### Production (systemd service)

Install and enable the included unit file:

```bash
sudo cp arabic-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arabic-ocr
```

The service runs as the current user, uses the `.venv` interpreter, binds to `0.0.0.0:5000`, and restarts automatically on failure.

**Edit the service file first** if your username or install path differs from the defaults — check the `User=` and `WorkingDirectory=` lines.

View logs:

```bash
journalctl -u arabic-ocr -f
```

Check model loading state:

```bash
curl http://localhost:5000/status
# {"status": "loading"}   -- still loading
# {"status": "ready"}     -- model ready
# {"status": "error", "message": "..."} -- something went wrong
```

---

## Web UI

Open `http://localhost:5000` in a browser.

- Drag-and-drop or browse to upload an image
- Solve the CAPTCHA (prevents abuse on shared installations)
- The result appears in an RTL Arabic text area
- Copy to clipboard or download as `.txt`
- Rate the result (thumbs up / thumbs down) or paste a correction — this builds a fine-tuning dataset over time

**Supported formats:** PNG, JPG/JPEG, TIFF, WEBP, BMP — up to 32 MB.

### Training readiness dashboard

Visit `/dashboard` to see:

- Number of usable training pairs and progress toward milestones (50 / 100 / 200 / 500)
- Feedback breakdown (confirmed correct, human corrections, weak signal)
- A **Notify Admin** button that logs a notification and optionally sends an email

To enable email notifications, set these environment variables in the systemd service file or your shell:

```
ADMIN_EMAIL=you@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=user
SMTP_PASS=pass
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/dashboard` | Training readiness dashboard |
| `GET` | `/status` | Model state: `loading` / `ready` / `error` |
| `GET` | `/api/readiness` | Training readiness stats as JSON |
| `GET` | `/captcha` | Fetch a fresh CAPTCHA image (sets session cookie) |
| `POST` | `/process` | Run OCR. Multipart form: `image` (file), `captcha` (string). Returns `{"text": "...", "job_id": "..."}` |
| `POST` | `/feedback` | Submit rating. Form: `job_id`, `rating` (`up`/`down`), `corrected_text` (optional) |
| `POST` | `/notify-admin` | Record a fine-tuning notification (6 h rate limit) |
| `POST` | `/download` | Download text as `.txt`. Form field: `text` |

Example — OCR via curl (requires a valid CAPTCHA session; easier from the browser UI for manual use):

```bash
curl http://localhost:5000/status
```

---

## CLI batch processor

Place input images in the `images/` directory, then run:

```bash
source .venv/bin/activate

# Optional — resize, denoise, enhance contrast
python preprocess_images.py    # reads images/, writes processed_images/

# Run OCR
python ocr_arabic.py           # reads processed_images/, writes output.txt
```

To use raw images without preprocessing, edit the top of `ocr_arabic.py`:

```python
IMAGE_DIR = "images"   # instead of "processed_images"
```

Results are written to `output.txt` with one section per image.

---

## Automated deployment (CD pipeline)

Changes are deployed automatically to the running service after being merged to `main`.

### How it works

```
feature branch → PR → CI must pass → merge to main → within 5 min: deployed & health-checked
```

**CI gate (GitHub Actions)** — `.github/workflows/ci.yml` runs on every PR and push to `main`:
- Syntax-checks all `.py` files
- Verifies critical imports resolve (`AutoProcessor`, `Qwen2_5_VLForConditionalGeneration`, etc.)
- Confirms `requirements.txt` pins a CUDA torch build (prevents the CPU-only regression)

Configure GitHub branch protection to require this workflow to pass before merging.

**Deployment (systemd timer)** — `arabic-ocr-deploy.timer` fires every 5 minutes on the server:
1. Fetches `origin/main` — exits immediately if already up to date
2. Pulls new commits
3. Rsyncs code files into the service directory (never touches `.venv` or runtime state)
4. Re-runs `setup.sh` if `requirements.txt` or `setup.sh` changed
5. Restarts the service
6. Polls `/status` every 10 s for up to 10 min — reports failure if the model does not reach `ready`

To install the timer on a new machine:

```bash
# Copy unit files
sudo cp arabic-ocr.service /etc/systemd/system/
sudo cp arabic-ocr-deploy.service /etc/systemd/system/    # if shipping these files
sudo cp arabic-ocr-deploy.timer /etc/systemd/system/

# Allow the service user to restart the OCR service without a password
echo "$USER ALL=(ALL) NOPASSWD: /bin/systemctl restart arabic-ocr" \
  | sudo tee /etc/sudoers.d/arabic-ocr-deploy
sudo chmod 440 /etc/sudoers.d/arabic-ocr-deploy

sudo systemctl daemon-reload
sudo systemctl enable --now arabic-ocr-deploy.timer
```

To trigger a deploy manually at any time:

```bash
/path/to/arabic-handwritten-to-text/deploy.sh
```

---

## Project structure

```
arabic-handwritten-to-text/
├── app.py                        # Flask web app + background model loader
├── ocr_arabic.py                 # CLI batch processor
├── preprocess_images.py          # Standalone image preprocessing script
├── export_dataset.py             # Export archive data as a fine-tuning dataset
├── requirements.txt              # Python dependencies (CUDA 12.4 PyTorch pinned)
├── setup.sh                      # One-shot environment setup script
├── deploy.sh                     # CD script: pull → sync → restart → health-check
├── arabic-ocr.service            # systemd unit for the Flask service
├── templates/
│   ├── index.html                # Web UI (RTL, dark theme)
│   └── dashboard.html            # Training readiness dashboard
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: syntax, imports, CUDA pin check
├── .devcontainer/
│   ├── Dockerfile                # Dev Container image (pytorch base + project deps)
│   └── devcontainer.json         # VS Code Dev Container config (--gpus=all)
├── images/                       # Drop raw input images here (not tracked by git)
├── processed_images/             # Output of preprocess_images.py (not tracked)
├── archive/                      # Per-job artifacts for debugging and fine-tuning (not tracked)
│   └── YYYY-MM-DD/
│       └── <job_id>/
│           ├── input.<ext>       # Original uploaded image
│           ├── processed.png     # Preprocessed image fed to the model
│           ├── meta.json         # Timing, model, feedback metadata
│           └── corrected.txt     # Human correction (only when text was changed)
└── uploads/                      # Temporary upload staging (not tracked)
```

---

## Troubleshooting

**`CUDA available: False` after setup**
The CPU-only PyTorch build was installed. Fix:
```bash
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.6.0+cu124 torchvision==0.21.0+cu124 --force-reinstall
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**`ModuleNotFoundError: Could not import module 'AutoProcessor'`**
`transformers` 5.x was installed; it requires a PyTorch version not yet available as a stable CUDA build. Pin it:
```bash
pip install "transformers>=4.49.0,<5.0.0"
```

**Out of memory (OOM) during inference**
Close other GPU-intensive applications and set the CUDA allocator hint:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
If you have multiple GPUs, the service automatically picks the one with the most free VRAM.

**Model takes very long to load on first run**
The HuggingFace hub downloads ~9 GB of weights. This is a one-time download; subsequent starts read from the local cache (`~/.cache/huggingface`).

**Service crashes immediately after a deploy**
Check the journal for the real error:
```bash
journalctl -u arabic-ocr -n 50
```
The deploy timer will log a failure and stop — it does not loop-restart on a broken deployment.

---

## Model notes

The checkpoint (`sherif1313/Arabic-English-handwritten-OCR-v3`) is a Qwen2.5-VL fine-tune for Arabic handwriting. It intentionally omits `lm_head.weight` from the saved state; both `app.py` and `ocr_arabic.py` work around this by tying the output embedding weights to the input embedding matrix at load time. This is safe and expected for this checkpoint.

The inference prompt instructs the model to return only the written text, without translation or explanation:

> اقرأ النص العربي الموجود في الصورة واستخرج النص فقط. لا تشرح. لا تترجم. لا تضف أي شيء غير النص المكتوب.
