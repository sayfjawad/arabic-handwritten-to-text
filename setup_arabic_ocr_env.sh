#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/arabic-ocr}"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================================"
echo "Arabic Handwritten OCR environment setup"
echo "Project dir: $PROJECT_DIR"
echo "Venv dir:    $VENV_DIR"
echo "============================================================"

mkdir -p "$PROJECT_DIR/images"
cd "$PROJECT_DIR"

echo
echo "[1/7] Checking Python..."
python3 --version

echo
echo "[2/7] Creating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists: $VENV_DIR"
fi

echo
echo "[3/7] Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "[4/7] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel packaging

echo
echo "[5/7] Installing PyTorch CUDA 12.1 build..."
# We use --extra-index-url so it can fall back to PyPI if a specific component is missing in the CUDA index.
# We also remove torchaudio as it's not required for this OCR project and often lacks 3.13 + CUDA builds.
python -m pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

echo
echo "[6/7] Installing OCR/model dependencies..."
python -m pip install --upgrade \
  transformers \
  accelerate \
  pillow \
  qwen-vl-utils \
  sentencepiece \
  protobuf \
  safetensors \
  opencv-python \
  bitsandbytes

echo
echo "[7/7] Writing requirements.txt..."
cat > requirements.txt <<'REQ'
--extra-index-url https://download.pytorch.org/whl/cu121
torch
torchvision
transformers
accelerate
pillow
qwen-vl-utils
sentencepiece
protobuf
safetensors
opencv-python
bitsandbytes
REQ

echo
echo "============================================================"
echo "Environment check"
echo "============================================================"

python - <<'PY'
import torch
import transformers
import cv2
import PIL
import bitsandbytes

print("Python environment OK")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("opencv:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available. OCR will be slow or may fail because the model is large.")
PY

echo
echo "============================================================"
echo "Setup complete."
echo
echo "Note about Conda Error:"
echo "If you see 'anaconda-auth (cannot import name RootModel)', run:"
echo "  conda update -n base anaconda-auth"
echo "  or: /home/sayf/miniconda3/bin/python -m pip install --upgrade pydantic"
echo
echo "Use:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python ocr_arabic.py"
echo "============================================================"
