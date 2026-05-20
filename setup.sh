#!/usr/bin/env bash
# Setup script for Arabic Handwritten OCR
# Tested on: Linux Mint / Ubuntu 22.04+
# Requirements: NVIDIA GPU with driver >= 525 (CUDA 12.x support)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── Helpers ───────────────────────────────────────────────────────────────────

step() { echo; echo "==> $*"; }
ok()   { echo "    OK: $*"; }
warn() { echo "    WARN: $*"; }

# ── 1. System packages ────────────────────────────────────────────────────────
# We check capabilities rather than dpkg so this works whether packages came
# from apt, conda, or were already present on the system.

step "Checking system packages..."

MISSING_APT=()

command -v curl &>/dev/null          || MISSING_APT+=(curl)
python3 -c "import venv" &>/dev/null || MISSING_APT+=(python3-venv)

# OpenCV needs libGL and libglib at runtime.
# Cache ldconfig output in a variable so pipefail doesn't trigger on ldconfig's
# own exit code; only grep's exit code determines the || branch.
_ldcache=$(ldconfig -p 2>/dev/null || true)
echo "$_ldcache" | grep -q "libGL.so.1"    || MISSING_APT+=(libgl1)
echo "$_ldcache" | grep -q "libglib-2.0"   || MISSING_APT+=(libglib2.0-0)
unset _ldcache

if [ "${#MISSING_APT[@]}" -gt 0 ]; then
    echo "    Installing via apt: ${MISSING_APT[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends "${MISSING_APT[@]}"
fi
ok "System packages"

# ── 2. uv ─────────────────────────────────────────────────────────────────────

step "Checking uv..."

if ! command -v uv &>/dev/null && ! [ -x "$HOME/.local/bin/uv" ]; then
    echo "    Downloading and installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Make uv available in this shell session even if .bashrc hasn't been sourced
export PATH="$HOME/.local/bin:$PATH"
ok "uv $(uv --version)"

# ── 3. Virtual environment ────────────────────────────────────────────────────

step "Creating virtual environment in $VENV..."

# Remove a broken venv if it exists; start clean
[ -d "$VENV" ] && rm -rf "$VENV"

# Python 3.13 is NOT supported by PyTorch's cu121 wheel index.
# Use 3.12 (uv downloads it automatically if missing) or fall back to 3.10.
uv venv "$VENV" --python ">=3.10,<3.13"
PYTHON="$VENV/bin/python"
ok "Python $($PYTHON --version)"

# ── 4. PyTorch with CUDA 12.1 ─────────────────────────────────────────────────
#
# PyTorch is installed separately with an explicit CUDA index URL.
# It is NOT listed in pyproject.toml to prevent package managers from
# silently pulling in the CPU-only or wrong-CUDA build from PyPI.

step "Installing PyTorch (CUDA 12.1)..."
uv pip install --python "$PYTHON" \
    "torch" "torchvision" \
    --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch installed"

# ── 5. Remaining dependencies ─────────────────────────────────────────────────

step "Installing project dependencies..."
uv pip install --python "$PYTHON" \
    "transformers>=4.49.0" \
    "accelerate>=0.24.0" \
    "pillow>=10.0.0" \
    "qwen-vl-utils>=0.0.2" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "safetensors>=0.4.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "bitsandbytes>=0.41.0"
ok "Dependencies installed"

# ── 6. Verify ─────────────────────────────────────────────────────────────────

step "Verifying installation..."
"$PYTHON" - <<'PY'
import torch, transformers, cv2, PIL, bitsandbytes
print(f"    torch        {torch.__version__}")
print(f"    transformers {transformers.__version__}")
print(f"    opencv       {cv2.__version__}")
print(f"    CUDA         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU          {torch.cuda.get_device_name(0)}")
else:
    print("    WARN: CUDA not available — OCR will run on CPU (slow)")
PY

# ── Done ──────────────────────────────────────────────────────────────────────

echo
echo "============================================================"
echo "Setup complete."
echo ""
echo "Activate:   source .venv/bin/activate"
echo "Preprocess: python preprocess_images.py   (optional)"
echo "Run OCR:    python ocr_arabic.py"
echo "============================================================"
