FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim nano \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch and torchvision are provided by the base image.
# Strip the index-url directives and torch pins before installing the rest
# so pip doesn't re-download 700 MB of PyTorch that's already present.
COPY requirements.txt .
RUN grep -vE '^--index-url|^--extra-index-url|^torch==|^torchvision==' requirements.txt \
    > /tmp/req-no-torch.txt && \
    pip install --no-cache-dir -r /tmp/req-no-torch.txt

CMD ["bash"]
