# Training image for MarsWorld PI-JEPA.
# Uses an official PyTorch CUDA runtime so GPU pods work out of the box.
# For CPU-only clusters, swap the base image for `python:3.11-slim` and install
# the CPU wheel of torch instead.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.source="MarsWorld PI-JEPA"
LABEL org.opencontainers.image.description="Physics-Informed JEPA for Martian climate"

WORKDIR /app

# Install package deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source and install the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -e .

# Default: train using the baked-in default config, auto-generating data if the
# mounted volume is empty. Override `command`/`args` in the Job spec as needed.
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/default.yaml", "--auto-generate"]
