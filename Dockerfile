FROM python:3.10-slim

# ---- system deps + NVIDIA CUDA apt repo keyring ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repo (Ubuntu 24.04) so we can install cuBLAS/cuDNN runtime libs
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -f cuda-keyring_1.1-1_all.deb

# Install CUDA 12 runtime libs required by onnxruntime-gpu CUDA EP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcublaslt12 \
    libcublas12 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /app

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: quick visibility into whether ORT sees CUDA at build time
# (This doesn't guarantee runtime GPU access, but confirms the CUDA EP can load its libs.)
RUN python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())" || true

# Preload models to avoid race conditions at runtime
COPY src/preload_models.py .
RUN python3 preload_models.py

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "src/run_analysis.py", "cameras.yml"]