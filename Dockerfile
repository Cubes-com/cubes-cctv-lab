FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload models to avoid race conditions at runtime
COPY src/preload_models.py .
RUN python3 preload_models.py

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command (overridden in docker-compose)
CMD ["python3", "src/run_analysis.py", "cameras.yml"]
