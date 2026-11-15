FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies: ffmpeg, libsm6, libxext6, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first to keep image lightweight and avoid GPU bloat
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Set up user permissions for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user && \
    mkdir -p /app/work /app/data && \
    chown -R user:user /app
USER user

EXPOSE 7860

# Run FastAPI via Uvicorn on Hugging Face port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
