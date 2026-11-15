#!/bin/bash
set -e

echo "=== Setting up ClipForge Codespace Environment ==="

# 1. System packages
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 curl unzip

# 2. Install Deno 2.9.8
echo "Installing Deno 2.9.8..."
curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh -s v2.9.8 || true

# 3. Install lightweight locked dependencies
if [ -f "requirements-cpu.lock" ]; then
    echo "Installing from requirements-cpu.lock..."
    pip install -r requirements-cpu.lock || pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "Codespace environment setup completed successfully."
