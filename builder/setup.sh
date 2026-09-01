#!/bin/bash
set -e

# Rendszerfüggőségek
apt-get update && apt-get install -y ffmpeg git

# Python csomagok
pip install --upgrade pip
pip install runpod torch torchvision diffusers transformers accelerate safetensors pillow requests imageio[ffmpeg]