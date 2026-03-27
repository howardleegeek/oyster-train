#!/bin/bash
# setup-phone.sh — Run this in Termux on an Android phone to set up
# the Oyster federated training client.
#
# Usage:
#   pkg install git
#   git clone https://github.com/howardleegeek/oyster-train.git
#   cd oyster-train
#   bash setup-phone.sh
#
# Then join the network:
#   python3 join.py --server <server-ip>:8080

set -e

echo "=== Oyster Phone Training Setup ==="
echo "Platform: $(uname -m)"
echo "Termux: $(command -v termux-info >/dev/null && echo yes || echo no)"

# 1. System packages
echo ""
echo "[1/4] Installing system packages..."
pkg update -y
pkg install -y python git cmake ninja clang

# 2. Python venv
echo ""
echo "[2/4] Creating Python environment..."
python3 -m venv .venv 2>/dev/null || python3 -m ensurepip
source .venv/bin/activate 2>/dev/null || true

# 3. Python packages
echo ""
echo "[3/4] Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flwr einops numpy msgpack-python pyyaml

# Optional: camera + sensors
echo ""
echo "[4/4] Installing Termux:API (optional, for camera data)..."
pkg install -y termux-api 2>/dev/null || echo "Termux:API not available — will use synthetic data"

# Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')

from models.lewm_config import get_simulation_config
from models.lewm_loader import load_lewm_model
cfg = get_simulation_config()
model = load_lewm_model(cfg)
n = sum(p.numel() for p in model.parameters())
print(f'LeWM model: {n/1e6:.1f}M params')

# Quick forward test
pixels = torch.randn(1, 8, 3, 96, 96)
actions = torch.randn(1, 8, 6)
out = model(pixels, actions)
print(f'Forward pass: loss={out[\"loss\"].item():.4f}')
print('Setup complete!')
"

echo ""
echo "=== Ready! ==="
echo "Join the network with:"
echo "  python3 join.py --server <server-ip>:8080"
echo ""
echo "With camera data:"
echo "  python3 join.py --server <server-ip>:8080 --data camera"
