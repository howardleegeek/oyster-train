#!/usr/bin/env python3
"""
join.py — Run on any device (phone/laptop/server) to join the Oyster
federated LeWM training network.

Usage:
    python3 join.py --server 100.95.165.3:8080
    python3 join.py --server 192.168.1.100:8080 --device cpu --steps 50
    python3 join.py --server <ip>:8080 --data camera   # Android + Termux:API
"""
import argparse
import gc
import logging
import os
import platform
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oyster-join")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_platform() -> dict:
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "is_android": os.path.exists("/data/data/com.termux"),
        "is_arm": platform.machine() in ("aarch64", "arm64", "armv8l", "armv7l"),
        "ram_gb": 0,
    }
    try:
        if info["is_android"]:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        else:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


def get_config_for_device(env: dict):
    from models.lewm_config import get_ubs1_config, get_simulation_config, get_gpu_config
    if env.get("is_android") or env.get("is_arm"):
        if env.get("ram_gb", 0) >= 6:
            logger.info("Config: UBS1 (6GB+ ARM)")
            return get_ubs1_config()
        else:
            logger.info("Config: simulation (low-RAM)")
            return get_simulation_config()
    elif torch.cuda.is_available():
        logger.info("Config: GPU")
        return get_gpu_config()
    else:
        logger.info("Config: simulation (CPU)")
        return get_simulation_config()


class DataSource:
    """Generates training data — synthetic or from Android camera."""

    def __init__(self, cfg, source="synthetic", num_samples=200):
        self.cfg = cfg
        self.source = source
        self.num_samples = num_samples
        self.img_size = cfg.encoder.image_size
        self.seq_len = cfg.data.sequence_length
        self.act_dim = cfg.data.action_dim * cfg.data.frameskip

    def get_dataloader(self, batch_size) -> DataLoader:
        if self.source == "camera":
            pixels, actions = self._try_camera(batch_size)
        else:
            pixels, actions = self._synthetic(batch_size)
        return DataLoader(
            TensorDataset(pixels, actions),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )

    def _synthetic(self, batch_size):
        n = max(self.num_samples // self.seq_len, batch_size * 2)
        return (
            torch.randn(n, self.seq_len, 3, self.img_size, self.img_size),
            torch.randn(n, self.seq_len, self.act_dim),
        )

    def _try_camera(self, batch_size):
        """Try Termux camera, fall back to synthetic."""
        try:
            return self._capture_camera(batch_size)
        except Exception as e:
            logger.warning(f"Camera unavailable ({e}), using synthetic")
            return self._synthetic(batch_size)

    def _capture_camera(self, batch_size):
        import subprocess
        import tempfile
        import json
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        n_seqs = max(batch_size * 2, 8)
        frames = []
        for _ in range(n_seqs * self.seq_len):
            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                subprocess.run(["termux-camera-photo", "-c", "0", f.name],
                               check=True, timeout=5)
                img = Image.open(f.name).convert("RGB")
                frames.append(transform(img))

        pixels = torch.stack(frames).view(n_seqs, self.seq_len, 3, self.img_size, self.img_size)

        # IMU sensor data as actions
        try:
            result = subprocess.run(
                ["termux-sensor", "-s", "accelerometer,gyroscope", "-n", "1"],
                capture_output=True, text=True, timeout=3,
            )
            sensor = json.loads(result.stdout)
            accel = sensor.get("accelerometer", {}).get("values", [0, 0, 0])
            gyro = sensor.get("gyroscope", {}).get("values", [0, 0, 0])
            action = torch.tensor(accel + gyro, dtype=torch.float32)
            actions = action.unsqueeze(0).unsqueeze(0).expand(n_seqs, self.seq_len, -1)
        except Exception:
            actions = torch.zeros(n_seqs, self.seq_len, self.act_dim)

        return pixels, actions


class OysterClient:
    """Flower-compatible client for federated LeWM training."""

    def __init__(self, cfg, device="cpu", data_source="synthetic"):
        from models.lewm_loader import load_lewm_model, extract_delta
        from compressor import CompressionPipeline

        self.cfg = cfg
        self.device = device
        self.model = load_lewm_model(cfg).to(device)
        self.extract_delta = extract_delta
        self.compression = CompressionPipeline(k_ratio=cfg.federation.compression_k_ratio)
        self.data = DataSource(cfg, source=data_source)

        n = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {n/1e6:.1f}M params on {device}")

    def get_parameters(self, config) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params: List[np.ndarray]):
        for p, v in zip(self.model.parameters(), params):
            p.data = torch.tensor(v, dtype=p.dtype, device=self.device)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        before = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

        steps = config.get("local_steps", self.cfg.training.local_steps)
        self.model.train()
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        loader = self.data.get_dataloader(self.cfg.training.batch_size)
        total_loss, step = 0.0, 0

        for epoch in range(max(1, steps // max(len(loader), 1) + 1)):
            for px, act in loader:
                if step >= steps:
                    break
                out = self.model(px.to(self.device), act.to(self.device))
                out["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                opt.step()
                opt.zero_grad()
                total_loss += out["loss"].item()
                step += 1
                if step % 10 == 0:
                    logger.info(f"  step {step}/{steps}: loss={out['loss'].item():.4f}")

        after = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        delta = self.extract_delta(before, after)
        arrays = [delta[k].numpy() for k in sorted(delta.keys())]

        avg = total_loss / max(step, 1)
        logger.info(f"Round done: {step} steps, loss={avg:.4f}")
        del before, after, delta
        gc.collect()
        return arrays, len(loader.dataset), {"loss": avg, "steps": step}

    def run_assessment(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loader = self.data.get_dataloader(self.cfg.training.batch_size)
        total, n = 0.0, 0
        with torch.no_grad():
            for px, act in loader:
                out = self.model(px.to(self.device), act.to(self.device))
                total += out["loss"].item() * px.size(0)
                n += px.size(0)
        avg = total / max(n, 1)
        return avg, n, {"loss": avg}


def make_flower_client(client: OysterClient):
    """Wrap OysterClient as a Flower NumPyClient."""
    import flwr as fl

    class _FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return client.get_parameters(config)
        def fit(self, parameters, config):
            return client.fit(parameters, config)
        def evaluate(self, parameters, config):
            return client.run_assessment(parameters, config)

    return _FlowerClient()


def main():
    parser = argparse.ArgumentParser(description="Join Oyster federated training network")
    parser.add_argument("--server", required=True, help="Flower server (ip:port)")
    parser.add_argument("--device", default="auto", help="cpu/cuda/mps/auto")
    parser.add_argument("--data", default="synthetic", choices=["synthetic", "camera"])
    parser.add_argument("--steps", type=int, default=None, help="Override local steps")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    env = detect_platform()
    logger.info(f"Platform: {env['platform']} {env['machine']}, "
                f"Python {env['python']}, PyTorch {env['torch']}")
    if env["is_android"]:
        logger.info(f"Android (Termux), RAM: {env['ram_gb']}GB")
    elif env["ram_gb"]:
        logger.info(f"RAM: {env['ram_gb']}GB")

    device = args.device if args.device != "auto" else detect_device()
    cfg = get_config_for_device(env)
    if args.steps:
        cfg.training.local_steps = args.steps
    if args.batch:
        cfg.training.batch_size = args.batch

    logger.info(f"Encoder: {cfg.encoder.backbone}, ~{cfg.estimated_params_m:.1f}M params, "
                f"~{cfg.estimated_memory_mb:.0f}MB memory")

    import flwr as fl
    client = OysterClient(cfg, device=device, data_source=args.data)
    logger.info(f"Connecting to {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=make_flower_client(client),
    )


if __name__ == "__main__":
    main()
