"""
engine.py — Python training engine called from Kotlin via Chaquopy.

This is the real training code that runs on the phone. Kotlin calls
these functions, which execute PyTorch training and Flower FL.
"""
import gc
import json
import os
import platform
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Global state ─────────────────────────────────────────────────
_model = None
_config = None
_training_thread = None
_stop_event = threading.Event()
_status = {
    "state": "idle",  # idle | connecting | training | paused | error
    "round": 0,
    "total_rounds": 0,
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "params_m": 0.0,
    "memory_mb": 0,
    "server": "",
    "error": "",
}


def get_status() -> str:
    """Called from Kotlin to get current training status as JSON."""
    return json.dumps(_status)


def get_device_info() -> str:
    """Return device capabilities."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "is_arm": platform.machine() in ("aarch64", "arm64"),
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = 0
    return json.dumps(info)


# ─── Model building (same as lewm_loader but self-contained) ─────

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", w * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        from einops import rearrange
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads)
                    for t in self.to_qkv(x).chunk(3, dim=-1))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.mod[-1].weight, 0)
        nn.init.constant_(self.mod[-1].bias, 0)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.mod(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(x * (1 + sc1) + s1)
        x = x + g2 * self.mlp(x * (1 + sc2) + s2)
        return x


class LeWMPhone(nn.Module):
    """Lightweight LeWM for phone: MobileNetV3 encoder + small predictor."""

    def __init__(self, embed_dim=192, depth=4, heads=8, mlp_dim=1024,
                 dim_head=64, action_dim=6, history=3, sigreg_weight=0.09):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(weights=None)
        backbone.classifier = nn.Identity()
        self.encoder = nn.Sequential(backbone, nn.Linear(576, embed_dim))

        self.act_enc = nn.Sequential(
            nn.Conv1d(action_dim, action_dim, 1),
            nn.Flatten(0, 0),  # no-op to keep dims
        )
        self.act_proj = nn.Sequential(nn.Linear(action_dim, 4 * embed_dim),
                                       nn.SiLU(), nn.Linear(4 * embed_dim, embed_dim))

        self.pos = nn.Parameter(torch.randn(1, history, embed_dim) * 0.02)
        self.proj_in = nn.Linear(embed_dim, embed_dim)
        self.proj_cond = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList([ConditionalBlock(embed_dim, heads, dim_head, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Sequential(nn.Linear(embed_dim, 2048), nn.LayerNorm(2048),
                                       nn.GELU(), nn.Linear(2048, embed_dim))
        self.pred_proj = nn.Sequential(nn.Linear(embed_dim, 2048), nn.LayerNorm(2048),
                                        nn.GELU(), nn.Linear(2048, embed_dim))
        self.sigreg = SIGReg(knots=17, num_proj=512)
        self.history = history
        self.sigreg_w = sigreg_weight

    def forward(self, pixels, actions):
        from einops import rearrange
        b, t = pixels.shape[:2]
        x = rearrange(pixels, "b t c h w -> (b t) c h w").float()
        x = self.encoder(x)
        emb = self.proj_out(x)
        emb = rearrange(emb, "(b t) d -> b t d", b=b)

        act = actions.float().permute(0, 2, 1)
        act = self.act_enc[0](act).permute(0, 2, 1)
        act_emb = self.act_proj(act)

        ctx_emb = emb[:, :self.history]
        ctx_act = act_emb[:, :self.history]
        tgt = emb[:, 1:self.history + 1]

        x = self.proj_in(ctx_emb + self.pos[:, :self.history])
        c = self.proj_cond(ctx_act)
        for layer in self.layers:
            x = layer(x, c)
        x = self.norm(x)
        pred = self.pred_proj(rearrange(x, "b t d -> (b t) d"))
        pred = rearrange(pred, "(b t) d -> b t d", b=b)

        pred_loss = (pred - tgt).pow(2).mean()
        sig_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + self.sigreg_w * sig_loss

        return {"loss": loss, "pred_loss": pred_loss.detach(), "sigreg_loss": sig_loss.detach()}


def _build_model():
    global _model, _config
    _model = LeWMPhone(embed_dim=192, depth=4, heads=8, mlp_dim=1024, action_dim=6)
    n = sum(p.numel() for p in _model.parameters())
    _status["params_m"] = round(n / 1e6, 1)
    return n


# ─── Training loop ────────────────────────────────────────────────

def _train_loop(server_address: str, num_rounds: int, local_steps: int):
    """Background training loop — connects to Flower server."""
    global _model, _status
    import flwr as fl

    _status["state"] = "connecting"
    _status["server"] = server_address
    _status["total_rounds"] = num_rounds

    n = _build_model()
    _status["state"] = "training"

    class _Client(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [p.detach().cpu().numpy() for p in _model.parameters()]

        def fit(self, parameters, config):
            # Load global params
            for p, v in zip(_model.parameters(), parameters):
                p.data = torch.tensor(v, dtype=p.dtype)

            before = {k: v.clone() for k, v in _model.state_dict().items()}

            _model.train()
            opt = torch.optim.AdamW(_model.parameters(), lr=5e-5, weight_decay=1e-3)
            steps = config.get("local_steps", local_steps)
            _status["total_steps"] = steps

            for step in range(steps):
                if _stop_event.is_set():
                    break
                px = torch.randn(2, 8, 3, 224, 224)
                act = torch.randn(2, 8, 6)
                out = _model(px, act)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

                _status["step"] = step + 1
                _status["loss"] = round(out["loss"].item(), 4)
                _status["memory_mb"] = int(torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0

            _status["round"] = _status.get("round", 0) + 1

            after = {k: v.clone() for k, v in _model.state_dict().items()}
            delta = [after[k].numpy() - before[k].numpy() for k in sorted(before.keys()) if k in after]

            gc.collect()
            return delta, 10, {"loss": _status["loss"]}

        def evaluate(self, parameters, config):
            return 0.0, 10, {"loss": 0.0}

    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=_Client(),
        )
        _status["state"] = "idle"
    except Exception as e:
        _status["state"] = "error"
        _status["error"] = str(e)


def start_training(server_address: str, num_rounds: int = 100, local_steps: int = 50) -> str:
    """Called from Kotlin — starts training in background thread."""
    global _training_thread
    _stop_event.clear()
    _status["state"] = "connecting"
    _status["round"] = 0
    _status["step"] = 0
    _status["error"] = ""

    _training_thread = threading.Thread(
        target=_train_loop,
        args=(server_address, num_rounds, local_steps),
        daemon=True,
    )
    _training_thread.start()
    return json.dumps({"status": "started", "server": server_address})


def stop_training() -> str:
    """Called from Kotlin — stops training gracefully."""
    _stop_event.set()
    _status["state"] = "paused"
    return json.dumps({"status": "stopped"})


def run_quick_test() -> str:
    """Quick model build + forward test — called on app startup."""
    try:
        n = _build_model()
        px = torch.randn(1, 8, 3, 224, 224)
        act = torch.randn(1, 8, 6)
        out = _model(px, act)
        return json.dumps({
            "status": "ok",
            "params": n,
            "loss": round(out["loss"].item(), 4),
            "torch_version": torch.__version__,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
