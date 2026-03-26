"""
LeWorldModel loader — builds the JEPA world model for federated training.

Adapts the original LeWM architecture (arxiv.org/abs/2603.19312) for
on-device training: device-agnostic ops, mobile-friendly encoders,
and Flower-compatible state dict extraction.
"""
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.lewm_config import LeWMConfig

logger = logging.getLogger(__name__)


# ─── SIGReg (fixed: device-agnostic) ─────────────────────────────

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Original code hardcodes device="cuda". This version infers device
    from the input tensor, enabling CPU and mobile GPU execution.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D)"""
        # FIX: use proj.device instead of hardcoded "cuda"
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ─── Transformer components ──────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on actions."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    """Autoregressive next-state predictor."""

    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim,
                 hidden_dim, output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    """Action encoder: raw action → embedding."""

    def __init__(self, input_dim=6, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── Encoder backends ────────────────────────────────────────────

def _build_mobilenet_encoder(cfg: LeWMConfig) -> Tuple[nn.Module, int]:
    """MobileNetV3-Small as pixel encoder (~2.5M params)."""
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    weights = MobileNet_V3_Small_Weights.DEFAULT if cfg.encoder.pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    # Remove classifier, keep features (output: 576-dim)
    backbone.classifier = nn.Identity()
    raw_dim = 576
    # Project to embed_dim
    proj = nn.Linear(raw_dim, cfg.embed_dim)
    encoder = nn.Sequential(backbone, proj)
    return encoder, cfg.embed_dim


def _build_vit_encoder(cfg: LeWMConfig) -> Tuple[nn.Module, int]:
    """ViT-Tiny encoder (matches original LeWM paper)."""
    try:
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            "tiny", patch_size=cfg.encoder.patch_size,
            image_size=cfg.encoder.image_size, pretrained=cfg.encoder.pretrained,
            use_mask_token=False,
        )
        return encoder, encoder.config.hidden_size
    except ImportError:
        from torchvision.models import vit_b_16
        logger.warning("stable_pretraining not available, using torchvision ViT")
        backbone = vit_b_16(pretrained=False)
        backbone.heads = nn.Identity()
        return backbone, 768


# ─── JEPA world model ────────────────────────────────────────────

class LeWM(nn.Module):
    """LeWorldModel adapted for federated phone training.

    Differences from original:
    - Device-agnostic (no CUDA hardcodes)
    - Swappable encoder backend (MobileNet/ViT)
    - Flat state_dict for Flower parameter exchange
    """

    def __init__(self, cfg: LeWMConfig):
        super().__init__()
        self.cfg = cfg

        # Build encoder
        if cfg.encoder.backbone == "mobilenet_v3_small":
            self.encoder, enc_dim = _build_mobilenet_encoder(cfg)
            self._encoder_type = "mobilenet"
        else:
            self.encoder, enc_dim = _build_vit_encoder(cfg)
            self._encoder_type = "vit"

        embed_dim = cfg.embed_dim
        action_dim = cfg.data.action_dim * cfg.data.frameskip

        self.predictor = ARPredictor(
            num_frames=cfg.training.history_size,
            input_dim=embed_dim,
            hidden_dim=enc_dim,
            output_dim=enc_dim,
            depth=cfg.predictor.depth,
            heads=cfg.predictor.heads,
            dim_head=cfg.predictor.dim_head,
            mlp_dim=cfg.predictor.mlp_dim,
            dropout=cfg.predictor.dropout,
            emb_dropout=cfg.predictor.emb_dropout,
        )

        self.action_encoder = Embedder(
            input_dim=action_dim, emb_dim=embed_dim,
        )

        self.projector = MLP(enc_dim, 2048, embed_dim)
        self.pred_proj = MLP(enc_dim, 2048, embed_dim)
        self.sigreg = SIGReg(knots=cfg.sigreg.knots, num_proj=cfg.sigreg.num_proj)

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations to embeddings.

        Args:
            pixels: (B, T, C, H, W) image sequence
        Returns:
            emb: (B, T, D) embeddings
        """
        b, t = pixels.shape[:2]
        x = rearrange(pixels, "b t c h w -> (b t) c h w").float()

        if self._encoder_type == "mobilenet":
            x = self.encoder(x)  # (B*T, D)
        else:
            output = self.encoder(x, interpolate_pos_encoding=True)
            x = output.last_hidden_state[:, 0]  # CLS token

        emb = self.projector(x)
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def forward(self, pixels: torch.Tensor, actions: torch.Tensor) -> Dict:
        """Full forward pass for training.

        Args:
            pixels: (B, T, C, H, W) image sequence
            actions: (B, T, action_dim)
        Returns:
            dict with loss, pred_loss, sigreg_loss
        """
        ctx_len = self.cfg.training.history_size
        lambd = self.cfg.sigreg.weight

        actions = torch.nan_to_num(actions, 0.0)
        emb = self.encode_pixels(pixels)  # (B, T, D)
        act_emb = self.action_encoder(actions)  # (B, T, D)

        n_preds = self.cfg.training.num_preds
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        # Target: shifted by num_preds, same length as context
        tgt_emb = emb[:, n_preds:ctx_len + n_preds]

        pred_raw = self.predictor(ctx_emb, ctx_act)
        pred_emb = self.pred_proj(rearrange(pred_raw, "b t d -> (b t) d"))
        pred_emb = rearrange(pred_emb, "(b t) d -> b t d", b=pixels.size(0))

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        loss = pred_loss + lambd * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


# ─── Public API ───────────────────────────────────────────────────

def load_lewm_model(cfg: LeWMConfig) -> LeWM:
    """Build a LeWM model from config."""
    model = LeWM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LeWM loaded: {n_params/1e6:.1f}M params ({n_train/1e6:.1f}M trainable)")
    logger.info(f"Encoder: {cfg.encoder.backbone}, embed_dim: {cfg.embed_dim}")
    logger.info(f"Estimated training memory: {cfg.estimated_memory_mb:.0f}MB")
    return model


def get_model_state(model: LeWM) -> Dict[str, torch.Tensor]:
    """Extract full state dict for Flower parameter exchange."""
    return {k: v.cpu() for k, v in model.state_dict().items()}


def set_model_state(model: LeWM, state: Dict[str, torch.Tensor]):
    """Load state dict from Flower server."""
    model.load_state_dict(state, strict=True)


def extract_delta(before: Dict[str, torch.Tensor],
                  after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute parameter delta between two checkpoints.

    No LoRA needed — LeWM is small enough for full-parameter federated training.
    """
    return {k: after[k] - before[k] for k in before if k in after}


def count_trainable_parameters(model: LeWM) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
