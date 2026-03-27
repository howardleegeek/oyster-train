"""
LeWorldModel (LeWM) configuration for federated training on phones.

LeWM is a ~15M parameter JEPA world model that learns to predict
next-state embeddings from pixel observations. Much lighter than
Qwen2.5-1.5B, making it ideal for on-device training.

Reference: arxiv.org/abs/2603.19312
"""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    """Vision encoder configuration.

    Default: MobileNetV3-Small (~2.5M params) for phone deployment.
    For simulation/GPU: use vit_tiny for accuracy.
    """
    backbone: Literal["mobilenet_v3_small", "vit_tiny", "efficientnet_b0"] = "mobilenet_v3_small"
    image_size: int = 224
    patch_size: int = 14  # only used for vit_tiny
    pretrained: bool = False
    hidden_size: int = 192  # output embedding dim


class PredictorConfig(BaseModel):
    """AR Predictor (Transformer with AdaLN-zero conditioning)."""
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0


class SIGRegConfig(BaseModel):
    """Sketch Isotropic Gaussian Regularizer."""
    weight: float = 0.09
    knots: int = 17
    num_proj: int = 1024


class LeWMTrainingConfig(BaseModel):
    """Local training config for phone clients."""
    lr: float = 5e-5
    weight_decay: float = 1e-3
    batch_size: int = 4  # phone-friendly
    local_steps: int = 200  # fewer than Qwen (500) since model is smaller
    history_size: int = 3  # number of context frames
    num_preds: int = 1  # predict 1 step ahead
    precision: Literal["fp32", "fp16"] = "fp16"
    gradient_clip: float = 1.0


class LeWMFederationConfig(BaseModel):
    """Federation config — no LoRA needed, full-parameter training."""
    num_rounds: int = 100
    fraction_fit: float = 0.1
    min_clients: int = 5  # fewer needed than LLM training
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    use_lora: bool = False  # 15M params = full training is fine
    compression_k_ratio: float = 0.01  # Top-K 1%


class LeWMDataConfig(BaseModel):
    """Data source configuration for phone cameras."""
    source: Literal["camera", "hdf5", "synthetic"] = "camera"
    frame_rate: int = 5  # 5 FPS capture
    sequence_length: int = 16  # frames per training sequence
    action_dim: int = 6  # IMU: [accel_x/y/z, gyro_x/y/z]
    frameskip: int = 1
    hdf5_path: Optional[str] = None  # for sim/eval mode


class LeWMConfig(BaseModel):
    """Complete LeWM federated training configuration."""
    encoder: EncoderConfig = EncoderConfig()
    predictor: PredictorConfig = PredictorConfig()
    sigreg: SIGRegConfig = SIGRegConfig()
    training: LeWMTrainingConfig = LeWMTrainingConfig()
    federation: LeWMFederationConfig = LeWMFederationConfig()
    data: LeWMDataConfig = LeWMDataConfig()

    @property
    def embed_dim(self) -> int:
        return self.encoder.hidden_size

    @property
    def estimated_params_m(self) -> float:
        """Rough parameter count in millions."""
        encoder_params = {
            "mobilenet_v3_small": 2.5,
            "vit_tiny": 5.5,
            "efficientnet_b0": 5.3,
        }
        predictor_params = (
            self.predictor.depth
            * (4 * self.embed_dim * self.predictor.dim_head * self.predictor.heads  # attn
               + 2 * self.embed_dim * self.predictor.mlp_dim)  # ffn
        ) / 1e6
        return encoder_params.get(self.encoder.backbone, 5.0) + predictor_params + 2.0  # +2M for projectors

    @property
    def estimated_memory_mb(self) -> float:
        """Estimated training memory in MB (FP16)."""
        params_mb = self.estimated_params_m * 2  # FP16 = 2 bytes/param
        # params + grads + adam states (2x) + activations
        return params_mb * 4 + 100  # ~100MB activations at batch_size=4


def get_ubs1_config() -> LeWMConfig:
    """Optimal config for UBS1 phone (6GB RAM, Mali-G57)."""
    return LeWMConfig(
        encoder=EncoderConfig(backbone="mobilenet_v3_small", image_size=224),
        predictor=PredictorConfig(depth=4, heads=8, mlp_dim=1024),  # smaller predictor
        training=LeWMTrainingConfig(batch_size=2, precision="fp16", local_steps=100),
        data=LeWMDataConfig(source="camera", frame_rate=5),
    )


def get_simulation_config() -> LeWMConfig:
    """CPU-friendly config for testing."""
    return LeWMConfig(
        encoder=EncoderConfig(backbone="mobilenet_v3_small", image_size=96),
        predictor=PredictorConfig(depth=2, heads=4, mlp_dim=512),
        training=LeWMTrainingConfig(batch_size=8, precision="fp32", local_steps=10),
        data=LeWMDataConfig(source="synthetic", frame_rate=2, sequence_length=8),
    )


def get_gpu_config() -> LeWMConfig:
    """Full-size config for GPU training (matches original LeWM paper)."""
    return LeWMConfig(
        encoder=EncoderConfig(backbone="vit_tiny", image_size=224),
        predictor=PredictorConfig(depth=6, heads=16, mlp_dim=2048),
        training=LeWMTrainingConfig(batch_size=32, precision="fp16", local_steps=500),
        data=LeWMDataConfig(source="hdf5"),
    )
