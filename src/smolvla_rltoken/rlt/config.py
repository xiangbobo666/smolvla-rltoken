"""Stage 1 RL Token encoder/decoder hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class RLTokenConfig:
    """Eq. 1–2 encoder/decoder. Widths default to local smolvla_base (SmolVLM2, D=960)."""

    vla_width: int = 960
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Fixed-instruction tasks (PegInsertion): reconstruct image connector tokens only.
    use_image_tokens_only: bool = True
    max_recon_tokens: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    steps: int = 5000
    batch_size: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLTokenConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLTokenConfig:
        import yaml

        with Path(path).open() as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_dict(payload)
