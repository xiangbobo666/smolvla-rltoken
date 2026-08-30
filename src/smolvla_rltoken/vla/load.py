"""Load a local or Hub SmolVLA checkpoint."""

from __future__ import annotations

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def load_smolvla_policy(
    checkpoint: str,
    device: str = "cuda",
    dtype: str | None = None,
) -> SmolVLAPolicy:
    """Load SmolVLA onto `device`.

    dtype: None keeps the checkpoint layout (typically bf16 VLM, fp32 projections).
    """
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    if dtype == "bfloat16":
        policy = policy.to(torch.bfloat16)
    elif dtype == "float32":
        policy = policy.to(torch.float32)
    policy = policy.to(device)
    policy.config.device = str(device)
    policy.eval()
    return policy
