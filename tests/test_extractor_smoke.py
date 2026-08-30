"""Optional GPU smoke test for prefix hidden extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from smolvla_rltoken.paths import SMOLVLA_BASE
from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor
from smolvla_rltoken.vla.load import load_smolvla_policy

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not Path(SMOLVLA_BASE).exists(), reason="smolvla_base checkpoint missing"),
]


def test_prefix_image_tokens_shape():
    from lerobot.configs.types import FeatureType, PolicyFeature

    device = "cuda"
    policy = load_smolvla_policy(str(SMOLVLA_BASE), device=device)
    policy.config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        "observation.images.base_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.hand_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    policy.config.validate_features()

    batch = {
        "observation.images.base_camera": torch.rand(1, 3, 128, 128, device=device),
        "observation.images.hand_camera": torch.rand(1, 3, 128, 128, device=device),
        "observation.state": torch.zeros(1, 9, device=device),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 8, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 8, dtype=torch.bool, device=device),
    }
    extractor = SmolVLAPrefixExtractor(policy)
    feats = extractor.extract(batch)
    z, mask = extractor.select_tokens(feats, image_only=True)
    assert z.ndim == 3 and z.shape[0] == 1 and z.shape[-1] == 960
    # Two 512x512 cameras after connector: typically 64 tokens each.
    assert z.shape[1] == 128
    assert mask.shape == z.shape[:2]
    assert mask.dtype == torch.bool
