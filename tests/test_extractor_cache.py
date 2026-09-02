"""CPU tests for optional prefix KV cache (no real VLA)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from smolvla_rltoken.vla.extractor import PREFIX_FILL_KV_CACHE, PREFIX_USE_CACHE, SmolVLAPrefixExtractor


class _FakeVLM:
    def __init__(self):
        self.last_use_cache = None

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
    ):
        del attention_mask, position_ids, past_key_values
        assert fill_kv_cache is True
        self.last_use_cache = use_cache
        prefix = inputs_embeds[0]
        kv = {"cached": True} if use_cache else None
        return [prefix], kv


class _FakeModel:
    def __init__(self):
        self.vlm_with_expert = _FakeVLM()
        self.chunk_size = 4
        self.max_action_dim = 8
        self.num_steps = 3
        self.denoise_calls = 0

    def embed_prefix(self, images, img_masks, tokens, masks, state=None):
        del images, img_masks, masks, state
        batch, n_lang = tokens.shape
        n_img = 4
        prefix_len = n_img + n_lang + 1
        embs = torch.arange(batch * prefix_len * 6, dtype=torch.float32).reshape(batch, prefix_len, 6)
        pad = torch.ones(batch, prefix_len, dtype=torch.bool)
        att = torch.ones(batch, prefix_len)
        return embs, pad, att

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        del prefix_pad_masks, timestep
        assert past_key_values == {"cached": True}
        self.denoise_calls += 1
        return torch.ones_like(x_t)


def _fake_policy() -> SimpleNamespace:
    model = _FakeModel()
    config = SimpleNamespace(num_steps=3, chunk_size=4, max_action_dim=8)

    def prepare_images(batch):
        del batch
        return None, None

    def prepare_state(batch):
        return batch["observation.state"]

    return SimpleNamespace(
        model=model,
        config=config,
        prepare_images=prepare_images,
        prepare_state=prepare_state,
    )


def _batch() -> dict[str, Tensor]:
    return {
        "observation.state": torch.zeros(1, 9),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }


def test_stage1_defaults_do_not_keep_kv():
    assert PREFIX_FILL_KV_CACHE is True
    assert PREFIX_USE_CACHE is False


def test_extract_default_omits_past_key_values():
    extractor = SmolVLAPrefixExtractor(_fake_policy())
    feats = extractor.extract(_batch())
    assert "past_key_values" not in feats
    assert extractor.model.vlm_with_expert.last_use_cache is False
    z, mask = extractor.select_tokens(feats, image_only=True)
    assert z.shape == (1, 4, 6)
    assert mask.shape == (1, 4)


def test_extract_use_cache_returns_kv():
    extractor = SmolVLAPrefixExtractor(_fake_policy())
    feats = extractor.extract(_batch(), use_cache=True)
    assert feats["past_key_values"] == {"cached": True}
    assert "prefix_pad_masks" in feats


def test_sample_reference_chunk_requires_cache():
    extractor = SmolVLAPrefixExtractor(_fake_policy())
    feats = extractor.extract(_batch())
    with pytest.raises(RuntimeError, match="use_cache=True"):
        extractor.sample_reference_chunk(feats)


def test_sample_reference_chunk_reuses_kv():
    policy = _fake_policy()
    extractor = SmolVLAPrefixExtractor(policy)
    feats = extractor.extract(_batch(), use_cache=True)
    chunk = extractor.sample_reference_chunk(feats)
    assert chunk.shape == (1, 4, 8)
    assert policy.model.denoise_calls == 3
    # x_0=0, each step x += dt * 1 with dt=-1/3, three steps -> -1
    torch.testing.assert_close(chunk, torch.full((1, 4, 8), -1.0))
