"""CPU tests for prefix-only SmolVLA extraction settings."""

from smolvla_rltoken.vla.extractor import PREFIX_FILL_KV_CACHE, PREFIX_USE_CACHE


def test_prefix_only_uses_self_attn_prefill_path():
    # cross_attn + expert=None requires fill_kv_cache=True; Stage 1 does not keep KV.
    assert PREFIX_FILL_KV_CACHE is True
    assert PREFIX_USE_CACHE is False
