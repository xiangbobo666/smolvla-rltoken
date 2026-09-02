"""Frozen SmolVLA prefix hidden extraction (Stage 1 / Stage 2)."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# Prefix-only must take ``forward_attn_layer``. SmolVLA's default
# ``attention_mode=cross_attn`` otherwise hits ``forward_cross_attn_layer``,
# which does ``inputs_embeds[1].dtype`` and crashes when the expert is None.
# ``fill_kv_cache=True`` selects that self-attn prefill path (same as
# ``sample_actions``). ``use_cache=False`` skips storing KV for Stage 1.
PREFIX_USE_CACHE = False
PREFIX_FILL_KV_CACHE = True


class SmolVLAPrefixExtractor:
    """Prefix-only VLM hidden states; does not run the action expert.

    Prefix layout from ``embed_prefix``: per-camera connector tokens, then
    language tokens, then one state token. Final-layer + RMSNorm, shape
    ``[B, M_total, 960]`` on smolvla_base (SmolVLM2, first 16 layers).
    """

    def __init__(self, policy):
        self.policy = policy
        self.model = policy.model
        self.config = policy.config

    @torch.no_grad()
    def extract(self, batch: dict[str, Tensor], *, use_cache: bool | None = None) -> dict:
        """Prefix hidden states. Stage 1 keeps ``use_cache=False``; Stage 2 passes True.

        ``fill_kv_cache`` stays True so prefix-only hits the self-attn prefill path.
        """
        if use_cache is None:
            use_cache = PREFIX_USE_CACHE
        model = self.model
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs_embeds, past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
            fill_kv_cache=PREFIX_FILL_KV_CACHE,
        )
        prefix_out = outputs_embeds[0]

        n_state = 1
        n_lang = tokens.shape[1]
        n_img = prefix_out.shape[1] - n_lang - n_state

        result = {
            "z": prefix_out.to(torch.float32),
            "pad_mask": prefix_pad_masks.bool(),
            "n_img_tokens": n_img,
            "prefix_pad_masks": prefix_pad_masks,
        }
        if use_cache:
            result["past_key_values"] = past_key_values
        return result

    def select_tokens(self, feats: dict, image_only: bool) -> tuple[Tensor, Tensor]:
        """Return ``(z, mask)``; image-only keeps connector tokens (cameras first)."""
        z, mask = feats["z"], feats["pad_mask"]
        if image_only:
            n = feats["n_img_tokens"]
            z, mask = z[:, :n], mask[:, :n]
        return z, mask

    @torch.no_grad()
    def sample_reference_chunk(self, feats: dict, num_steps: int | None = None) -> Tensor:
        """Sample ã_{1:H} by reusing the prefix KV cache from ``extract(use_cache=True)``.

        Returns padded actions ``[B, chunk_size, max_action_dim]`` in the VLA
        normalized space. Stage 2 then slices ``[:, :C, :action_dim]``.
        """
        if "past_key_values" not in feats or feats["past_key_values"] is None:
            raise RuntimeError("sample_reference_chunk requires extract(..., use_cache=True)")
        model = self.model
        if num_steps is None:
            num_steps = self.config.num_steps
        prefix_pad_masks = feats["prefix_pad_masks"]
        past_key_values = feats["past_key_values"]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        x_t = model.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device
        )
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_t, time_tensor)
            x_t = x_t + dt * v_t
        return x_t
