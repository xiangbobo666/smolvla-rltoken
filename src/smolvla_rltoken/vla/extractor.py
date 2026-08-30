"""Frozen SmolVLA prefix hidden extraction (Stage 1 / M3)."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


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
    def extract(self, batch: dict[str, Tensor]) -> dict:
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

        outputs_embeds, _past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        prefix_out = outputs_embeds[0]

        n_state = 1
        n_lang = tokens.shape[1]
        n_img = prefix_out.shape[1] - n_lang - n_state

        return {
            "z": prefix_out.to(torch.float32),
            "pad_mask": prefix_pad_masks.bool(),
            "n_img_tokens": n_img,
        }

    def select_tokens(self, feats: dict, image_only: bool) -> tuple[Tensor, Tensor]:
        """Return ``(z, mask)``; image-only keeps connector tokens (cameras first)."""
        z, mask = feats["z"], feats["pad_mask"]
        if image_only:
            n = feats["n_img_tokens"]
            z, mask = z[:, :n], mask[:, :n]
        return z, mask
