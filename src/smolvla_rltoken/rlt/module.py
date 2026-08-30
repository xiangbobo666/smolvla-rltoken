"""RL Token encoder/decoder (paper Eq. 1–2, teacher forcing)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from smolvla_rltoken.rlt.config import RLTokenConfig


def causal_attention_mask(size: int, device: torch.device | str | None = None) -> Tensor:
    """Additive mask: position i cannot attend to j > i (upper triangle is -inf)."""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class RLTokenModule(nn.Module):
    """Compress VLA tokens to z_rl; reconstruct them with a causal decoder."""

    def __init__(self, cfg: RLTokenConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        ff = int(cfg.d_model * cfg.mlp_ratio)

        self.enc_in_proj = nn.Linear(cfg.vla_width, d)
        self.e_rl = nn.Parameter(torch.randn(d) * 0.02)
        self.enc_pos = nn.Parameter(torch.zeros(cfg.max_recon_tokens + 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d,
            cfg.n_heads,
            ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_encoder_layers, norm=nn.LayerNorm(d))

        self.dec_in_proj = nn.Linear(cfg.vla_width, d)
        self.dec_pos = nn.Parameter(torch.zeros(cfg.max_recon_tokens + 1, d))
        dec_layer = nn.TransformerEncoderLayer(
            d,
            cfg.n_heads,
            ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, cfg.n_decoder_layers, norm=nn.LayerNorm(d))
        self.out_proj = nn.Linear(d, cfg.vla_width)

        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    def _subsample(self, z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        m = z.shape[1]
        if m <= self.cfg.max_recon_tokens:
            return z, mask
        idx = torch.linspace(0, m - 1, self.cfg.max_recon_tokens, device=z.device).long()
        return z[:, idx], mask[:, idx]

    def encode(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Eq. 1: z_rl = g_phi([z_{1:M}, e_rl])_{M+1}. z: [B, M, vla_width]."""
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z, mask = self._subsample(z, mask)
        b, m, _ = z.shape
        h = self.enc_in_proj(z)
        h = torch.cat([h, self.e_rl.expand(b, 1, -1)], dim=1)
        h = h + self.enc_pos[: m + 1]
        pad = torch.cat([~mask, torch.zeros(b, 1, dtype=torch.bool, device=z.device)], dim=1)
        out = self.encoder(h, src_key_padding_mask=pad)
        return out[:, -1]

    def reconstruct(
        self,
        z: Tensor,
        mask: Tensor | None = None,
        z_rl: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Teacher-forcing reconstruction. Returns (pred, z_rl, mask, z_bar).

        Decoder input: [z_rl, proj(z_1), ..., proj(z_{M-1})]; pred_i targets z_i.
        Loss reduction is masked mean over tokens then mean over D (not paper sum_i).
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar = z.detach()
        z_bar, mask = self._subsample(z_bar, mask)
        if z_rl is None:
            z_rl = self.encode(z_bar, mask)
        b, m, _ = z_bar.shape
        dec_in = torch.cat([z_rl.unsqueeze(1), self.dec_in_proj(z_bar[:, :-1])], dim=1)
        dec_in = dec_in + self.dec_pos[:m]
        out = self.decoder(dec_in, mask=causal_attention_mask(m, z.device))
        pred = self.out_proj(out)
        return pred, z_rl, mask, z_bar

    def reconstruction_loss(
        self, z: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        pred, z_rl, mask, z_bar = self.reconstruct(z, mask)
        err = (pred - z_bar).pow(2).mean(dim=-1)
        w = mask.float()
        loss = (err * w).sum() / w.sum().clamp(min=1.0)
        return loss, z_rl

    @torch.no_grad()
    def rl_token(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.encode(z, mask)
