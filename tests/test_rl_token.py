"""CPU tests for RL Token encoder/decoder (Eq. 1–2)."""

from __future__ import annotations

import torch

from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.rlt.module import RLTokenModule, causal_attention_mask


def _tiny_module() -> RLTokenModule:
    cfg = RLTokenConfig(
        vla_width=16,
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        max_recon_tokens=16,
        dropout=0.0,
    )
    module = RLTokenModule(cfg)
    module.eval()
    return module


def test_encode_shape():
    module = _tiny_module()
    z = torch.randn(3, 7, 16)
    z_rl = module.encode(z)
    assert z_rl.shape == (3, 32)


def test_causal_mask_upper_triangle():
    mask = causal_attention_mask(5, device="cpu")
    assert mask.shape == (5, 5)
    assert torch.isneginf(mask[0, 1])
    assert mask[1, 0].item() == 0.0
    assert torch.isfinite(mask.diag()).all()


def test_decoder_causal_future_tokens_ignored():
    module = _tiny_module()
    z = torch.randn(2, 6, 16)
    z_rl = module.encode(z)
    pred1, _, _, _ = module.reconstruct(z, z_rl=z_rl)
    z_future = z.clone()
    z_future[:, -1] = z_future[:, -1] + 10.0
    pred2, _, _, _ = module.reconstruct(z_future, z_rl=z_rl)
    # pred[:, i] sees z_rl and z[:, :i]; last-token change must not affect earlier preds.
    torch.testing.assert_close(pred1[:, :-1], pred2[:, :-1], atol=1e-5, rtol=1e-5)


def test_reconstruction_stop_gradient_on_z():
    module = RLTokenModule(
        RLTokenConfig(
            vla_width=16,
            d_model=32,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            max_recon_tokens=16,
        )
    )
    module.train()
    z = torch.randn(2, 5, 16, requires_grad=True)
    loss, z_rl = module.reconstruction_loss(z)
    loss.backward()
    assert z.grad is None
    assert z_rl.requires_grad
    assert module.e_rl.grad is not None and module.e_rl.grad.abs().sum() > 0
    assert module.out_proj.weight.grad is not None and module.out_proj.weight.grad.abs().sum() > 0
