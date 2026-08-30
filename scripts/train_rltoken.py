#!/usr/bin/env python
"""Stage 1: train RL Token encoder/decoder on PegInsertion demos (Eq. 2).

VLA is frozen. Each step runs an online prefix forward (no disk embedding cache).

Example (after SFT, or smolvla_base to smoke-test the loop):

  python scripts/train_rltoken.py \\
    --checkpoint /root/autodl-tmp/smolvla-rltoken/models/lerobot/smolvla_base \\
    --dataset-root /root/autodl-tmp/smolvla-rltoken/data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18082")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18082")

import torch

from smolvla_rltoken.paths import DATASET_REPO_ID, DATASET_ROOT, RL_TOKEN_CONFIG_PATH, RL_TOKEN_OUTPUT_DIR, SMOLVLA_BASE
from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.rlt.module import RLTokenModule
from smolvla_rltoken.vla.dataset import build_dataset_and_processors
from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor
from smolvla_rltoken.vla.load import load_smolvla_policy


def _move_batch_tensors(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train(args: argparse.Namespace) -> None:
    device = args.device
    cfg = RLTokenConfig.from_yaml(args.config)
    cfg.d_model = args.d_model
    cfg.n_encoder_layers = args.n_layers
    cfg.n_decoder_layers = args.n_layers
    cfg.steps = args.steps
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.use_image_tokens_only = not args.all_prefix_tokens

    print(f"[stage1] loading SmolVLA from {args.checkpoint} ...")
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    dataset, preprocessor, _ = build_dataset_and_processors(
        policy, args.dataset, args.dataset_root
    )
    extractor = SmolVLAPrefixExtractor(policy)
    policy.requires_grad_(False)
    policy.eval()

    rl_token = RLTokenModule(cfg).to(device)
    n_params = sum(p.numel() for p in rl_token.parameters())
    print(f"[stage1] RL token module: {n_params / 1e6:.1f}M params, d_model={cfg.d_model}")

    params = list(rl_token.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = preprocessor(batch)
        batch = _move_batch_tensors(batch, device)

        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
        loss, _ = rl_token.reconstruction_loss(z, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        opt.step()
        step += 1

        if step % args.log_freq == 0 or step == 1:
            speed = step / (time.time() - t0)
            print(
                f"[stage1] step {step}/{cfg.steps} loss_ro={loss.item():.5f} "
                f"z_shape={tuple(z.shape)} ({speed:.2f} it/s)",
                flush=True,
            )

        if step % args.save_freq == 0 or step == cfg.steps:
            ckpt = {"rl_token": rl_token.state_dict(), "config": cfg.to_dict(), "step": step}
            torch.save(ckpt, out_dir / "rl_token.pt")
            print(f"[stage1] saved {out_dir / 'rl_token.pt'}", flush=True)

    print(f"[stage1] done; saved to {out_dir / 'rl_token.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 RL Token reconstruction training")
    p.add_argument("--checkpoint", default=str(SMOLVLA_BASE))
    p.add_argument("--dataset", default=DATASET_REPO_ID)
    p.add_argument("--dataset-root", default=str(DATASET_ROOT))
    p.add_argument("--config", default=str(RL_TOKEN_CONFIG_PATH))
    p.add_argument("--out", default=str(RL_TOKEN_OUTPUT_DIR))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--all-prefix-tokens", action="store_true", help="Reconstruct lang+state too")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=500)
    train(p.parse_args())


if __name__ == "__main__":
    main()
