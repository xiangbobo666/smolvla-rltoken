"""Actor-Critic updates: critic on executed actions, actor on newly sampled ones."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from smolvla_rltoken.rl.actor import ChunkActor
from smolvla_rltoken.rl.config import ActorCriticConfig, OnlineRLConfig
from smolvla_rltoken.rl.critic import ChunkCritic
from smolvla_rltoken.rl.replay import ChunkReplayBuffer
from smolvla_rltoken.rl.td import bootstrap_coeff, discounted_return


class RLTAgent:
    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device = "cpu"):
        if cfg.use_residual_actor:
            raise ValueError("V1 locks a non-residual Actor (use_residual_actor=false)")
        self.cfg = cfg
        self.device = torch.device(device)
        ac = cfg.to_actor_critic()
        self.ac: ActorCriticConfig = ac
        self.actor = ChunkActor(ac).to(self.device)
        self.critic = ChunkCritic(ac).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self._update_count = 0

    @torch.no_grad()
    def act(self, x: Tensor, ref_chunk: Tensor, deterministic: bool = False) -> Tensor:
        self.actor.eval()
        action = self.actor.sample(x, ref_chunk, deterministic=deterministic)
        return action

    def compute_td_target(self, batch: dict[str, Tensor]) -> Tensor:
        """y = R^{(n)} + (1-terminated) * gamma^n * Q'(x', pi(x', ã'))."""
        with torch.no_grad():
            ref_next = self.actor.apply_ref_dropout(batch["next_reference_action"])
            a_next = self.actor.sample(batch["x_next"], ref_next)
            q_next = self.critic_target.min_q(batch["x_next"], a_next)
            returns = discounted_return(batch["reward_sequence"], batch["n_steps"], self.cfg.gamma)
            coeff = bootstrap_coeff(batch["n_steps"], batch["terminated"], self.cfg.gamma)
            return returns + coeff * q_next

    def update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        target = self.compute_td_target(batch)
        q = self.critic(batch["x"], batch["executed_action"])
        critic_loss = F.mse_loss(q, target.unsqueeze(0).expand_as(q))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                target_param.lerp_(param, self.cfg.tau)
        return {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(q.mean().item()),
            "target_mean": float(target.mean().item()),
        }

    def update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        self.actor.train()
        ref = batch["reference_action"]
        ref_in = self.actor.apply_ref_dropout(ref)
        new_action = self.actor.sample(batch["x"], ref_in)
        q = self.critic.min_q(batch["x"], new_action)
        bc = (new_action - ref).pow(2).mean(dim=(1, 2))
        actor_loss = (-q + self.cfg.bc_beta * bc).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "actor_q": float(q.mean().item()),
            "bc_dist": float(bc.mean().item()),
        }

    def update(self, batch_or_fn: dict[str, Tensor] | Callable[[], dict[str, Tensor]]) -> dict[str, float]:
        batch = batch_or_fn() if callable(batch_or_fn) else batch_or_fn
        metrics = self.update_critic(batch)
        self._update_count += 1
        if self._update_count % self.cfg.critic_updates_per_actor == 0:
            actor_batch = batch_or_fn() if callable(batch_or_fn) else batch
            metrics.update(self.update_actor(actor_batch))
        return metrics

    def run_offline_updates(self, replay: ChunkReplayBuffer, n_updates: int) -> dict[str, float]:
        """Warm-start on the warmup buffer before the Actor controls the env.

        ``offline_updates`` in the returned metrics is how many updates actually
        ran. A short warmup buffer yields 0, which the caller must surface rather
        than treat as a completed warm-start.
        """
        metrics: dict[str, float] = {}
        batch_size = self.cfg.batch_size
        done = 0
        for _ in range(n_updates):
            if len(replay) < batch_size:
                break
            metrics = self.update(
                lambda: replay.sample(
                    batch_size,
                    success_frac=self.cfg.success_sample_frac,
                    reward_frac=self.cfg.reward_sample_frac,
                )
            )
            done += 1
        metrics["offline_updates"] = float(done)
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self._update_count = int(state.get("update_count", 0))
