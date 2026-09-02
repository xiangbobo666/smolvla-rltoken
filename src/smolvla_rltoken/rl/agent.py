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

BatchFn = Callable[[], dict[str, Tensor]]


def bc_distance(action: Tensor, reference: Tensor, reduction: str) -> Tensor:
    """Per-sample BC penalty. ``sum`` is the paper's squared L2 over the chunk."""
    squared = (action - reference).pow(2)
    if reduction == "sum":
        return squared.flatten(1).sum(dim=-1)
    if reduction == "mean":
        return squared.flatten(1).mean(dim=-1)
    raise ValueError(f"bc_reduction must be 'sum' or 'mean', got {reduction!r}")


class RLTAgent:
    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device = "cpu"):
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
        """Rollout action: mu plus ``explore_std`` noise, never ``action_std``."""
        self.actor.eval()
        return self.actor.sample(
            x, ref_chunk, deterministic=deterministic, std=self.cfg.explore_std
        )

    def compute_td_target(self, batch: dict[str, Tensor]) -> Tensor:
        """y = R^{(n)} + (1-terminated) * gamma^n * Q'(x', pi(x', a_tilde'))."""
        with torch.no_grad():
            a_next = self.actor.sample(
                batch["x_next"], batch["next_reference_action"], apply_dropout=True
            )
            q_next = self.critic_target.min_q(
                batch["x_next"], a_next, batch["next_reference_action"]
            )
            returns = discounted_return(batch["reward_sequence"], batch["n_steps"], self.cfg.gamma)
            coeff = bootstrap_coeff(batch["n_steps"], batch["terminated"], self.cfg.gamma)
            return returns + coeff * q_next

    def update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        target = self.compute_td_target(batch)
        q = self.critic(batch["x"], batch["executed_action"], batch["reference_action"])
        critic_loss = F.mse_loss(q, target.unsqueeze(0).expand_as(q))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                target_param.lerp_(param, self.cfg.tau)
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(q.mean().item()),
            "q_std": float(q.mean(dim=0).std().item()) if q.shape[-1] > 1 else 0.0,
            "target_mean": float(target.mean().item()),
        }
        metrics.update(self._q_discrimination(batch, q))
        return metrics

    @staticmethod
    def _q_discrimination(batch: dict[str, Tensor], q: Tensor) -> dict[str, float]:
        """Split Q by success-episode membership.

        A Critic that has learned anything must score success-episode chunks
        above the rest. Without this split a Critic collapsed to a constant is
        indistinguishable from a good one: both show a tiny ``critic_loss``.
        """
        flag = batch.get("success_slot")
        if flag is None:
            return {}
        mask = flag.reshape(-1) >= 0.5
        q_mean = q.mean(dim=0).detach()
        out: dict[str, float] = {}
        if bool(mask.any()):
            out["q_success_mean"] = float(q_mean[mask].mean().item())
        if bool((~mask).any()):
            out["q_rest_mean"] = float(q_mean[~mask].mean().item())
        if "q_success_mean" in out and "q_rest_mean" in out:
            out["q_gap"] = out["q_success_mean"] - out["q_rest_mean"]
        return out

    def update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """-Q on a noisy sample, BC on the deterministic mean.

        BC uses ``mu`` so that the logged ``bc_dist`` is the actual distance to
        the frozen VLA chunk instead of that distance plus a constant
        ``C * action_dim * action_std^2`` noise floor.
        """
        self.actor.train()
        ref = batch["reference_action"]
        mean = self.actor.mu(batch["x"], ref, apply_dropout=True)
        noise = self.ac.action_std * torch.randn_like(mean) if self.ac.action_std > 0 else 0.0
        new_action = mean + noise
        captured: dict[str, Tensor] = {}
        if new_action.requires_grad:
            new_action.register_hook(lambda grad: captured.setdefault("q", grad))
        q = self.critic.min_q(batch["x"], new_action, ref)
        bc = bc_distance(mean, ref, self.cfg.bc_reduction)
        actor_loss = (-q + self.cfg.bc_beta * bc).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        metrics = {
            "actor_loss": float(actor_loss.item()),
            "actor_q": float(q.mean().item()),
            "bc_dist": float(bc.mean().item()),
            "bc_dist_det": float((mean - ref).pow(2).mean().item()),
        }
        metrics.update(self._actor_diagnostics(batch, mean, ref, captured.get("q")))
        return metrics

    def _actor_diagnostics(
        self,
        batch: dict[str, Tensor],
        mean: Tensor,
        ref: Tensor,
        q_grad: Tensor | None,
    ) -> dict[str, float]:
        """Is the Critic actually asking the Actor to leave the frozen VLA chunk?

        ``bc_dist`` alone cannot answer that: a tiny residual is equally
        consistent with "the Critic sees no better action" and with "BC is too
        strong to let the Actor move". The stationary point of
        ``-Q(mu) + beta * sum (mu - ref)^2`` is ``mu - ref = dQ/dmu / (2 beta)``,
        so logging both gradient norms makes ``bc_beta`` a measured quantity
        instead of a guess: ``grad_ratio`` near 1 means the Actor has settled,
        far above 1 means it is still being pushed away from the reference.

        ``q_adv_det`` is the other half: the Critic's own opinion of the
        deterministic Actor chunk versus the frozen VLA chunk on the same
        states. If it stays at 0 the policy gradient has nothing to say and
        lowering ``bc_beta`` would only amplify noise.
        """
        out: dict[str, float] = {}
        eps = 1e-12
        n = mean.shape[0]
        if q_grad is not None:
            # backward() saw ``(-q + ...).mean()``; undo the 1/N to recover dQ/dmu.
            grad_q = (q_grad * n).pow(2).mean().sqrt()
            grad_bc = (2.0 * self.cfg.bc_beta * (mean - ref)).detach().pow(2).mean().sqrt()
            out["grad_q_rms"] = float(grad_q.item())
            out["grad_bc_rms"] = float(grad_bc.item())
            out["grad_ratio"] = float((grad_q / (grad_bc + eps)).item())
        with torch.no_grad():
            q_ref = self.critic.min_q(batch["x"], ref, ref)
            q_det = self.critic.min_q(batch["x"], mean.detach(), ref)
            out["q_ref_mean"] = float(q_ref.mean().item())
            out["q_adv_det"] = float((q_det - q_ref).mean().item())
            executed = batch.get("executed_action")
            if executed is not None:
                q_exec = self.critic.min_q(batch["x"], executed, ref)
                out["q_adv_exec"] = float((q_exec - q_ref).mean().item())
        return out

    def bc_pretrain(
        self, replay: ChunkReplayBuffer, n_updates: int
    ) -> dict[str, float]:
        """Distill the frozen VLA chunk into the Actor. No Q term, no dropout.

        Runs before the first Q gradient so the Actor controls the env from a
        policy that reproduces the SFT baseline. ``bc_updates`` reports how many
        updates actually ran; a short buffer yields 0.
        """
        metrics: dict[str, float] = {"bc_updates": 0.0}
        batch_size = self.cfg.batch_size
        done = 0
        last = 0.0
        for _ in range(max(0, int(n_updates))):
            if len(replay) < batch_size:
                break
            batch = replay.sample(batch_size)
            self.actor.train()
            mean = self.actor.mu(batch["x"], batch["reference_action"])
            loss = (mean - batch["reference_action"]).pow(2).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_opt.step()
            last = float(loss.item())
            done += 1
        metrics["bc_updates"] = float(done)
        if done:
            metrics["bc_pretrain_loss"] = last
        return metrics

    @torch.no_grad()
    def reference_fidelity(
        self, replay: ChunkReplayBuffer, batch_size: int | None = None
    ) -> float:
        """Mean squared distance between the deterministic Actor and the reference.

        This is the handover gate quantity. Returns ``inf`` on an empty buffer so
        an unverifiable Actor never passes the gate.
        """
        if len(replay) < 1:
            return float("inf")
        n = min(int(batch_size or self.cfg.batch_size), len(replay))
        batch = replay.sample(n)
        self.actor.eval()
        mean = self.actor.mu(batch["x"], batch["reference_action"])
        return float((mean - batch["reference_action"]).pow(2).mean().item())

    def update(
        self,
        batch_or_fn: dict[str, Tensor] | BatchFn,
        actor_batch_or_fn: dict[str, Tensor] | BatchFn | None = None,
    ) -> dict[str, float]:
        """One Critic step, and one Actor step every ``critic_updates_per_actor``.

        ``actor_batch_or_fn`` lets the Actor use its own sampling distribution;
        it falls back to the Critic's batch when omitted.
        """
        batch = batch_or_fn() if callable(batch_or_fn) else batch_or_fn
        metrics = self.update_critic(batch)
        self._update_count += 1
        if self._update_count % self.cfg.critic_updates_per_actor == 0:
            source = actor_batch_or_fn if actor_batch_or_fn is not None else batch_or_fn
            actor_batch = source() if callable(source) else source
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
                self.critic_sampler(replay), self.actor_sampler(replay)
            )
            done += 1
        metrics["offline_updates"] = float(done)
        return metrics

    def critic_sampler(self, replay: ChunkReplayBuffer) -> BatchFn:
        return lambda: replay.sample(
            self.cfg.batch_size,
            success_frac=self.cfg.success_sample_frac,
            reward_frac=self.cfg.reward_sample_frac,
        )

    def actor_sampler(self, replay: ChunkReplayBuffer) -> BatchFn:
        return lambda: replay.sample(
            self.cfg.batch_size,
            success_frac=self.cfg.actor_success_sample_frac,
            reward_frac=self.cfg.actor_reward_sample_frac,
        )

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
