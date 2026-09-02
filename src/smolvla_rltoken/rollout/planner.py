"""Plan an action chunk from a frozen VLA (+ optional Actor).

``observe`` runs VLA+encoder once (prefix KV cache on). ``act`` chooses the
executed chunk: reference during warmup, Actor otherwise. Human intervention
is not wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from smolvla_rltoken.vla.evaluation import EXPECTED_ACTION_DIM, EXPECTED_STATE_DIM, TASK_PROMPT


def _infer_batch_size(observation: Any) -> int:
    if isinstance(observation, Mapping) and "t" in observation:
        ticks = np.asarray(observation["t"]).reshape(-1)
        return max(1, int(ticks.shape[0]))
    if isinstance(observation, Mapping) and "agent" in observation:
        try:
            qpos = np.asarray(observation["agent"]["qpos"])
        except (KeyError, TypeError):
            return 1
        if qpos.ndim >= 2:
            return max(1, int(qpos.shape[0]))
        return 1
    return 1


@dataclass
class ObserveResult:
    z_rl: Tensor
    proprio: Tensor
    reference_action: Tensor


def _check_action_bounds(
    bounds: tuple[Tensor, Tensor] | None, action_dim: int
) -> tuple[Tensor, Tensor] | None:
    if bounds is None:
        return None
    lo, hi = bounds
    lo, hi = lo.detach().cpu().reshape(-1).float(), hi.detach().cpu().reshape(-1).float()
    if lo.numel() != action_dim or hi.numel() != action_dim:
        raise ValueError(
            f"action_bounds must have {action_dim} dims, got {lo.numel()} and {hi.numel()}"
        )
    if bool((hi <= lo).any()):
        raise ValueError("action_bounds must satisfy hi > lo on every dimension")
    return lo, hi


class Planner(Protocol):
    def observe(self, observation: Any) -> ObserveResult: ...

    def observe_batch(self, observation: Any) -> ObserveResult: ...

    def act(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor: ...

    def act_batch(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor: ...

    def to_env_action(self, normalized_step: Tensor) -> np.ndarray: ...

    def to_env_actions(self, normalized_steps: Tensor) -> np.ndarray: ...


class MockPlanner:
    """Deterministic planner for CPU tests and ``--smoke``."""

    def __init__(
        self,
        *,
        chunk_len: int = 10,
        action_dim: int = EXPECTED_ACTION_DIM,
        rl_token_dim: int = 8,
        proprio_dim: int = EXPECTED_STATE_DIM,
    ):
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.rl_token_dim = rl_token_dim
        self.proprio_dim = proprio_dim
        self.observe_count = 0
        self.use_actor_calls: list[bool] = []

    def observe(self, observation: Any) -> ObserveResult:
        batched = self.observe_batch(observation)
        return ObserveResult(
            z_rl=batched.z_rl[0],
            proprio=batched.proprio[0],
            reference_action=batched.reference_action[0],
        )

    def observe_batch(self, observation: Any) -> ObserveResult:
        self.observe_count += 1
        n = float(self.observe_count)
        batch_size = _infer_batch_size(observation)
        z_rl = torch.full((batch_size, self.rl_token_dim), n)
        proprio = torch.full((batch_size, self.proprio_dim), n * 0.01)
        reference = torch.full((batch_size, self.chunk_len, self.action_dim), n * 0.1)
        return ObserveResult(z_rl=z_rl, proprio=proprio, reference_action=reference)

    def act(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor:
        chunk = self.act_batch(features, use_actor=use_actor, deterministic=deterministic)
        if chunk.shape[0] != 1:
            raise ValueError(f"act() expects 1 env, got batch {chunk.shape[0]}")
        return chunk[0]

    def act_batch(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor:
        del deterministic
        self.use_actor_calls.append(use_actor)
        reference = features.reference_action
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if use_actor:
            return reference + 1.0
        return reference.clone()

    def to_env_action(self, normalized_step: Tensor) -> np.ndarray:
        return self.to_env_actions(normalized_step).reshape(-1)

    def to_env_actions(self, normalized_steps: Tensor) -> np.ndarray:
        array = np.asarray(normalized_steps.detach().cpu().numpy(), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array.reshape(-1, self.action_dim)


class FrozenVLAPlanner:
    """Frozen SmolVLA + frozen encoder; Actor is optional and not frozen.

    ``action_bounds`` is a per-dimension ``(lo, hi)`` pair in the VLA normalized
    space (see ``vla/action_bounds.py``). It clips the *Actor* chunk only. The
    warmup reference is executed verbatim so that pure-VLA rollouts match the
    SFT evaluation chain exactly.
    """

    def __init__(
        self,
        *,
        extractor,
        encoder,
        actor=None,
        preprocessor=None,
        postprocessor=None,
        chunk_len: int = 10,
        action_dim: int = EXPECTED_ACTION_DIM,
        proprio_dim: int = EXPECTED_STATE_DIM,
        image_only: bool = True,
        task: str = TASK_PROMPT,
        action_bounds: tuple[Tensor, Tensor] | None = None,
    ):
        self.extractor = extractor
        self.encoder = encoder
        self.actor = actor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.image_only = image_only
        self.task = task
        self.action_bounds = _check_action_bounds(action_bounds, action_dim)
        # Diagnostic only: the reference is never clipped, but a VLA that
        # predicts far outside the demonstrated range is worth surfacing.
        self.reference_out_of_bounds = 0

    def _prepare_batch(self, observation: Any) -> dict[str, Tensor]:
        from smolvla_rltoken.vla.evaluation import prepare_policy_observation_batch

        batch = prepare_policy_observation_batch(observation, task=self.task)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        return batch

    @torch.no_grad()
    def observe_batch(self, observation: Any) -> ObserveResult:
        batch = self._prepare_batch(observation)
        feats = self.extractor.extract(batch, use_cache=True)
        z, mask = self.extractor.select_tokens(feats, self.image_only)
        z_rl = self.encoder.rl_token(z, mask).float()
        state = batch["observation.state"]
        if state.ndim == 3:
            state = state.squeeze(1)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        proprio = state[:, : self.proprio_dim].float()
        ref_full = self.extractor.sample_reference_chunk(feats)
        reference = ref_full[:, : self.chunk_len, : self.action_dim].float()
        return ObserveResult(
            z_rl=z_rl.detach().cpu(),
            proprio=proprio.detach().cpu(),
            reference_action=reference.detach().cpu(),
        )

    def observe(self, observation: Any) -> ObserveResult:
        batched = self.observe_batch(observation)
        if batched.z_rl.shape[0] != 1:
            raise ValueError(f"observe() expects 1 env, got batch {batched.z_rl.shape[0]}")
        return ObserveResult(
            z_rl=batched.z_rl[0],
            proprio=batched.proprio[0],
            reference_action=batched.reference_action[0],
        )

    def act(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor:
        chunk = self.act_batch(features, use_actor=use_actor, deterministic=deterministic)
        if chunk.shape[0] != 1:
            raise ValueError(f"act() expects 1 env, got batch {chunk.shape[0]}")
        return chunk[0]

    def clamp_action(self, chunk: Tensor) -> Tensor:
        """Per-dimension safety clip in the normalized space. Actor output only."""
        if self.action_bounds is None:
            return chunk
        lo, hi = self.action_bounds
        return torch.clamp(chunk, lo.to(chunk.device), hi.to(chunk.device))

    def _count_reference_out_of_bounds(self, reference: Tensor) -> None:
        if self.action_bounds is None:
            return
        lo, hi = self.action_bounds
        ref = reference[..., : self.action_dim]
        outside = (ref < lo) | (ref > hi)
        self.reference_out_of_bounds += int(outside.flatten(1).any(dim=-1).sum())

    def act_batch(
        self,
        features: ObserveResult,
        *,
        use_actor: bool,
        deterministic: bool = False,
    ) -> Tensor:
        reference = features.reference_action
        z_rl = features.z_rl
        proprio = features.proprio
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        self._count_reference_out_of_bounds(reference)
        if not use_actor or self.actor is None:
            # Warmup runs the frozen VLA chunk verbatim, matching the SFT eval chain.
            return reference
        x = torch.cat([z_rl, proprio], dim=-1)
        device = next(self.actor.parameters()).device
        chunk = self.actor.sample(x.to(device), reference.to(device), deterministic=deterministic)
        return self.clamp_action(chunk.detach().cpu())

    def to_env_action(self, normalized_step: Tensor) -> np.ndarray:
        return self.to_env_actions(normalized_step).reshape(-1)

    def to_env_actions(self, normalized_steps: Tensor) -> np.ndarray:
        steps = normalized_steps
        if steps.ndim == 1:
            steps = steps.unsqueeze(0)
        steps = steps[..., : self.action_dim]
        if self.postprocessor is None:
            array = np.asarray(steps.detach().cpu().numpy(), dtype=np.float32)
        else:
            out = self.postprocessor(steps)
            if hasattr(out, "detach"):
                out = out.detach()
            if hasattr(out, "cpu"):
                out = out.cpu()
            if hasattr(out, "numpy"):
                out = out.numpy()
            array = np.asarray(out, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
        return array[:, : self.action_dim]
