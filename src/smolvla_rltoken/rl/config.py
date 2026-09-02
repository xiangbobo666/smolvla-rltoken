"""Stage 2 online RL hyperparameters. YAML is the source of truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from smolvla_rltoken.paths import (
    CAMERA_CONFIG_PATH,
    ONLINE_RL_CONFIG_PATH,
    ONLINE_RL_OUTPUT_DIR,
    RL_TOKEN_STAGE1_RUN,
    SFT_LAST_PRETRAINED,
    WANDB_PROJECT,
)
from smolvla_rltoken.vla.action_bounds import DEFAULT_BOUND_MARGIN
from smolvla_rltoken.vla.evaluation import TASK_PROMPT


@dataclass
class ActorCriticConfig:
    """MLP sizes for the chunk Actor and twin Q."""

    rl_token_dim: int = 512
    proprio_dim: int = 9
    chunk_len: int = 10
    action_dim: int = 8
    hidden_dim: int = 256
    n_layers: int = 2
    n_critics: int = 2
    # Noise added to mu inside the training graph (actor loss and TD backup).
    action_std: float = 0.05
    ref_dropout: float = 0.0
    # a = a_tilde + Delta with a zero-initialized residual head.
    use_residual_actor: bool = True
    # Feed the Critic (x, a_tilde, (a - a_tilde) / critic_residual_scale) instead
    # of (x, a). See ChunkCritic and stage2_survey.md 13.5. False reverts to V1.
    critic_residual_input: bool = True
    # Divisor for the residual channel; must be > 0. Resolved from explore_std by
    # OnlineRLConfig.to_actor_critic.
    critic_residual_scale: float = 0.02

    @property
    def x_dim(self) -> int:
        return self.rl_token_dim + self.proprio_dim


@dataclass
class OnlineRLConfig:
    """Launch + algorithm settings for Stage 2 (consumed by ``train_online_rl.py``)."""

    vla_checkpoint: str = str(SFT_LAST_PRETRAINED)
    rl_token_checkpoint: str = str(RL_TOKEN_STAGE1_RUN)
    output_dir: str = str(ONLINE_RL_OUTPUT_DIR)
    camera_config: str = str(CAMERA_CONFIG_PATH)
    job_name: str = "smolvla_rltoken_stage2"

    chunk_len: int = 10
    vla_horizon: int = 50
    action_dim: int = 8
    proprio_dim: int = 9
    rl_token_dim: int = 512
    max_episode_steps: int = 200
    reward: str = "sparse_success"
    dense_reward_debug: bool = False

    hidden_dim: int = 256
    n_layers: int = 2
    n_critics: int = 2
    # Noise added to mu inside the training graph (actor loss and TD backup).
    action_std: float = 0.05
    # Rollout exploration noise, decoupled from action_std. The collected
    # action is mu + N(0, explore_std) and is what the Critic is trained on.
    explore_std: float = 0.02
    # Actor safety clip = demonstrated normalized range expanded by this factor.
    # Never a bare +/-1: this checkpoint normalizes ACTION with MEAN_STD.
    action_bound_margin: float = DEFAULT_BOUND_MARGIN
    ref_dropout: float = 0.0
    bc_beta: float = 1.0
    # "sum" is the paper's ||a - a_tilde||^2 over C * action_dim elements.
    # "mean" is that divided by C * action_dim and needs bc_beta ~80x larger
    # for the same anchor strength.
    bc_reduction: str = "sum"
    # Pure-BC distillation of the frozen VLA chunk into the Actor, run once when
    # warmup ends and before any Q gradient. A zero-initialized residual Actor
    # already matches the reference, so this is a cheap regression guard.
    bc_pretrain_updates: int = 2000
    # Handover gate: the deterministic Actor must be within this mean squared
    # error of the reference before it is allowed to control the env. Warmup is
    # extended (and BC pretraining retried) until it passes.
    handover_bc_threshold: float = 1.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    utd: int = 5
    critic_updates_per_actor: int = 2
    batch_size: int = 256
    # Fraction of each *Critic* batch drawn from successful-episode slots, then
    # from positive-reward slots. Remainder is uniform over the rest. V1 has
    # no PER; 0 disables the extra mix and falls back to uniform.
    success_sample_frac: float = 0.25
    reward_sample_frac: float = 0.05
    # The Actor batch is sampled separately: upsampling successes there biases
    # the policy gradient toward states the current Actor never visits.
    actor_success_sample_frac: float = 0.0
    actor_reward_sample_frac: float = 0.0
    buffer_capacity: int = 200_000
    warmup_env_steps: int = 32_000
    offline_updates_after_warmup: int = 1000
    total_env_steps: int = 1_000_000
    log_freq: int = 160
    save_freq: int = 5000
    # V1 locked: one executed chunk = one transition. Do not set to 2.
    stride: int = 1
    # Residual Actor (a = a_tilde + Delta, zero-init head). False reverts to the
    # non-residual V1 Actor, which measured 0.9-2.4% success against a 15%
    # frozen-VLA baseline; see stage2_survey.md 13.3.
    use_residual_actor: bool = True
    # Critic input in residual coordinates. The V1 Critic saw (x, a) with a
    # varying by only explore_std around an O(1)-O(5) MEAN_STD reference, and it
    # measured action-blind: q_gap=0.60 on states against q_adv_det=0.0013 on
    # actions, itself 14x below its own TD residual. False reverts to V1.
    critic_residual_input: bool = True
    # Divisor for the Critic's residual channel. 0 means "use explore_std", so a
    # one-explore_std perturbation lands at 1.0 on that input.
    critic_residual_scale: float = 0.0
    # Live tripwire on bc_dist_det (deterministic Actor vs the frozen VLA chunk,
    # mean squared over C * action_dim). The residual Critic input multiplies
    # dQ/da by 1 / critic_residual_scale, so bc_beta is no longer calibrated and
    # its stationary point can land in the collapse zone. 1.6e-3 is an RMS of
    # 0.04 normalized units, about 0.46 degrees per arm joint: 16x the handover
    # gate, above the bc_beta=0.1 prediction of 13.4 so it does not block the
    # intended tuning band, and 30x below 13.3's measured collapse at 4.86e-2.
    # Tripping aborts the run and reports the bc_beta that lands on the ceiling,
    # which turns a doomed run into a grad_q_rms measurement. 0 disables.
    actor_drift_ceiling: float = 1.6e-3
    human_intervention: bool = False
    num_envs: int = 16
    # Parallel collect only. ManiSkill sets reconfiguration_freq=0 for
    # num_envs > 1, so peg/box geometry is frozen at the first reconfigure unless
    # a full reset with reconfigure=True is forced periodically. Counted in
    # finished episodes across all envs; 0 disables it (single env reconfigures
    # every episode on its own).
    reconfigure_every_episodes: int = 16
    use_image_tokens_only: bool = True
    # Periodic frozen-VLA probe. Without it ``vla_success_rate`` freezes at its
    # warmup value and there is no reference baseline to compare the Actor
    # against for the rest of the run. Counted in finished episodes; 0 disables.
    # The probe resets the batch, runs pure-reference chunks for
    # ``reference_probe_env_steps`` (0 = num_envs * max_episode_steps), then
    # resets again, so probe episodes are never mixed with Actor chunks.
    reference_probe_every_episodes: int = 320
    reference_probe_env_steps: int = 0
    # Run a second probe window right after the reference one, with the Actor in
    # control and exploration noise off. The stochastic rollout rate confounds
    # "the residual is bad" with "explore_std is expensive"; this pair separates
    # them at the same training state and on the same reconfigured batch. Costs
    # one extra probe window, so the probe overhead doubles.
    probe_deterministic_actor: bool = True

    task: str = TASK_PROMPT
    seed: int = 0
    device: str = "cuda"
    dtype: str | None = None
    sim_backend: str = "physx_cuda"
    mock_env: bool = False
    mock_vla: bool = False
    # MockChunkEnv success step; None keeps the mock reward identically zero.
    mock_success_at: int | None = None

    wandb_enable: bool = True
    wandb_project: str = WANDB_PROJECT
    wandb_disable_artifact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_actor_critic(self) -> ActorCriticConfig:
        return ActorCriticConfig(
            rl_token_dim=self.rl_token_dim,
            proprio_dim=self.proprio_dim,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_critics=self.n_critics,
            action_std=self.action_std,
            ref_dropout=self.ref_dropout,
            use_residual_actor=self.use_residual_actor,
            critic_residual_input=self.critic_residual_input,
            critic_residual_scale=self.resolved_critic_residual_scale(),
        )

    def resolved_critic_residual_scale(self) -> float:
        """Divisor for the Critic's residual channel. Never 0.

        Falls back to ``explore_std`` so the knob can be left unset, and to 1.0
        when exploration is off entirely (a degenerate config where no rescaling
        is meaningful anyway).
        """
        if self.critic_residual_scale > 0:
            return float(self.critic_residual_scale)
        if self.explore_std > 0:
            return float(self.explore_std)
        return 1.0

    def probe_env_steps(self) -> int:
        """Env steps one frozen-VLA probe window runs for."""
        if self.reference_probe_env_steps > 0:
            return int(self.reference_probe_env_steps)
        return max(1, int(self.num_envs)) * int(self.max_episode_steps)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineRLConfig:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        if "dtype" in payload and payload["dtype"] in {"", "null", "None"}:
            payload["dtype"] = None
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path = ONLINE_RL_CONFIG_PATH) -> OnlineRLConfig:
        import yaml

        with Path(path).open() as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_dict(payload)
