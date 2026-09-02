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
    """MLP sizes for the non-residual chunk Actor and twin Q."""

    rl_token_dim: int = 512
    proprio_dim: int = 9
    chunk_len: int = 10
    action_dim: int = 8
    hidden_dim: int = 256
    n_layers: int = 2
    n_critics: int = 2
    action_std: float = 0.05
    ref_dropout: float = 0.5

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
    action_std: float = 0.05
    # Actor safety clip = demonstrated normalized range expanded by this factor.
    # Never a bare +/-1: this checkpoint normalizes ACTION with MEAN_STD.
    action_bound_margin: float = DEFAULT_BOUND_MARGIN
    ref_dropout: float = 0.5
    bc_beta: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    utd: int = 5
    critic_updates_per_actor: int = 2
    batch_size: int = 256
    # Fraction of each update batch drawn from successful-episode slots, then
    # from positive-reward slots. Remainder is uniform over the rest. V1 has
    # no PER; 0 disables the extra mix and falls back to uniform.
    success_sample_frac: float = 0.25
    reward_sample_frac: float = 0.05
    buffer_capacity: int = 200_000
    warmup_env_steps: int = 32_000
    offline_updates_after_warmup: int = 1000
    total_env_steps: int = 1_000_000
    log_freq: int = 160
    save_freq: int = 5000
    # V1 locked: one executed chunk = one transition. Do not set to 2.
    stride: int = 1
    use_residual_actor: bool = False
    human_intervention: bool = False
    num_envs: int = 16
    # Parallel collect only. ManiSkill sets reconfiguration_freq=0 for
    # num_envs > 1, so peg/box geometry is frozen at the first reconfigure unless
    # a full reset with reconfigure=True is forced periodically. Counted in
    # finished episodes across all envs; 0 disables it (single env reconfigures
    # every episode on its own).
    reconfigure_every_episodes: int = 16
    use_image_tokens_only: bool = True

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
        )

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
