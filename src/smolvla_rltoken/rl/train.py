"""Stage 2 online RL: CPU preflight and the synchronous single-env loop."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smolvla_rltoken.paths import (
    ONLINE_RL_CONFIG_PATH,
    ONLINE_RL_OUTPUT_DIR,
    ONLINE_RL_SMOKE_OUTPUT_DIR,
    RL_TOKEN_STAGE1_RUN,
    SFT_LAST_PRETRAINED,
    SFT_OUTPUT_DIR,
)
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.vla.action_bounds import describe_action_bounds
from smolvla_rltoken.vla.evaluation import TASK_PROMPT, check_eval_inputs

ONLINE_RL_SMOKE_JOB_NAME = "smolvla_rltoken_stage2_smoke"
ONLINE_RL_SMOKE_ENV_STEPS = 72
# Must stay >= ONLINE_RL_SMOKE_BATCH_SIZE * ONLINE_RL_SMOKE_CHUNK_LEN so the
# smoke actually reaches the post-warmup offline warm-start.
ONLINE_RL_SMOKE_WARMUP_STEPS = 16
ONLINE_RL_SMOKE_CHUNK_LEN = 4
ONLINE_RL_SMOKE_BATCH_SIZE = 4
ONLINE_RL_SMOKE_OFFLINE_UPDATES = 4
ONLINE_RL_SMOKE_UTD = 2
ONLINE_RL_SMOKE_BUFFER = 64
ONLINE_RL_SMOKE_MAX_EPISODE_STEPS = 12
ONLINE_RL_SMOKE_HIDDEN = 32
ONLINE_RL_SMOKE_BC_PRETRAIN = 4
# Long enough that a 7-step mock episode can finish inside each probe window, so
# the smoke records probe_ref and probe_det episodes and not just the phase
# transitions; short enough that Actor control still runs between probe cycles.
ONLINE_RL_SMOKE_PROBE_EVERY = 1
ONLINE_RL_SMOKE_PROBE_STEPS = 8
ONLINE_RL_GPU_SMOKE_JOB_NAME = "smolvla_rltoken_stage2_gpu_smoke"
ONLINE_RL_GPU_SMOKE_NUM_ENVS = 4
# Four chunk rounds with two warmup rounds: the late-write buffer then holds a
# full batch when warmup ends, so the GPU smoke exercises the offline warm-start.
ONLINE_RL_GPU_SMOKE_CHUNKS = 4
ONLINE_RL_GPU_SMOKE_WARMUP_CHUNKS = 2
ONLINE_RL_SMOKE_SUCCESS_AT = 7
EPISODE_METRIC_WINDOW = 20
EPISODES_FILENAME = "episodes.jsonl"
# How many times warmup may be extended while the Actor fails the handover
# fidelity gate before the run gives up. Only warmup has run at that point, so
# aborting is cheap compared with letting a broken Actor own the rollout.
HANDOVER_MAX_ATTEMPTS = 5


@dataclass
class OnlineRLCheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        details = "\n".join(f"  - {item}" for item in self.errors)
        raise FileNotFoundError(f"Stage 2 preflight failed:\n{details}")


@dataclass
class OnlineRLTrainResult:
    env_steps: int
    output_dir: str
    buffer_size: int
    did_offline: bool
    used_actor: bool
    warmup_reference_matched: bool
    offline_updates: int = 0
    episodes: int = 0
    successes: int = 0
    vla_episodes: int = 0
    vla_successes: int = 0
    actor_episodes: int = 0
    actor_successes: int = 0
    det_episodes: int = 0
    det_successes: int = 0
    gradient_steps: int = 0
    reconfigures: int = 0
    bc_pretrain_updates: int = 0
    handover_bc_dist: float = float("inf")
    handover_attempts: int = 0
    reference_probes: int = 0
    deterministic_probes: int = 0


class EpisodeTracker:
    """Episode success accounting, split by who controlled the chunks.

    Episodes that mix warmup and Actor chunks count as Actor episodes, so the
    ``vla_*`` numbers stay comparable with the pure-VLA SFT evaluation.

    ``mode`` separates the four ways an episode can be produced. ``warmup`` and
    ``probe_ref`` are both pure frozen VLA and share the ``vla_*`` counters;
    ``actor`` is the stochastic on-policy rollout; ``probe_det`` is the Actor
    with exploration noise disabled and gets its own ``det_*`` counters. The
    ``det`` bucket is what separates "the Actor is worse" from "the exploration
    noise is expensive", which the stochastic rate alone cannot distinguish.
    """

    MODES = ("warmup", "actor", "probe_ref", "probe_det")

    def __init__(self, path: Path | None = None, window: int = EPISODE_METRIC_WINDOW):
        self.path = path
        self.window = window
        self.episodes = 0
        self.successes = 0
        self.vla_episodes = 0
        self.vla_successes = 0
        self.actor_episodes = 0
        self.actor_successes = 0
        self.det_episodes = 0
        self.det_successes = 0
        self._recent_success: deque[bool] = deque(maxlen=window)
        self._recent_steps: deque[int] = deque(maxlen=window)
        # Cumulative rates hide the trend once tens of thousands of episodes are
        # in. The windowed Actor rate against the windowed reference rate is the
        # comparison that decides whether the run is working.
        self._recent_actor: deque[bool] = deque(maxlen=window)
        self._recent_vla: deque[bool] = deque(maxlen=window)
        self._recent_det: deque[bool] = deque(maxlen=window)

    def record(self, outcome: Any, env_steps: int, mode: str | None = None) -> None:
        if mode is None:
            mode = "actor" if outcome.use_actor else "warmup"
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.episodes += 1
        self.successes += int(outcome.success)
        if mode == "probe_det":
            self.det_episodes += 1
            self.det_successes += int(outcome.success)
            self._recent_det.append(bool(outcome.success))
        elif mode == "actor":
            self.actor_episodes += 1
            self.actor_successes += int(outcome.success)
            self._recent_actor.append(bool(outcome.success))
        else:
            self.vla_episodes += 1
            self.vla_successes += int(outcome.success)
            self._recent_vla.append(bool(outcome.success))
        self._recent_success.append(bool(outcome.success))
        self._recent_steps.append(int(outcome.steps))
        if self.path is None:
            return
        payload = {
            "episode_id": int(outcome.episode_id),
            "env_index": int(getattr(outcome, "env_index", 0)),
            "use_actor": bool(outcome.use_actor),
            "mode": mode,
            "steps": int(outcome.steps),
            "success": bool(outcome.success),
            "terminated": bool(outcome.terminated),
            "truncated": bool(outcome.truncated),
            "env_steps": int(env_steps),
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "episodes": float(self.episodes),
            "successes": float(self.successes),
        }
        if self.vla_episodes:
            out["vla_success_rate"] = self.vla_successes / self.vla_episodes
        if self.actor_episodes:
            out["actor_success_rate"] = self.actor_successes / self.actor_episodes
        if self.det_episodes:
            out["det_success_rate"] = self.det_successes / self.det_episodes
        if self._recent_vla:
            out[f"vla_success_rate_{self.window}"] = sum(self._recent_vla) / len(self._recent_vla)
        if self._recent_actor:
            out[f"actor_success_rate_{self.window}"] = sum(self._recent_actor) / len(
                self._recent_actor
            )
        if self._recent_det:
            out[f"det_success_rate_{self.window}"] = sum(self._recent_det) / len(self._recent_det)
        if self._recent_success:
            out[f"success_rate_{self.window}"] = sum(self._recent_success) / len(
                self._recent_success
            )
            out[f"episode_steps_{self.window}"] = sum(self._recent_steps) / len(self._recent_steps)
        return out


def is_smoke_output_root(output_dir: str | Path, *, root: str | Path | None = None) -> bool:
    parent = Path(root) if root is not None else ONLINE_RL_SMOKE_OUTPUT_DIR
    return Path(output_dir).resolve() == parent.resolve()


def is_throwaway_smoke_output(output_dir: str | Path, *, root: str | Path | None = None) -> bool:
    """True for the smoke parent or any per-run subdirectory under it."""
    parent = Path(root) if root is not None else ONLINE_RL_SMOKE_OUTPUT_DIR
    path = Path(output_dir).resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def is_formal_output_root(output_dir: str | Path, *, root: str | Path | None = None) -> bool:
    parent = Path(root) if root is not None else ONLINE_RL_OUTPUT_DIR
    return Path(output_dir).resolve() == parent.resolve()


def smoke_run_tag(cfg: OnlineRLConfig) -> str:
    """Directory tag: mock_smoke or gpu_smoke_envN."""
    if cfg.mock_env or cfg.mock_vla:
        return "mock_smoke"
    return f"gpu_smoke_env{int(cfg.num_envs)}"


def new_run_directory(root: str | Path, *, when: datetime | None = None) -> Path:
    return new_tagged_run_directory(root, tag="run", when=when)


def new_tagged_run_directory(
    root: str | Path,
    *,
    tag: str,
    when: datetime | None = None,
) -> Path:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    safe_tag = tag.strip().replace("/", "_").replace(" ", "_")
    if not safe_tag:
        raise ValueError("run tag must not be empty")
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{safe_tag}_{stamp}"
    if path.exists():
        for index in range(2, 100):
            candidate = parent / f"{safe_tag}_{stamp}_{index}"
            if not candidate.exists():
                path = candidate
                break
        else:
            raise FileExistsError(f"could not allocate a free run directory under {parent}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def allocate_formal_run_dir(
    cfg: OnlineRLConfig,
    *,
    when: datetime | None = None,
    root: str | Path | None = None,
) -> OnlineRLConfig:
    parent = Path(root) if root is not None else ONLINE_RL_OUTPUT_DIR
    if is_throwaway_smoke_output(cfg.output_dir) or not is_formal_output_root(cfg.output_dir, root=parent):
        return cfg
    run_dir = new_run_directory(cfg.output_dir, when=when)
    stamp = run_dir.name.removeprefix("run_")
    cfg.output_dir = str(run_dir)
    if stamp not in cfg.job_name:
        cfg.job_name = f"{cfg.job_name}_{stamp}"
    return cfg


def allocate_smoke_run_dir(
    cfg: OnlineRLConfig,
    *,
    when: datetime | None = None,
    root: str | Path | None = None,
) -> OnlineRLConfig:
    """Create ``<tag>_YYYYMMDD_HHMMSS`` under the smoke parent. Never overwrite siblings."""
    parent = Path(root) if root is not None else ONLINE_RL_SMOKE_OUTPUT_DIR
    if not is_smoke_output_root(cfg.output_dir, root=parent):
        return cfg
    run_dir = new_tagged_run_directory(parent, tag=smoke_run_tag(cfg), when=when)
    cfg.output_dir = str(run_dir)
    cfg.job_name = run_dir.name
    return cfg


def apply_smoke_overrides(cfg: OnlineRLConfig) -> OnlineRLConfig:
    """Short mock loop. CLI allocates a tagged child under the smoke parent."""
    cfg.total_env_steps = ONLINE_RL_SMOKE_ENV_STEPS
    cfg.warmup_env_steps = ONLINE_RL_SMOKE_WARMUP_STEPS
    cfg.chunk_len = ONLINE_RL_SMOKE_CHUNK_LEN
    cfg.batch_size = ONLINE_RL_SMOKE_BATCH_SIZE
    cfg.offline_updates_after_warmup = ONLINE_RL_SMOKE_OFFLINE_UPDATES
    cfg.utd = ONLINE_RL_SMOKE_UTD
    cfg.buffer_capacity = ONLINE_RL_SMOKE_BUFFER
    cfg.max_episode_steps = ONLINE_RL_SMOKE_MAX_EPISODE_STEPS
    cfg.hidden_dim = ONLINE_RL_SMOKE_HIDDEN
    cfg.bc_pretrain_updates = ONLINE_RL_SMOKE_BC_PRETRAIN
    cfg.reference_probe_every_episodes = ONLINE_RL_SMOKE_PROBE_EVERY
    cfg.reference_probe_env_steps = ONLINE_RL_SMOKE_PROBE_STEPS
    cfg.log_freq = 8
    cfg.save_freq = ONLINE_RL_SMOKE_ENV_STEPS
    cfg.output_dir = str(ONLINE_RL_SMOKE_OUTPUT_DIR)
    cfg.job_name = ONLINE_RL_SMOKE_JOB_NAME
    cfg.mock_env = True
    cfg.mock_vla = True
    # Let the mock episodes actually succeed so the smoke covers the sparse
    # reward -> TD -> episode metric path instead of an all-zero reward loop.
    cfg.mock_success_at = ONLINE_RL_SMOKE_SUCCESS_AT
    cfg.device = "cpu"
    cfg.wandb_enable = False
    cfg.wandb_disable_artifact = True
    # Formal YAML uses 16 parallel envs; keep the mock smoke single-env and cheap.
    cfg.num_envs = 1
    cfg.reconfigure_every_episodes = 0
    return cfg


def scale_gpu_smoke_horizon(cfg: OnlineRLConfig) -> OnlineRLConfig:
    n_envs = max(1, int(cfg.num_envs))
    cfg.total_env_steps = n_envs * cfg.chunk_len * ONLINE_RL_GPU_SMOKE_CHUNKS
    cfg.warmup_env_steps = n_envs * cfg.chunk_len * ONLINE_RL_GPU_SMOKE_WARMUP_CHUNKS
    cfg.batch_size = max(1, min(cfg.buffer_capacity, n_envs))
    cfg.log_freq = max(1, n_envs * cfg.chunk_len)
    cfg.save_freq = cfg.total_env_steps
    return cfg


def apply_gpu_smoke_overrides(cfg: OnlineRLConfig, *, num_envs: int | None = None) -> OnlineRLConfig:
    """Short real VLA+ManiSkill loop with parallel envs. CLI allocates a tagged child."""
    cfg.num_envs = int(num_envs or ONLINE_RL_GPU_SMOKE_NUM_ENVS)
    cfg.reconfigure_every_episodes = cfg.num_envs
    cfg.chunk_len = 10
    cfg.vla_horizon = 50
    cfg.max_episode_steps = 200
    cfg.buffer_capacity = 256
    cfg.offline_updates_after_warmup = 4
    cfg.bc_pretrain_updates = ONLINE_RL_SMOKE_BC_PRETRAIN
    # Too few episodes finish in four chunk rounds for a probe to be meaningful.
    cfg.reference_probe_every_episodes = 0
    cfg.utd = 2
    cfg.output_dir = str(ONLINE_RL_SMOKE_OUTPUT_DIR)
    cfg.job_name = ONLINE_RL_GPU_SMOKE_JOB_NAME
    cfg.mock_env = False
    cfg.mock_vla = False
    cfg.device = "cuda"
    cfg.wandb_enable = False
    cfg.wandb_disable_artifact = True
    return scale_gpu_smoke_horizon(cfg)


def apply_cli_overrides(cfg: OnlineRLConfig, args: Any) -> OnlineRLConfig:
    if getattr(args, "smoke", False):
        apply_smoke_overrides(cfg)
    if getattr(args, "gpu_smoke", False):
        apply_gpu_smoke_overrides(cfg, num_envs=getattr(args, "num_envs", None))
    elif getattr(args, "num_envs", None) is not None:
        cfg.num_envs = int(args.num_envs)
    if getattr(args, "vla_checkpoint", None) is not None:
        cfg.vla_checkpoint = args.vla_checkpoint
    if getattr(args, "rl_token_checkpoint", None) is not None:
        cfg.rl_token_checkpoint = args.rl_token_checkpoint
    if getattr(args, "output_dir", None) is not None:
        cfg.output_dir = args.output_dir
    if getattr(args, "device", None) is not None:
        cfg.device = args.device
    if getattr(args, "total_env_steps", None) is not None:
        cfg.total_env_steps = args.total_env_steps
    return cfg


def check_online_rl(cfg: OnlineRLConfig) -> OnlineRLCheckResult:
    """CPU-only preflight: YAML locks + paths. Does not load SmolVLA or ManiSkill."""
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {
        "vla_checkpoint": cfg.vla_checkpoint,
        "rl_token_checkpoint": cfg.rl_token_checkpoint,
        "output_dir": cfg.output_dir,
        "chunk_len": cfg.chunk_len,
        "vla_horizon": cfg.vla_horizon,
        "stride": cfg.stride,
        "use_residual_actor": cfg.use_residual_actor,
        "human_intervention": cfg.human_intervention,
        "num_envs": cfg.num_envs,
        "reconfigure_every_episodes": cfg.reconfigure_every_episodes,
        "reward": cfg.reward,
        "success_sample_frac": cfg.success_sample_frac,
        "reward_sample_frac": cfg.reward_sample_frac,
        "actor_success_sample_frac": cfg.actor_success_sample_frac,
        "actor_reward_sample_frac": cfg.actor_reward_sample_frac,
        "bc_reduction": cfg.bc_reduction,
        "bc_beta": cfg.bc_beta,
        "bc_pretrain_updates": cfg.bc_pretrain_updates,
        "handover_bc_threshold": cfg.handover_bc_threshold,
        "explore_std": cfg.explore_std,
        "ref_dropout": cfg.ref_dropout,
        "reference_probe_every_episodes": cfg.reference_probe_every_episodes,
        "reference_probe_env_steps": cfg.probe_env_steps(),
        "probe_deterministic_actor": cfg.probe_deterministic_actor,
        "mock_env": cfg.mock_env,
        "mock_vla": cfg.mock_vla,
        "config_path": str(ONLINE_RL_CONFIG_PATH),
    }

    if cfg.stride != 1:
        errors.append(f"V1 locks stride=1 (no chunk subsample); got {cfg.stride}")
    if not cfg.use_residual_actor:
        warnings.append(
            "use_residual_actor=false reverts to the V1 non-residual Actor, which measured "
            "0.9-2.4% success against a 15% frozen-VLA baseline because it cannot reproduce "
            "the SFT chunk on handover (stage2_survey.md 13.3). Ablation only"
        )
    if cfg.bc_reduction not in {"sum", "mean"}:
        errors.append(f"bc_reduction must be 'sum' or 'mean'; got {cfg.bc_reduction!r}")
    elif cfg.bc_reduction == "mean" and cfg.bc_beta < 10:
        warnings.append(
            f"bc_reduction='mean' divides the paper's squared L2 by chunk_len * action_dim = "
            f"{cfg.chunk_len * cfg.action_dim}, so bc_beta={cfg.bc_beta} is a much weaker anchor "
            "than it looks; the measured failure had bc_beta=1.0 with 'mean'"
        )
    if cfg.explore_std < 0:
        errors.append(f"explore_std must be non-negative; got {cfg.explore_std}")
    if cfg.bc_pretrain_updates < 0:
        errors.append(f"bc_pretrain_updates must be non-negative; got {cfg.bc_pretrain_updates}")
    if cfg.handover_bc_threshold <= 0:
        errors.append(
            f"handover_bc_threshold must be positive; got {cfg.handover_bc_threshold}"
        )
    for name in ("actor_success_sample_frac", "actor_reward_sample_frac"):
        value = getattr(cfg, name)
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]; got {value}")
    if cfg.actor_success_sample_frac + cfg.actor_reward_sample_frac > 1.0:
        errors.append(
            "actor_success_sample_frac + actor_reward_sample_frac must be <= 1; got "
            f"{cfg.actor_success_sample_frac} + {cfg.actor_reward_sample_frac}"
        )
    if cfg.reference_probe_env_steps < 0:
        errors.append(
            f"reference_probe_env_steps must be non-negative; got {cfg.reference_probe_env_steps}"
        )
    if cfg.reference_probe_every_episodes < 0:
        errors.append(
            "reference_probe_every_episodes must be non-negative; got "
            f"{cfg.reference_probe_every_episodes}"
        )
    elif cfg.reference_probe_every_episodes == 0:
        warnings.append(
            "reference_probe_every_episodes=0 disables the frozen-VLA probe, so "
            "vla_success_rate stays frozen at its warmup value for the whole run"
        )
    if cfg.probe_deterministic_actor and cfg.reference_probe_every_episodes == 0:
        warnings.append(
            "probe_deterministic_actor=true has no effect while "
            "reference_probe_every_episodes=0; the noise-free Actor rate "
            "(det_success_rate) will never be measured"
        )
    if cfg.human_intervention:
        errors.append("V1 does not implement human intervention")
    if cfg.num_envs < 1:
        errors.append(f"num_envs must be positive; got {cfg.num_envs}")
    elif cfg.num_envs > 1:
        # Parallel collect is allowed, but ManiSkill freezes peg/box geometry when
        # num_envs > 1 (reconfiguration_freq=0), so a run without periodic
        # reconfigure would train on only num_envs distinct pegs from start to end.
        if cfg.reconfigure_every_episodes <= 0:
            message = (
                f"num_envs={cfg.num_envs} freezes PegInsertion geometry: ManiSkill uses "
                "reconfiguration_freq=0 for parallel envs, so only num_envs distinct pegs "
                "would ever be sampled. Set reconfigure_every_episodes > 0"
            )
            if is_throwaway_smoke_output(cfg.output_dir):
                warnings.append(f"{message} (tolerated for smoke runs)")
            else:
                errors.append(message)
        elif cfg.reconfigure_every_episodes < cfg.num_envs:
            warnings.append(
                f"reconfigure_every_episodes={cfg.reconfigure_every_episodes} is below "
                f"num_envs={cfg.num_envs}: a full reconfigure drops in-flight episodes, so "
                "this reconfigures more than once per batch wave"
            )
    if cfg.num_envs == 1 and cfg.reconfigure_every_episodes:
        warnings.append(
            "reconfigure_every_episodes is ignored for num_envs=1; ManiSkill already "
            "reconfigures every episode (reconfiguration_freq=1)"
        )
    if cfg.reward != "sparse_success":
        errors.append(f"official RL reward must be sparse_success; got {cfg.reward!r}")
    if not 0.0 <= cfg.success_sample_frac <= 1.0:
        errors.append(
            f"success_sample_frac must be in [0, 1]; got {cfg.success_sample_frac}"
        )
    if not 0.0 <= cfg.reward_sample_frac <= 1.0:
        errors.append(
            f"reward_sample_frac must be in [0, 1]; got {cfg.reward_sample_frac}"
        )
    if cfg.success_sample_frac + cfg.reward_sample_frac > 1.0:
        errors.append(
            f"success_sample_frac + reward_sample_frac must be <= 1; got "
            f"{cfg.success_sample_frac} + {cfg.reward_sample_frac}"
        )
    if cfg.dense_reward_debug:
        warnings.append("dense_reward_debug is on; official numbers must use sparse success")
    if cfg.chunk_len < 1:
        errors.append("chunk_len must be positive")
    if cfg.vla_horizon < cfg.chunk_len:
        errors.append("vla_horizon must be >= chunk_len")
    if cfg.action_bound_margin <= 0:
        errors.append(f"action_bound_margin must be positive; got {cfg.action_bound_margin}")
    if cfg.action_dim != 8:
        errors.append(f"PegInsertion action_dim must be 8; got {cfg.action_dim}")
    if cfg.proprio_dim != 9:
        errors.append(f"PegInsertion proprio_dim must be 9; got {cfg.proprio_dim}")
    if cfg.task != TASK_PROMPT:
        errors.append(f"task must match SFT eval {TASK_PROMPT!r}")
    if not cfg.mock_vla:
        if cfg.chunk_len != 10:
            errors.append(f"V1 chunk_len is 10 (execute 10, not SFT 35/50); got {cfg.chunk_len}")
        if cfg.vla_horizon != 50:
            errors.append(f"V1 vla_horizon is 50; got {cfg.vla_horizon}")
        # One executed chunk is one transition, so warmup must fill a full batch;
        # otherwise the post-warmup offline warm-start silently does nothing and
        # the Actor controls the env with untrained weights.
        needed = cfg.batch_size * cfg.chunk_len
        if cfg.warmup_env_steps < needed:
            errors.append(
                f"warmup_env_steps={cfg.warmup_env_steps} yields "
                f"{cfg.warmup_env_steps // max(1, cfg.chunk_len)} transitions, below "
                f"batch_size={cfg.batch_size}; need warmup_env_steps >= "
                f"batch_size * chunk_len = {needed}"
            )
        eval_info = check_eval_inputs(cfg.vla_checkpoint, cfg.camera_config)
        errors.extend(eval_info.get("errors") or [])
        info["eval_preflight"] = {k: v for k, v in eval_info.items() if k != "errors"}
        token_path = Path(cfg.rl_token_checkpoint)
        if not token_path.is_file():
            errors.append(
                f"missing Stage 1 encoder checkpoint: {token_path}. "
                f"Expected something like {RL_TOKEN_STAGE1_RUN}"
            )
        vla_path = Path(cfg.vla_checkpoint)
        if vla_path.resolve() == SFT_OUTPUT_DIR.resolve():
            errors.append(
                f"vla_checkpoint looks like the SFT run root. Pass {SFT_LAST_PRETRAINED}"
            )
        else:
            try:
                info["action_bounds"] = describe_action_bounds(
                    cfg.vla_checkpoint,
                    margin=cfg.action_bound_margin,
                    action_dim=cfg.action_dim,
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"cannot derive normalized action bounds: {exc}")

    info["ok"] = not errors
    return OnlineRLCheckResult(ok=not errors, errors=errors, warnings=warnings, info=info)


def _build_frozen_planner(cfg: OnlineRLConfig, agent) -> Any:
    from lerobot.policies.factory import make_pre_post_processors

    from smolvla_rltoken.rlt.load import load_frozen_encoder
    from smolvla_rltoken.rollout.planner import FrozenVLAPlanner
    from smolvla_rltoken.vla.action_bounds import normalized_action_bounds
    from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor
    from smolvla_rltoken.vla.load import load_smolvla_policy

    policy = load_smolvla_policy(cfg.vla_checkpoint, device=cfg.device, dtype=cfg.dtype)
    policy.eval()
    policy.requires_grad_(False)
    encoder = load_frozen_encoder(cfg.rl_token_checkpoint, device=cfg.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(cfg.vla_checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    bounds = normalized_action_bounds(
        cfg.vla_checkpoint,
        margin=cfg.action_bound_margin,
        action_dim=cfg.action_dim,
    )
    return FrozenVLAPlanner(
        extractor=SmolVLAPrefixExtractor(policy),
        encoder=encoder,
        actor=agent.actor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
        proprio_dim=cfg.proprio_dim,
        image_only=cfg.use_image_tokens_only,
        task=cfg.task,
        action_bounds=bounds,
        explore_std=cfg.explore_std,
    )


def train_online_rl(
    cfg: OnlineRLConfig,
    *,
    planner=None,
    env=None,
    agent=None,
) -> OnlineRLTrainResult:
    """Synchronous collect/update loop. VLA and the RL Token encoder stay frozen."""
    import torch

    from smolvla_rltoken.envs.chunk_env import MockChunkEnv, make_chunk_env
    from smolvla_rltoken.rl.agent import RLTAgent
    from smolvla_rltoken.rl.replay import ChunkReplayBuffer
    from smolvla_rltoken.rollout.collector import ChunkCollector
    from smolvla_rltoken.rollout.planner import MockPlanner

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if agent is None:
        agent = RLTAgent(cfg, device=cfg.device)
    if env is None:
        if cfg.mock_env:
            env = MockChunkEnv(
                success_at=cfg.mock_success_at,
                max_episode_steps=cfg.max_episode_steps,
                action_dim=cfg.action_dim,
                num_envs=cfg.num_envs,
            )
        else:
            import json

            camera_specs = json.loads(Path(cfg.camera_config).read_text())
            env = make_chunk_env(
                camera_specs,
                sim_backend=cfg.sim_backend,
                max_episode_steps=cfg.max_episode_steps,
                dense_reward_debug=cfg.dense_reward_debug,
                action_dim=cfg.action_dim,
                num_envs=cfg.num_envs,
            )
    if planner is None:
        if cfg.mock_vla:
            planner = MockPlanner(
                chunk_len=cfg.chunk_len,
                action_dim=cfg.action_dim,
                rl_token_dim=cfg.rl_token_dim,
                proprio_dim=cfg.proprio_dim,
            )
        else:
            planner = _build_frozen_planner(cfg, agent)

    replay = ChunkReplayBuffer(
        cfg.buffer_capacity,
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
        device=cfg.device,
    )
    if cfg.num_envs > 1:
        from smolvla_rltoken.rollout.collector import BatchedChunkCollector

        collector = BatchedChunkCollector(
            env,
            planner,
            replay,
            chunk_len=cfg.chunk_len,
            action_dim=cfg.action_dim,
            num_envs=cfg.num_envs,
            reconfigure_every_episodes=cfg.reconfigure_every_episodes,
        )
    else:
        collector = ChunkCollector(
            env, planner, replay, chunk_len=cfg.chunk_len, action_dim=cfg.action_dim
        )
    collector.reset(seed=cfg.seed)

    env_steps = 0
    did_offline = False
    offline_updates = 0
    used_actor = False
    warmup_reference_matched = True
    gradient_steps = 0
    metrics: dict[str, float] = {}
    handover_done = False
    handover_attempts = 0
    handover_deadline = cfg.warmup_env_steps
    handover_bc_dist = float("inf")
    bc_pretrain_updates = 0
    probe_end_steps = 0
    probe_phase = ""
    reference_probes = 0
    deterministic_probes = 0
    episodes_since_probe = 0
    probe_window = cfg.probe_env_steps()
    tracker = EpisodeTracker(path=out_dir / EPISODES_FILENAME)
    run = None
    if cfg.wandb_enable:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.job_name,
            config=cfg.to_dict(),
            dir=str(out_dir),
        )

    while env_steps < cfg.total_env_steps:
        warmup = not handover_done and env_steps < handover_deadline
        if (not handover_done) and env_steps >= handover_deadline:
            handover_attempts += 1
            # The Actor must reproduce the frozen VLA chunk before it owns the
            # rollout. A residual Actor passes this at zero cost; a drifted one
            # is caught here instead of after a million wasted env steps.
            bc_metrics = agent.bc_pretrain(replay, cfg.bc_pretrain_updates)
            bc_pretrain_updates += int(bc_metrics.get("bc_updates", 0.0))
            handover_bc_dist = agent.reference_fidelity(replay)
            passed = handover_bc_dist <= cfg.handover_bc_threshold
            print(
                f"[stage2] handover attempt={handover_attempts} "
                f"bc_updates={int(bc_metrics.get('bc_updates', 0.0))} "
                f"bc_dist_det={handover_bc_dist:.3e} "
                f"threshold={cfg.handover_bc_threshold:.3e} passed={passed}",
                flush=True,
            )
            if not passed:
                if handover_attempts >= HANDOVER_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Actor failed the handover fidelity gate {handover_attempts} times: "
                        f"bc_dist_det={handover_bc_dist:.3e} > "
                        f"handover_bc_threshold={cfg.handover_bc_threshold:.3e}. Letting it "
                        "control the env would collapse the rollout below the frozen-VLA "
                        "baseline. Check use_residual_actor, bc_pretrain_updates and bc_beta."
                    )
                handover_deadline += probe_window
                warmup = True
                print(
                    f"[stage2] warmup extended to {handover_deadline} env steps",
                    flush=True,
                )
            else:
                offline_metrics = agent.run_offline_updates(
                    replay, cfg.offline_updates_after_warmup
                )
                offline_updates = int(offline_metrics.pop("offline_updates", 0.0))
                did_offline = offline_updates > 0
                if cfg.offline_updates_after_warmup > 0 and offline_updates == 0:
                    print(
                        "[stage2] WARNING: offline warm-start skipped "
                        f"(buffer={len(replay)} < batch_size={cfg.batch_size}). Raise "
                        "warmup_env_steps to at least batch_size * chunk_len = "
                        f"{cfg.batch_size * cfg.chunk_len}.",
                        flush=True,
                    )
                else:
                    print(f"[stage2] offline warm-start updates={offline_updates}", flush=True)
                    metrics = offline_metrics
                # The warm-start runs Q gradients, so report how far it moved
                # the Actor off the reference before it takes control.
                post = agent.reference_fidelity(replay)
                print(f"[stage2] post-warm-start bc_dist_det={post:.3e}", flush=True)
                handover_done = True
                warmup = False
                # Probes are spaced from the start of Actor control, not from
                # the start of the run: warmup episodes are already reference.
                episodes_since_probe = 0

        if (
            handover_done
            and not probe_phase
            and cfg.reference_probe_every_episodes > 0
            and episodes_since_probe >= cfg.reference_probe_every_episodes
        ):
            # A probe must not share episodes with Actor chunks, so the batch is
            # reset on the way in and on the way out. Both resets reconfigure,
            # which is the same cost the periodic reconfigure already pays.
            collector.reset(options={"reconfigure": True})
            probe_phase = "reference"
            probe_end_steps = env_steps + probe_window
            reference_probes += 1
            print(f"[stage2] reference probe #{reference_probes} for {probe_window} steps", flush=True)

        probing = bool(probe_phase)
        if probe_phase == "reference":
            chunk_actor, chunk_deterministic, mode = False, False, "probe_ref"
        elif probe_phase == "actor_det":
            chunk_actor, chunk_deterministic, mode = True, True, "probe_det"
        else:
            chunk_actor, chunk_deterministic = not warmup, False
            mode = "actor" if chunk_actor else "warmup"

        result = collector.run_chunk(use_actor=chunk_actor, deterministic=chunk_deterministic)
        env_steps += result.n_steps
        if not result.use_actor:
            # Reference chunks must be executed verbatim: no clip, no margin.
            warmup_reference_matched &= result.executed_matches_reference
        else:
            used_actor = True
        for outcome in result.finished:
            tracker.record(outcome, env_steps, mode=mode)
        if not probe_phase:
            episodes_since_probe += len(result.finished)
        if result.episode_done:
            collector.reset()
        if probe_phase and env_steps >= probe_end_steps:
            collector.reset(options={"reconfigure": True})
            if probe_phase == "reference" and cfg.probe_deterministic_actor:
                # Back to back with the reference probe and on the same
                # reconfigured batch, so the pair isolates the cost of the
                # exploration noise from the cost of the learned residual.
                probe_phase = "actor_det"
                probe_end_steps = env_steps + probe_window
                deterministic_probes += 1
                print(
                    f"[stage2] deterministic-actor probe #{deterministic_probes} "
                    f"for {probe_window} steps",
                    flush=True,
                )
            else:
                probe_phase = ""
                probe_end_steps = 0
                episodes_since_probe = 0
        if handover_done and len(replay) >= cfg.batch_size:
            # UTD is per transition, not per collect call: N parallel envs add N
            # transitions per chunk, so a fixed count would silently divide the
            # gradient-to-data ratio by N.
            critic_sampler = agent.critic_sampler(replay)
            actor_sampler = agent.actor_sampler(replay)
            for _ in range(cfg.utd * result.added):
                metrics = agent.update(critic_sampler, actor_sampler)
                gradient_steps += 1
        if env_steps % cfg.log_freq < result.n_steps or env_steps >= cfg.total_env_steps:
            episode_metrics = tracker.metrics()
            pool_metrics = {
                "replay_success_slots": float(replay.n_success_slots),
                "replay_reward_slots": float(replay.n_reward_slots),
            }
            line = (
                f"[stage2] steps={env_steps} buffer={len(replay)} "
                f"warmup={warmup} probe={probing} offline={offline_updates} "
                + " ".join(
                    f"{k}={v:.4f}"
                    for k, v in {**episode_metrics, **pool_metrics, **metrics}.items()
                )
            )
            print(line, flush=True)
            if run is not None:
                run.log(
                    {
                        "env_steps": env_steps,
                        "buffer": len(replay),
                        "probing": float(probing),
                        "probe_phase_actor_det": float(probe_phase == "actor_det"),
                        "reference_probes": float(reference_probes),
                        "deterministic_probes": float(deterministic_probes),
                        **episode_metrics,
                        **pool_metrics,
                        **metrics,
                    }
                )
        if cfg.save_freq > 0 and (
            env_steps % cfg.save_freq < result.n_steps or env_steps >= cfg.total_env_steps
        ):
            torch.save(
                {"agent": agent.state_dict(), "config": cfg.to_dict(), "env_steps": env_steps},
                out_dir / "online_rl.pt",
            )

    # The trailing mid-episode chunk is a valid transition; do not drop it.
    collector.flush_pending()
    if run is not None:
        run.finish()
    reconfigures = int(getattr(collector, "reconfigure_count", 0))
    print(
        f"[stage2] done; episodes={tracker.episodes} successes={tracker.successes} "
        f"vla={tracker.vla_successes}/{tracker.vla_episodes} "
        f"actor={tracker.actor_successes}/{tracker.actor_episodes} "
        f"det={tracker.det_successes}/{tracker.det_episodes} "
        f"grad_steps={gradient_steps} reconfigures={reconfigures} "
        f"probes={reference_probes}/{deterministic_probes} "
        f"handover_bc_dist={handover_bc_dist:.3e}; "
        f"saved to {out_dir / 'online_rl.pt'}",
        flush=True,
    )
    return OnlineRLTrainResult(
        env_steps=env_steps,
        output_dir=str(out_dir),
        buffer_size=len(replay),
        did_offline=did_offline,
        used_actor=used_actor,
        warmup_reference_matched=warmup_reference_matched,
        offline_updates=offline_updates,
        episodes=tracker.episodes,
        successes=tracker.successes,
        vla_episodes=tracker.vla_episodes,
        vla_successes=tracker.vla_successes,
        actor_episodes=tracker.actor_episodes,
        actor_successes=tracker.actor_successes,
        det_episodes=tracker.det_episodes,
        det_successes=tracker.det_successes,
        gradient_steps=gradient_steps,
        reconfigures=reconfigures,
        bc_pretrain_updates=bc_pretrain_updates,
        handover_bc_dist=handover_bc_dist,
        handover_attempts=handover_attempts,
        reference_probes=reference_probes,
        deterministic_probes=deterministic_probes,
    )


def smoke_config(cfg: OnlineRLConfig | None = None) -> OnlineRLConfig:
    if cfg is None:
        cfg = OnlineRLConfig.from_yaml()
    return apply_smoke_overrides(replace(cfg))
