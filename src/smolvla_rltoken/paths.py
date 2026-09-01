"""Local artifact paths for this repository (see AGENTS.md)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHAMRK_DIR = REPO_ROOT / "benchamrk"

HF_HOME = REPO_ROOT / ".cache" / "huggingface"
SMOLVLA_BASE = REPO_ROOT / "models" / "lerobot" / "smolvla_base"
DATASET_REPO_ID = "wkal/smolvla-rlt"
DATASET_ROOT = (
    REPO_ROOT / "data" / "lerobot" / "PegInsertionSide-v1" / "motionplanning_rgb_pd_joint_pos"
)
SFT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft" / "peg_insertion"
SFT_SMOKE_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft" / "peg_insertion_smoke"
SFT_CONFIG_PATH = REPO_ROOT / "configs" / "vla" / "smolvla_sft.yaml"
# LeRobot writes weights here; pass this to Stage 1 --checkpoint, not SFT_OUTPUT_DIR.
SFT_LAST_PRETRAINED = SFT_OUTPUT_DIR / "checkpoints" / "last" / "pretrained_model"
WANDB_PROJECT = "smolvla-rltoken"
RL_TOKEN_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_token"
RL_TOKEN_SMOKE_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_token_smoke"
RL_TOKEN_PRESSURE_OUTPUT_DIR = REPO_ROOT / "outputs" / "rl_token_pressure"
RL_TOKEN_CONFIG_PATH = REPO_ROOT / "configs" / "rlt" / "rl_token.yaml"

# smolvla_base was trained with camera{1,2,3}; PegInsertion uses descriptive keys.
SFT_IMAGE_RENAME_MAP = {
    "observation.images.environment_camera": "observation.images.camera1",
    "observation.images.hand_camera": "observation.images.camera2",
    "observation.images.insertion_camera": "observation.images.camera3",
}
