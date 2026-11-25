"""CPU tests for path/script portability across hosts (ROADMAP Phase 0)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from smolvla_rltoken.paths import (
    ONLINE_RL_CONFIG_PATH,
    REPO_ROOT,
    RL_TOKEN_CONFIG_PATH,
    SFT_CONFIG_PATH,
    resolve_repo_path,
)
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rlt.train import RLTokenTrainConfig
from smolvla_rltoken.vla.sft import SFTConfig

FORBIDDEN_MACHINE_LITERALS = (
    "/root/autodl-tmp",
    "/root/miniconda3",
    "127.0.0.1:18082",
)


def test_configs_and_scripts_have_no_legacy_machine_paths_or_proxies():
    targets = [
        *sorted((REPO_ROOT / "configs").rglob("*.yaml")),
        *sorted((REPO_ROOT / "scripts").glob("*.sh")),
        *sorted((REPO_ROOT / "scripts").glob("*.py")),
    ]
    assert targets
    offenders: list[str] = []
    for path in targets:
        text = path.read_text()
        for literal in FORBIDDEN_MACHINE_LITERALS:
            if literal in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {literal!r}")
    assert not offenders, "\n".join(offenders)


def test_shell_launchers_pass_bash_syntax_check():
    launchers = sorted((REPO_ROOT / "scripts").glob("*.sh"))
    assert launchers
    for launcher in launchers:
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{launcher.name}: {result.stderr}"


def test_resolve_repo_path_handles_relative_and_absolute():
    assert Path(resolve_repo_path("outputs/sft/peg_insertion")) == REPO_ROOT / "outputs/sft/peg_insertion"
    assert resolve_repo_path("/tmp/custom_dir") == "/tmp/custom_dir"


def test_all_yaml_configs_resolve_paths_under_repo_root():
    sft = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    rlt = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    rl = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)

    for value in (
        sft.policy_path,
        sft.dataset_root,
        sft.output_dir,
        rlt.checkpoint,
        rlt.dataset_root,
        rlt.output_dir,
        rl.vla_checkpoint,
        rl.rl_token_checkpoint,
        rl.output_dir,
        rl.camera_config,
    ):
        resolved = Path(value)
        assert resolved.is_absolute(), value
        assert REPO_ROOT in resolved.parents, value
