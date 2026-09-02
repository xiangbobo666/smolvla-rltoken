# Stage 2：Online RL

本目录是 **冻结 SmolVLA + 冻结 RL Token encoder 之后的 chunk-level Actor-Critic** 的调研与实现约定。算法公式仍以知识库为准；本目录记录对照论文/三仓之后的**用户锁定决策**和编码切片。

- [stage2_survey.md](stage2_survey.md) — 完整调研：何时介入、rollout / replay / actor / critic、稀疏奖励映射、M5–M8 验收顺序。**写或改 Stage 2 代码前必读**，其中 13.1 节是首轮 code review 后的三处修正（动作边界必须 per-dim 推导而不是 ±1、warmup 步数必须 ≥ `batch_size * chunk_len`、训练循环必须暴露 episode 成功率），13.2 节是并行环境解锁（`num_envs>1` 可用于正式训练，但必须同时设 `reconfigure_every_episodes`，否则 peg 几何全程冻结）。与文件早期表述冲突时以这两节为准。

核心闭环入口：`python scripts/train_online_rl.py --check` / `--smoke` / `--gpu-smoke`；正式训 `bash scripts/train_online_rl.sh`（网页终端）。`--check` 会打印解析出的 per-dim 归一化动作边界，跑正式训练前先确认它不是 ±1。`--gpu-smoke` 用并行环境做短资源探测，产物进 `outputs/online_rl_smoke/<tag>_YYYYMMDD_HHMMSS/`，并写 `benchamrk/rl/`。训练每个 run 目录还会写 `episodes.jsonl`（逐 episode 的 success / steps / 是否 Actor 控制）。正式评测入口仍后置。

相关知识库：

- [SmolVLA_RLT_plan.md](../knowledge-base/SmolVLA_RLT_plan.md) Part 3–6
- [算法实现.md](../knowledge-base/算法实现.md)
- [RLT_actor_critic协同工作.md](../knowledge-base/详解/RLT_actor_critic协同工作.md)
- [RLT_Actor.md](../knowledge-base/详解/RLT_Actor.md)
- [RLT_Critic.md](../knowledge-base/详解/RLT_Critic.md)
- [数据管线.md](../knowledge-base/数据管线.md)

Stage 1 进度仍在 [docs/rltoken-training/](../rltoken-training/README.md)。
