# SmolVLA + RL Token (RLT) on ManiSkill PegInsertionSide-v1

在仿真里复现 RL Token（RLT）思路：先把 SmolVLA 在 PegInsertion 演示上做 SFT，再把冻结 VLA 的 prefix hidden 压缩成一个 RL Token，最后以「冻结 VLA 给出的 reference chunk」为条件，在线训练一个 chunk 级 Actor-Critic，去修正 VLA 在精细插入阶段的动作。

> **当前状态（2026-11-25，workspace-b）**：代码、配置、测试、实验记录、极简 Conda + `uv` 环境（`smolvla-rlt`）和可移植路径均已就绪；SFT / Stage 1 / Stage 2 checkpoint 仍需在本机重新训练（见 [docs/ROADMAP.md](docs/ROADMAP.md)）。

---

## 1. 整体流程

```text
Stage 0  SFT        smolvla_base ──(1000 条 motion-planning 演示, 3 相机 512²)──▶ SFT policy
Stage 1  RL Token   冻结 SFT VLM prefix hidden [B, M≈192, 960] ──encoder──▶ z_rl ∈ R^512
                    （causal decoder 做 teacher-forcing 重建，loss_ro）
Stage 2  Online RL  x = [z_rl, proprio(9)] ; ã = 冻结 VLA chunk 的前 C=10 步
                    Actor  a = ã + Δ(x, ã)          （残差，末层零初始化）
                    Critic Q(x, ã, (a-ã)/scale)      （twin Q，残差坐标输入）
                    奖励   稀疏 info["success"]，chunk 级 n-step TD
```

| 阶段 | 入口 | 配置（YAML 是唯一真源，CLI 显式传参才覆盖） | 包内实现 |
| --- | --- | --- | --- |
| Stage 0 SFT | `scripts/train_sft.py` / `train_sft.sh` | `configs/vla/smolvla_sft.yaml` | `vla/sft.py`（调用 `lerobot-train`） |
| SFT 评测 | `scripts/eval.py` / `eval_sft.sh` | 命令行 | `vla/evaluation.py`、`benchmark.py` |
| Stage 1 RL Token | `scripts/train_rltoken.py` / `train_rltoken.sh` | `configs/rlt/rl_token.yaml` | `rlt/`、`vla/extractor.py` |
| Stage 2 Online RL | `scripts/train_online_rl.py` / `train_online_rl.sh` | `configs/rl/actor_critic.yaml` | `rl/`、`rollout/`、`envs/chunk_env.py` |
| 3D 回放录制 | `scripts/record_viewer3d.py` | 命令行 | `viewer3d.py`（方案见 `docs/site/viewer3d.md`） |
| 数据 / 相机工具 | `build_three_camera_lerobot_dataset.py`、`tune_cameras.py`、`preview_cameras.py`、`render_camera_videos.py` | `configs/vla/peg_insertion_three_cameras.json` | `envs/maniskill_env.py` |

所有入口都支持 `--check`（只用 CPU 做预检，不加载 VLA 和仿真器）。Stage 0/1/2 支持 `--smoke`，Stage 1 有 `--pressure`，Stage 2 有 `--gpu-smoke`。

## 2. 已有结果（旧 AutoDL 实例，RTX 4090）

| 实验 | 结果 | 记录 |
| --- | --- | --- |
| SFT 20k step，每次推理执行 35/50 步，100 局 | **成功率 22%**（Wilson 95% [15%, 31%]） | `benchamrk/sft/run_20260901_121513.md` |
| Stage 1，带 val，在约 3300/5000 步处中断 | 最佳 `val_loss_ro≈0.0747`（未训练时约 1.1） | `docs/rltoken-training/` |
| Stage 2，非残差 Actor（2 个 run） | 交接后马上崩：Actor 0.9–2.4%，冻结 VLA 15% | `docs/online-rl/stage2_survey.md` §13.3 |
| Stage 2，残差 Actor | 交接成功，但 Actor 被 BC 钉在 reference 上，RL 等于没起作用（约 10.6% vs 15.1%） | §13.4 |
| Stage 2 诊断 | Critic 看不到动作差异（`q_adv_det` 比自身 TD 残差小 14 倍）。已改成残差坐标输入，**还没跑过正式 run** | §13.5 |

**到目前为止，RL 还没有在冻结 VLA 基线之上带来提升。** 下一步怎么做见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 3. 目录结构

```text
configs/
  vla/smolvla_sft.yaml                 Stage 0 SFT
  vla/peg_insertion_three_cameras.json 三相机位姿 / 分辨率（数据集、评测、RL 共用）
  rlt/rl_token.yaml                    Stage 1
  rl/actor_critic.yaml                 Stage 2（每个旋钮都有注释，说明来源实验）
src/smolvla_rltoken/
  paths.py          仓库内产物路径（相对 REPO_ROOT）
  benchmark.py      benchamrk/ Markdown 记录渲染
  vla/              SmolVLA 加载、数据集、prefix 抽取、动作边界、SFT 预检、评测
  rlt/              RL Token 模块（encoder/decoder）、训练、压力测试、加载
  rl/               Actor、Critic、TD、replay、agent 更新、训练循环、GPU smoke
  rollout/          Planner（冻结 VLA + encoder + Actor）、Collector（单/并行）、Transition
  envs/             三相机 PegInsertion 环境、稀疏奖励 chunk env、Mock env
scripts/            薄 CLI + bash 启动器
tests/              CPU 测试（不需要 GPU、仿真器或数据）
benchamrk/          每次评测 / smoke / 压力测试的 Markdown 记录（目录名沿用这个拼写）
docs/
  knowledge-base/   RLT 设计与算法笔记（计划、loss、Actor/Critic 详解、数据管线）
  rltoken-training/ Stage 1 适配进度
  rltoken-extraction-survey/  VLA hidden → RL Token 三仓对照
  online-rl/        Stage 2 调研与用户锁定决策（stage2_survey.md §13 是最新结论）
  ROADMAP.md        重建与优化路线图
  site/             实验记录站内容（progress.yaml、指标 schema、3D 查看器方案），见 AGENTS.md §8
```

## 4. 环境

workspace-b 上采用**极简 Conda 壳 + `uv pip install`**（Miniforge 位于 `~/miniforge3`，`uv` 位于 `~/.local/bin/uv`，全局硬链接缓存 `~/.cache/uv`，不动系统驱动/CUDA）：

```bash
~/miniforge3/bin/conda create -n smolvla-rlt python=3.11 -y
conda activate smolvla-rlt
uv pip install "lerobot[smolvla]==0.4.4" "torch==2.8.0" "torchcodec==0.7.*" mani_skill pytest
# 或按锁文件复现：uv pip install -r requirements-lock.txt
uv pip install --no-build-isolation -e .
python -c "import sapien; print(sapien.render.get_device_summary())"  # 必须识别到 RTX 4090 (Supported: 1)
```

需要的产物（都放在仓库内，已被 `.gitignore` 忽略）：

| 产物 | 路径 | 来源 |
| --- | --- | --- |
| SmolVLA 基座 | `models/lerobot/smolvla_base` | HF `lerobot/smolvla_base` |
| 三相机 LeRobot 数据集 | `data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos` | HF `wkal/smolvla-rlt`（1000 局 / 149,055 帧） |
| ManiSkill 原始演示（只在重建数据集时需要） | `data/maniskill/demos/PegInsertionSide-v1/` | `python -m mani_skill.utils.download_demo PegInsertionSide-v1` |
| SFT / Stage 1 / Stage 2 输出 | `outputs/{sft,rl_token,online_rl}/` | 需要重新训练 |

## 5. 常用命令

```bash
conda activate smolvla-rlt
pytest                                         # 全部 CPU 测试

# Stage 0
python scripts/train_sft.py --check
python scripts/train_sft.py --smoke
bash scripts/train_sft.sh                      # 正式 20k step
bash scripts/eval_sft.sh --check               # 评测，写 outputs/eval + benchamrk/sft

# Stage 1
python scripts/train_rltoken.py --check
python scripts/train_rltoken.py --pressure     # batch 16–64 显存/loss 探测
python scripts/train_rltoken.py                # 正式：outputs/rl_token/run_<utc>/

# Stage 2
python scripts/train_online_rl.py --check      # 会打印 per-dim 动作边界，必须不是 ±1
python scripts/train_online_rl.py --smoke      # Mock env + Mock VLA，全 CPU
python scripts/train_online_rl.py --gpu-smoke --num-envs 4
bash scripts/train_online_rl.sh                # 正式：outputs/online_rl/run_<utc>/

# 3D 回放（写 outputs/viewer3d/<key>.{glb,traj.json,json}）
python scripts/record_viewer3d.py              # 演示回放，physx_cpu，不占显存
python scripts/record_viewer3d.py --checkpoint <pretrained_model> --seed 10000   # SFT 策略，需要 GPU
```

长任务用 `nohup ... > LOG 2>&1 & echo $! > X.pid` 在后台启动，用 `tail --pid` / `grep -m1` 等待，不要固定 `sleep`（详见 workspace 的 CLAUDE.md）。`*.sh` 启动器已支持自动探测 Conda 路径并移除失效旧代理；正式长训练的 agent shell 保护逻辑仍保留（是否放开待确认）。

## 6. 约定

- **评测必须产出 Markdown**：每次评测在 `benchamrk/<stage>/` 下写状态为 `complete|failed|interrupted` 的记录，原始 JSON/JSONL/视频放在 `outputs/`。这部分输出要有 CPU 测试覆盖。
- **算法改动先读文档**：`docs/online-rl/stage2_survey.md`（§13.1–13.5 覆盖前文）和 `docs/knowledge-base/`。代码行为改了，同一次提交里要把对应笔记一起更新。
- 每次正式训练都新建 `run_YYYYMMDD_HHMMSS/` 目录，不写进父目录，也不删除同级的其他 run。

更多 agent 侧的规则见 [AGENTS.md](AGENTS.md)。
