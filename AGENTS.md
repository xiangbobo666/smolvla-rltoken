# Project instructions（smolvla-rltoken）

项目概览、目录结构和常用命令见 [README.md](README.md)；重建与优化计划见 [docs/ROADMAP.md](docs/ROADMAP.md)。本文件只写 agent 必须遵守的规则和资源登记表。

## 1. 运行环境（workspace-b，2026-11-25 迁移后）

- 工作区：workspace-b，Ubuntu / Linux。仓库位于 `/home/dev/workspace/smolvla-rltoken`，用户为 `dev`，无 root 权限。
- 机器是**多用户共享**的 RTX 4090（24 GB）。启动任何 GPU 任务前先运行 `nvidia-smi` 看剩余显存，不要抢占或终止其他用户的进程。梳理时其他用户已占用约 21 GB。
- 依赖装在用户目录：Conda 在 `~/miniforge3`，环境优先建在独立的 Conda env 里。不要用 sudo，不要修改系统 NVIDIA 驱动、系统 CUDA、全局库或其他用户的环境。确实需要系统级改动（例如 Vulkan 所需的 `libegl1` / `nvidia_icd.json`）时，先向用户说明必要性、影响、风险和回滚方式，拿到当次授权后再执行。
- 不查询、也不输出主机名、真实 IP、地理位置或连接信息。描述环境时只用「workspace-b / Ubuntu」。
- 长任务：用 `nohup ... > LOG 2>&1 & echo $! > X.pid` 启动，用 `timeout N tail --pid=... -f /dev/null` 或 `grep -m1` 配合 `tail --pid -F` 等待，禁止固定 `sleep`。每个任务用独立的 LOG 和 PID 文件。完整规则见 workspace 的 CLAUDE.md。
- 下载：只从官方上游（PyPI、Hugging Face、GitHub、ManiSkill 官方）下载，不使用 Aliyun、清华等国内镜像。旧的本地代理 `[IP]:18082` 是 AutoDL 时期的，**在本机已失效**；需要代理时向用户确认。

## 2. 资源登记表

这是 agent 查找本地资源的唯一依据。新增、移动、替换或删除数据集、模型、checkpoint、缓存或 Conda 环境时，在同一次改动里更新本节。登记前先确认路径确实存在；过时的条目要删除或明确标为 ❌。

所有大产物放在仓库内已被 gitignore 的目录（`data/`、`models/`、`outputs/`、`.cache/`）。`/home` 分区剩余约 117 GB。

| 资源 | 路径（相对仓库根目录） | 状态 |
| --- | --- | --- |
| Conda 环境 `smolvla-rlt`（极简 Conda Py3.11.18 + `uv pip install`：`lerobot[smolvla]==0.4.4`，`torch==2.8.0+cu128`，`torchcodec==0.7.0`，`mani_skill==3.0.1`，`sapien==3.0.3`，`pytest`，`-e .`） | `~/miniforge3/envs/smolvla-rlt` | ✅ 已建（2026-11-25；env 增量 311 MB，共享 `~/.cache/uv` 硬链接） |
| `uv`（用户级包管理器） | `~/.local/bin/uv` | ✅ 0.13.10（全局缓存 `~/.cache/uv` 约 8.6 GB，跨项目/env 同分区硬链接共享） |
| Vulkan（ManiSkill / SAPIEN 渲染） | 系统（`/usr/share/vulkan/icd.d/nvidia_icd.json`） | ✅ SAPIEN 3.0.3 已识别 RTX 4090（`Supported: 1`；完整渲染/`physx_cuda` 待显存空闲时实测） |
| HF 缓存 | `.cache/huggingface`（启动器会设置 `HF_HOME`） | ✅ 已建，含 `SmolVLM2-500M-Video-Instruct`（约 2.0 GB） |
| SmolVLA 基座 | `models/lerobot/smolvla_base` | ✅ 已从 HF `lerobot/smolvla_base` 下载（873 MB，10 文件） |
| ManiSkill 原始演示（motionplanning / rl，各 1000 局，含 `env_states`） | `data/maniskill/demos/PegInsertionSide-v1/{motionplanning,rl}` | ✅ 2026-11-25 从 HF `haosulab/ManiSkill_Demonstrations` 下载（67 MB）；用于 3D 回放链路验证和重建数据集 |
| 3D 回放录制（`<key>.{glb,traj.json,json}`） | `outputs/viewer3d/` | ✅ 演示 `demo-mp-ep0000-seed2`、`demo-rl-ep0005-seed1201263687`（每个 GLB 约 12 MB） |
| LeRobot 三相机 512² 训练集（1000 局 / 149,055 帧，相机 `environment_camera`、`hand_camera`、`insertion_camera`） | `data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos` | ✅ 已从 HF `wkal/smolvla-rlt` 完整下载（3009 文件，约 2.3 GB） |
| SFT 输出 / last 权重 | `outputs/sft/peg_insertion/checkpoints/last/pretrained_model` | ❌ 需要重训 |
| Stage 1 输出根目录 | `outputs/rl_token/run_YYYYMMDD_HHMMSS/{rl_token.pt,rl_token_best.pt}` | ❌ 需要重训；Stage 2 加载 **best** |
| Stage 2 正式 / smoke 输出 | `outputs/online_rl/run_*/`、`outputs/online_rl_smoke/<tag>_*/` | ❌ |
| 评测 Markdown 记录 | `benchamrk/{sft,rlt,rl}/`（拼写沿用 `benchamrk`） | ✅ 在 git 里 |

**路径状态（2026-11-25 已迁移）**：`configs/*/*.yaml` 已改为相对 `REPO_ROOT` 的可移植路径（由各 `*Config.from_yaml` 解析），`scripts/*.{sh,py}` 已移除 `/root/miniconda3` 和失效旧代理 `18082`。`benchamrk/` 历史记录与 `paths.py` 中默认引用的旧 Stage 1 run 名（`run_20260901_145328`）仍为旧机器历史记录，重训 Stage 1 后再更新指向新的 best 权重。

## 3. 训练 / 评测运行规则

- 每个入口都有 `--check`（只用 CPU，不加载 VLA 和仿真器）。改动配置后先跑 `--check`，再跑 smoke，最后才跑正式训练。
- 正式训练每次新建 `run_YYYYMMDD_HHMMSS/`，不写进父目录，也不删除同级的其他 run。smoke 写进带 tag 的单独目录。
- 不要在同一张 GPU 上同时跑 SFT 和 Stage 1/2；启动前先确认其他用户的显存占用。
- `*.sh` 启动器会拒绝 Cursor/Codex 的 agent shell（在旧环境里，agent 进程大约 20 分钟后会被杀掉）。在 workspace-b 上，是否允许 agent 用 nohup 启动正式长训练，需要先问用户；这个检测逻辑也要随 Phase 0 一起评估。
- Stage 2 正式训练前：`--check` 打印的 per-dim 动作边界必须不是 ±1；`num_envs > 1` 时必须设置 `reconfigure_every_episodes >= num_envs`；`warmup_env_steps >= batch_size * chunk_len`。

## 4. 评测记录（强制）

每次评测（SFT 基线，以及之后所有 RLT、RL、消融、回归、对比评测）除了在 `outputs/` 下写原始的 JSON/JSONL/视频，还必须在 `benchamrk/<stage>/` 下写 Markdown 记录：

- 评测运行期间就创建并更新这份记录，最终状态明确标为 `complete`、`failed` 或 `interrupted`。
- 至少包含：UTC 时间、阶段/方法、模型及解析后的 checkpoint、任务/环境、成功定义、seed 集、局数、并行环境数、max steps、chunk 执行设置、相关配置、汇总指标（含 Wilson CI）、终止原因统计、逐局结果，以及原始产物的链接。
- 评测代码只有在实现了这份 Markdown 输出、并且有 CPU 测试覆盖时，才算完成。复用 `smolvla_rltoken.benchmark` 的格式。

## 5. 算法与设计文档

改动系统设计、算法、训练目标、Actor-Critic 行为或数据管线之前，先读对应的笔记：

- `docs/knowledge-base/SmolVLA_RLT_plan.md`：总体设计；`算法实现.md` 与 `详解/`：loss 和算法细节；`数据管线.md`：数据。
- `docs/rltoken-training/`：Stage 1 已适配的部分和延后的工作。
- `docs/online-rl/stage2_survey.md`：Stage 2 的锁定决策。**§13.1–13.5 覆盖该文件前面的内容**，要点：
  - 13.1：动作空间是 `MEAN_STD`，Actor 用从 checkpoint 推导的 per-dim 边界做 clip（绝不能是 ±1），warmup reference 不做 clip；warmup 步数要够一个 batch；训练循环必须报告 episode 成功率。
  - 13.2：允许 `num_envs > 1`；UTD 按 transition 计；局部 reset；`episode_id` 全局唯一；必须周期性 reconfigure（否则 peg 几何会被冻结）。
  - 13.3：残差 Actor `a = ã + Δ`，末层零初始化；BC 用 `sum` 归约作用在确定性 μ 上；`ref_dropout=0`，并且只遮蔽网络输入；`explore_std` 和 `action_std` 分开；保留未执行的动作尾巴；Actor 用均匀采样，只有 Critic 对成功样本上采样；BC 预训练加交接门槛。
  - 13.4：Actor 被 BC 钉死，稳定点 `μ − ref = (dQ/dμ) / (2β)`。不要凭理论去调 `bc_beta`，先读 `grad_ratio`、`q_adv_det`、`det_success_rate`；`q_gap` 衡量的是状态区分度，不是动作敏感度。
  - 13.5：Critic 看不到动作差异，现在输入改为 `(x, ã, (a − ã)/critic_residual_scale)`，`reference_chunk` 是必填参数。这会让 `dQ/da` 放大约 50 倍，所以 `bc_beta` 的标定失效了；正式 run 保留 `actor_drift_ceiling=1.6e-3`，smoke 时关闭。
- 新的 Stage 2 结论按 13.6、13.7…… 继续往下编号。
- `docs/knowledge-base/` 是项目背景，不是绝对真理。出现分歧时，以验证过的代码行为、测试、实验结果和用户的明确决定为准，并把分歧记进对应笔记。代码改动使知识库不准确时，在同一次改动里更新笔记，保持 `附件/` 和相对链接完整。

## 6. 代码约定

- YAML 是唯一真源，CLI 只有显式传入时才覆盖。新增配置旋钮时要写注释，说明它来自哪个实验或决策，并补上 `--check` 校验。
- `tests/` 只放 CPU 测试，不依赖 GPU、仿真器或数据集；GPU 路径用 Mock planner / Mock env 覆盖。提交前跑 `pytest`。
- 新的训练诊断指标要能从日志或 W&B 读出来，并在 survey 里说明怎么读、有什么判据。

## 7. 参考 RLT 仓库

以下仓库在旧机器上是 `/root/autodl-tmp/` 下的同级目录，**本机没有**；需要时 clone 到 `/home/dev/workspace/third_party/` 或仓库内被 gitignore 的 `third_party/`。只参考思路，不要照搬；和本仓库的知识库或验证过的行为冲突时，以本仓库为准。

- `RajatDandekar/RL-Token-SmolVLA`：SmolVLA / LeRobot 0.4.4 上的 RLT（SO-101），包括 `smolvla_rlt` policy 和 Stage 1/2 脚本。
- `yknxh/rlt-openpi`：OpenPI 上的 RLT，包括残差 Actor、twin Q、replay、VLA embedding hook。
- `afengleafs/smollvla_rltoken`：LeRobot 0.5.1 上的 SmolVLA RLT，包括 prefix embedding 抽取和 RL token 尺寸设计。

## 8. 实验记录站（rl-experiment-site）

本仓库以 `adapter: standard` 接入 `../rl-experiment-site`（协议见 `../rl-experiment-site/docs/integration-guide.md`，分支规范见 `docs/repo-workflow.md`）。站点读取展示分支上**已提交**的内容文件和 `docs/`，曲线和表格则从工作区的 `outputs/` 实时读取。

| 方向 | 展示分支 | 站点项目 id | 站点内容文件 | 实验报告目录 |
| --- | --- | --- | --- | --- |
| 主线 | `main` | `smolvla-rltoken` | `docs/site/progress.yaml` | `docs/`（笔记）、`benchamrk/`（评测记录） |

- `main` 是统一主线。新的独立算法 / 任务方向从 `main` 切分支，使用 `docs/site/<track>.yaml` 和 `docs/experiments/<track>/`，不修改也不删除其他方向的文件。验证完成后直接 `git merge` 回 `main`。
- **标准文件由代码自动写出**（`smolvla_rltoken.site_log`，x 轴字段统一为 `step`）：Stage 1 写 `metrics.jsonl` / `eval.jsonl`（val）/ `tables.json`；Stage 2 写 `metrics.jsonl`（x = env steps）/ `eval.jsonl`（每组 ref/det 探针一行）/ `tables.json`（按模式的成功率，带 Wilson CI）；`scripts/eval.py` 在评测目录写 `tables.json`。SFT 在 `lerobot-train` 里运行，训练曲线需要用 `python scripts/export_site_metrics.py` 从启动器日志转换（训练中可以反复运行）。新的评测脚本（如 `eval_rl.py`）复用 `site_log.eval_summary_tables`，并补 CPU 测试。
- 指标的中文名、单位和门槛线在 `docs/site/schemas/{sft,stage1,stage2}.json` 中定义，在 run 的 `schema:` 字段引用。新增训练诊断指标时，同时在对应 schema 里登记（`auto_discover` 已打开，不登记也会显示，只是没有中文名）。
- 每次**正式** run 开始后，在 `progress.yaml` 的 `runs` 里登记 `dirs: [outputs/<stage>/run_YYYYMMDD_HHMMSS]` 和 `schema`，并挂到对应阶段。smoke 不登记。阶段验证完成、写好 `benchamrk/` 记录或 `docs/` 笔记后，追加或补全 `stages`，同步更新 `project.status` / `highlights` / `roadmap`。叙述里的数字以记录为准，没有记录的结论不写。
- 提交前运行 `../rl-experiment-site/.venv/bin/python ../rl-experiment-site/tools/check_content.py smolvla-rltoken`，确认 PASS。
- 需要新视频时，写好 `videos` 条目（`src` 指向具体的 `.mp4`），在站点目录运行 `.venv/bin/python tools/build_media.py --project smolvla-rltoken --only <id>`。3D 回放：`python scripts/record_viewer3d.py` 写 `outputs/viewer3d/<key>.{glb,traj.json,json}`（演示回放不占显存，策略录制需要 GPU），站点直接读这个目录（`projects.yaml` 的 `recordings:`），录完刷新 `#/p/smolvla-rltoken/viewer` 即可回放，不用拷贝也不用重启站点；方案和当前状态见 `docs/site/viewer3d.md`。
