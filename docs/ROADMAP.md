# 重建与优化路线图（2026-11-25）

目标：在 workspace-b 上把项目重新跑起来，然后**让 RL 真正在冻结 VLA 基线之上带来提升**，并且能**定量说明 RL Token 起了多大作用**。

本文件是计划，不是锁定决策。真正落地的设计决策仍然记在 `docs/online-rl/stage2_survey.md` §13 之后（继续编号 13.6、13.7……）和 `docs/knowledge-base/`。

---

## 0. 现状判断

| 问题 | 证据 | 影响 |
| --- | --- | --- |
| 产物全部丢失 | 本机上没有 `data/`、`models/`、`outputs/`，也没有 Conda 环境 `smolvla-rlt` | SFT → Stage 1 → Stage 2 要全部重跑 |
| 路径绑死在旧机器上 | 3 个 YAML 和 `*.sh` 里写的是 `/root/autodl-tmp/...`、`/root/miniconda3/...`，代理也是旧代理 | 直接运行会失败 |
| GPU 是共享的 | 4090 24 GB，梳理时其他用户的进程已经占用约 21 GB | Stage 2 的 16 环境（约 7 GB+）可能跑不起来，要先确认空闲时段 |
| SFT 基线偏低 | 22%（35/50 步执行）；Stage 2 warmup 用 C=10 时只有 15% | RL 可修正的上限、成功样本的密度都受影响 |
| RL 还没有带来提升 | §13.3–13.5：先是交接崩溃，然后 BC 钉死 Actor，最后发现 Critic 看不到动作差异 | 核心问题，13.5 的修复还没验证 |
| RL Token 的作用没有测量 | 只有重建 loss，没有任何「z_rl 对控制有没有用」的指标 | 无法回答「RL Token 起了什么作用」 |
| RL 正式评测没有实现 | AGENTS.md 要求 `benchamrk/rl/` 评测记录，但 Stage 2 只有训练内的 probe | 无法和 SFT 基线做同口径对比 |

---

## Phase 0：环境与可复现性（先做，阻塞后面所有阶段）

1. **用户级 Conda 环境**：在 `~/miniforge3` 下建 `smolvla-rlt`，版本和旧环境一致（lerobot 0.4.4 / torch 2.8.0+cu128 / torchcodec 0.7.x / ManiSkill 3）。把实际安装的版本导出成 `environment.yml` 或 `requirements-lock.txt` 提交进仓库。
2. **Vulkan 自检**：运行 `vulkaninfo --summary`。如果缺 `libegl1` 或 `nvidia_icd.json`，这属于系统级改动，必须先征得用户授权，不能自己 sudo。
3. **路径可移植**：YAML 里的路径改成相对 `REPO_ROOT`（由 loader 负责解析），或者支持 `${REPO_ROOT}` 占位符；`*.sh` 的 `CONDA_SH` 改成 `$(conda info --base)`；代理改为可选的环境变量；「拒绝 agent shell」的逻辑改为基于 nohup + PID 的规则。补一个 CPU 测试，断言配置里不出现 `/root/autodl-tmp`。
4. **下载产物**：`lerobot/smolvla_base`，`wkal/smolvla-rlt` 数据集（只从官方 HF 下载，不用国内镜像）。
5. **`pytest` 全绿**，并且所有 `--check` 都能通过。
6. **清理仓库**：删除空的骨架文件（`models/*.py`、`rl/{actor_update,critic_update,replay_buffer,rollout,trainer}.py`、`scripts/{train_rl,rollout}.py`、`tests/test_replay_buffer.py`）。

## Phase 1：重建基线与统一评测

1. **SFT 重训**（20k step，配置不变）作为可比对的基线，然后跑 100 局评测（seeds 10000–10099），确认能回到约 22%。
2. **chunk 执行长度扫描**：在同一批 seeds 上，冻结 VLA 分别用 C ∈ {10, 20, 35, 50} 执行，每个设置跑 200 局。15% 和 22% 的差距目前在噪声范围内，需要搞清楚 RL 的 C=10 是不是在白白损失基线。这个结果也决定 Stage 2 的 `chunk_len`。
3. **实现 `scripts/eval_rl.py`**：加载 `online_rl.pt` 后，在固定 seeds 上同时评测「冻结 VLA / 确定性 Actor / 带噪 Actor」，按 AGENTS.md 的格式写 `benchamrk/rl/`，并补 CPU 测试。之后所有「是否有提升」的结论都以它为准，不再依赖训练中 n=16 的 probe。
4. Stage 1 重训：跑满 5000 步（带 val），选 best。

## Phase 2：让 RL Token 的作用可测量，再优化它

RLT 的前提是 z_rl 包含了「精细阶段需要的信息」（peg 和孔的相对位姿、接触状态），但到目前为止只验证了它能重建 VLM token。要注意：SFT 时 VLM 是冻结的（`train_expert_only`），所以 prefix hidden 基本就是 smolvla_base 的通用特征。

1. **线性探针（核心指标）**：在演示数据或 rollout 上采集 `(z_rl, 仿真特权状态)`，用 ridge 回归预测 peg 相对孔口的位置/姿态、peg 半长/半径、夹爪状态、是否接近成功。对照组：
   - 只用 proprio（9 维）
   - prefix token 平均池化（960 维）
   - 随机初始化的 encoder
   - 用 smolvla_base 抽取的特征和用 SFT 抽取的特征

   写进 `benchamrk/rlt/probe_*.md`。这就是「RL Token 的作用」的直接度量。
2. **Stage 2 状态消融**：`x ∈ {proprio, z_rl+proprio, pooled_prefix+proprio}`，在相同 seeds 和 env step 下比较 Critic 的 TD 残差、`q_adv_det` 和最终成功率。如果 z_rl 跟只用 proprio 打平，就说明 RL Token 没有起作用。
3. **RL Token 的候选改进**（根据探针结果挑，不要一次全上）：
   - 输入不只用 image token，还加上 state / language token（`--all-prefix-tokens` 已经支持）。
   - 抽取层：除了最后一层，也试中间层，或 action expert 的 KV / suffix 特征（这部分是 SFT 真正训练过的）。
   - 加入 insertion_camera 的高分辨率 token 比例，或在 `max_recon_tokens` 下按相机分配 token 预算，而不是 `linspace` 均匀下采样。
   - 辅助目标（偏离论文，需要用户拍板）：从 z_rl 预测 reference chunk、预测 n 步后的 proprio、预测 episode 是否成功。
   - 时序：拼接上一帧的 z_rl，或用 Δz 表示接触前后的变化。

## Phase 3：Stage 2 算法（按 §13.5 遗留的问题排序）

1. **验证 13.5 的残差 Critic**：跑一次正式 run，保留 `actor_drift_ceiling`。读 `grad_q_rms`、`q_adv_det` 与 TD 残差的比值，用天花板报出的 β 标定 `bc_beta`。
2. **P0-2 chunk 级相关探索噪声**：现在是对 80 个分量独立加 `randn_like`，在毫米级插入任务里只是抖动。改为每个 chunk 采样一个低维方向（比如每个关节一个常量偏置，或者沿时间维平滑的 OU 噪声）。
3. **P0-3 混合探索**：一部分 env 用较大的 `explore_std`，给 Critic 提供更宽的动作支撑；其余 env 保持小噪声，守住成功率。
4. **P1 配对 probe**：ref / det probe 用相同 seeds、相同几何，并增大每次的局数。在 `eval_rl.py` 落地之前，先靠这个读数。
5. **P2 有界残差**：Δ = δ·tanh(·)，从结构上限制幅度，配合 β 阶梯 / 自适应 β（让 `bc_dist_det` 跟踪一个目标值，类似 SAC 的温度自动调节）。
6. **视界标定**：γ=0.99 每步，成功平均出现在第 137 步左右；评估 chunk 级 γ^C 和更大 γ 的影响。
7. **（V2，需要用户改判 §1.1）关键阶段介入**：论文里 RL 只负责精细阶段。可以另开实验：用 VLA 跑接近段，在 peg 进入孔口邻域时（由特权状态或 z_rl 分类器判定）再交给 Actor；或者从「已对准」的初始状态开局训练。这是提升信噪比最大的一个杠杆，但要先拍板。

## Phase 4：提升 VLA 本身（与 Phase 2/3 并行，决定上限）

- SFT 更长训练 / lr schedule / 解冻部分 VLM 层（会改变 prefix 特征，Stage 1 需要随之重训）。
- 把 ManiSkill 的 RL 演示（`demos/.../rl`）也加入 SFT 数据，或做数据增强（相机扰动）。
- 每次 SFT 变动都按 Phase 1 的协议重新评测，基线编号写进 `benchamrk/sft/`。

---

## 建议的执行顺序

Phase 0 → Phase 1（1、3、4；2 可以和训练并行）→ Phase 2.1 探针 → Phase 3.1 → 根据读数选 Phase 2.3 / 3.2–3.5 → Phase 4。

每一步结束时都应该有一个 `benchamrk/` 记录和一句结论（有效 / 无效 / 证据不足），再决定下一步。
