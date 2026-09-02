# Stage 2 Online RL 调研与锁定决策

> 给下一 session 写代码用。先读本文件，再改 `src/smolvla_rltoken/`。
> 日期：2026-09-01。不打断正在跑的 Stage 1 RL Token 训练。
> 论文：Xu et al., *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*（Physical Intelligence，https://pi.website/research/rlt ，arXiv:2604.23073）。

---

## 0. 下一 session 怎么开始

1. **不要**从 agent shell 拉起长时间 GPU 训练（Stage 1 正式训、Stage 2 正式训同样规则）。CPU 测试、`--check`、短 `--smoke` / `--gpu-smoke` 可以。
2. 对照本文件第 1 节的七条锁定决策；与三仓或知识库原文冲突时，**以第 1 节为准**。
3. 按第 8 节 **M5 → M6 → M7 → M8** 推进：每步有 CPU 测试通过后再写下一块。不要一次写完整 trainer。
4. Stage 2 代码进 `src/smolvla_rltoken/`（包名已是这个，不是计划草稿里的 `smolvla_rlt`）。评测 Markdown 进 `benchamrk/rl/`（拼写保持现有 `benchamrk`）。
5. 改了算法语义时，同步改知识库对应笔记，并在本文件「与知识库差异」处留一句。

核心训练闭环（M5–M8）已落地；正式成功率评测入口仍后置。`--gpu-smoke` 用并行环境做短资源探测并写 `benchamrk/rl/gpu_smoke_*.md`。**先读 13.1 与 13.2 节**：13.1 是首轮 review 后修正的动作边界 / warmup 步数 / episode 指标三处，13.2 是并行环境解锁（覆盖第 1.6 条原来的「V1 单环境」）。本文件早期表述与这两节冲突时以这两节为准。已有：

| 已有 | 路径 |
| --- | --- |
| 冻结 prefix 抽取 | `src/smolvla_rltoken/vla/extractor.py`（Stage 1 默认 `use_cache=False`；Stage 2 `use_cache=True` + `sample_reference_chunk`） |
| per-dim 动作边界 | `src/smolvla_rltoken/vla/action_bounds.py`（从 checkpoint 的 `norm_map` + `action.min/max` 推导；MEAN_STD ≠ ±1） |
| RL Token encoder/decoder | `src/smolvla_rltoken/rlt/module.py`；Stage 2 只调用 `encode` / `rl_token`（`rlt/load.py`） |
| Actor / Critic / Replay | `src/smolvla_rltoken/rl/`（非残差 MLP、twin Q、无 stride） |
| 晚一拍 collector | `src/smolvla_rltoken/rollout/` |
| chunk env | `src/smolvla_rltoken/envs/chunk_env.py`（`r=float(success)`） |
| 三相机 ManiSkill | `PegInsertionSideThreeCamera-v1`，`src/smolvla_rltoken/envs/maniskill_env.py` |
| SFT 评测（success 真值） | `src/smolvla_rltoken/vla/evaluation.py`，`scripts/eval_sft.sh` |
| Stage 1 权重（无 val 的 5000 步） | `outputs/rl_token/run_20260901_204708/rl_token.pt` |
| Stage 1 权重（带 val，**Stage 2 默认**） | `outputs/rl_token/run_20260901_145328/rl_token_best.pt` |
| SFT last | `outputs/sft/peg_insertion/checkpoints/last/pretrained_model` |
| Stage 2 入口 | `scripts/train_online_rl.py` / `scripts/train_online_rl.sh`；YAML `configs/rl/actor_critic.yaml` |

conda：`smolvla-rlt`。大文件只放 `/root/autodl-tmp`。下载走 `HTTP_PROXY=http://127.0.0.1:18082`。

---

## 1. 用户锁定的七条决策（2026-09-01）

调研后列出的待确认项，用户逐条拍板如下。实现不得静默改这些默认值。

### 1.1 何时把控制权给 Actor（原问题 1）

**锁定：B — 从标准 `env.reset()` 开局，对齐现有 SFT eval；warmup 之后全程 Actor。**

- 每个 episode 从 PegInsertion 自然起点开始（peg/box 随机在桌面），**不** reset 到「已经对准孔口」。
- warmup：执行冻结 SmolVLA 的 reference chunk。
- warmup 结束后：每个 chunk 边界都跑 Actor（仍以 VLA 的 \(\tilde a\) 为条件）。
- 容易段不靠门控切走 VLA，靠 \(\beta\|a-\tilde a\|^2\) 把 Actor 钉在 VLA 附近。
- **不做**论文真机那套「人手决定何时 handover 到 critical phase」。
- **不做**学出来的「精细操作才启用 Actor」开关。

论文里「VLA 做简单、Actor 做精细」是 **episode 协议 / 人手切换**，不是 Actor 内部 gate。本仓库仿真 PegInsertion 整段约 200 control step，选 B 是为了和 SFT eval 同一任务分布。以后若要做论文 A（只练插入段），另开实验，不改 B 的默认代码路径。

### 1.2 Actor 参数化（原问题 2）

**锁定：论文 / 知识库 / afengleafs — 直接 \(\pi_\theta(x,\tilde a)\)，不是残差。**

\[
a_{1:C} = \pi_\theta(x,\tilde a_{1:C})
= \mathcal N(\mu_\theta(x,\tilde a_{1:C}), \sigma^2 I)
\]

- **禁止** rlt-openpi 的 \(a=\tilde a+\Delta\) 与末层零初始化（那是工程偏离，不是论文公式）。
- 靠近 VLA 完全由 BC 项 \(\beta\|a-\tilde a\|^2\) 完成。
- warmup 刚结束 Actor 仍是随机网络：用 **warmup 数据上的离线更新** 再让 Actor 控环境（见 1.6 / M7），不要改成残差来「保平安」。

### 1.3 Stride（原问题 3）

**锁定：V1 不使用 stride=2。**

- 一次环境执行的长度为 \(C\) 的 chunk = **一条** replay transition。
- 论文 Sec. V「Subsampling Action Chunks」和 afengleafs 的跨 chunk 窗口组装 **明确未做**。代码、config、文档都要写 `stride` 未启用，避免后人当成已实现。
- 以后做 stride 是 V2；需要 \(H>C\) 的移位 reference，以及「等下一个 chunk 执行完再 emit \(o>0\) 窗口」。

### 1.4 稀疏奖励（原问题 4）— 能映射，且必须映射

**锁定：用本仓库评测同一套 `info["success"]` 做成论文式稀疏奖励；不用 dense `env.reward` 当 RL 目标。**

结论：**能映射。** SFT 评测已经把 ManiSkill `info["success"]` 当作唯一成功真值；RL 把它变成逐步 \(r_t\in\{0,1\}\)，语义与论文「成功 +1、否则 0」一致。

#### 几何定义（ManiSkill `PegInsertionSide-v1.has_peg_inserted`）

peg 红端（head）在孔坐标系下：

- \(x \ge -0.015\)（沿孔轴向插入过阈值）
- \(y,z\) 落在孔半径内

`evaluate()` 返回 `success` 与 `peg_head_pos_at_hole`。任务说明写的是 white end 插入过半；实现实际用的是 **head 位置**。RL 与 SFT eval 必须继续用这个官方 `success`，不要另写几何判据。

#### 现有评测怎么用它

`src/smolvla_rltoken/vla/evaluation.py`：

- 每步读 `info["success"]`；True 立即停、记成功。
- 200-step 上限记失败（`time_limit` / `max_steps`）。
- **成功不看 `reward`。** `scripts/eval_sft.sh` 的环境却是 `reward_mode="dense"`，所以 `EpisodeResult.reward_sum` 是 shaping（成功局常见几百），**不能**当 RL 回报。
- 已完成 SFT eval（`benchamrk/sft/run_20260901_121513.md`）：held-out seed 10000 起 100 局，成功率 22%，成功步数均值约 137。这是 Stage 2 的 VLA warmup 能力下限参考。

#### Gym / ManiSkill 信号怎么接到 TD

PegInsertion 的 `evaluate()` **没有** `fail` 键。因此：

| 信号 | 来源 | RL 用法 |
| --- | --- | --- |
| `info["success"]` | `has_peg_inserted` | \(r_t=1\) 当且仅当本步 success；否则 \(0\) |
| `terminated` | ManiSkill：`terminated = success`（无 fail） | success 时 bootstrap **清零** |
| `truncated` | `max_episode_steps=200` 的 time limit（BaseEnv.step 里 truncated 恒 False，由 Gym wrapper 置位） | 超时 **不清零** bootstrap（`done=0` 入库） |
| `env.reward`（当前 eval 的 dense） | reaching / grasp / pre-insert / insert shaping，成功时改写成 10 | **正式 RL 忽略** |
| `reward_mode="sparse"` | 无 fail 时 `reward = success` 的 0/1 | 与手工映射等价；仍建议代码里显式 `float(success)`，避免以后改 reward_mode 踩坑 |

逐步奖励（写入 replay 的 `reward_sequence`）：

```text
r_t = 1.0 if info["success"] else 0.0
```

chunk 折扣回报：

\[
R^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i}
\]

成功发生在 chunk 第 \(k\) 步则 \(n=k\)，用 \(\gamma^n\) 而不是固定 \(\gamma^C\)。成功步之后不再执行剩余 chunk 步。

**禁止：**

- 把 dense reward 写进 Critic TD（那会变成 shaping RL，不是论文设定）。
- 失败给 \(-1\)（ManiSkill 在「有 fail 键」时才会这样；PegInsertion 没有。论文也是失败 0）。
- 另写「插入深度 / 距离」当成功。

知识库 M9「先 dense debug 再切 sparse」：只允许 **显式 debug 开关**，默认关闭。正式数字、W&B 主曲线、`benchamrk/rl/` 一律 sparse `info["success"]`。

### 1.5 干预、dropout、\(\beta\)（原问题 5）

| 项 | 锁定 |
| --- | --- |
| 人类干预 | **不要。** 无 VR、无 leader、无 `get_intervention` 覆盖动作。接口以后可留 `None`，V1 不接。 |
| Reference dropout | **0.5**（论文 App. B）。只 mask Actor **输入** 的 \(\tilde a\)；BC 目标仍用未 mask 的 \(\tilde a\)。推理 / rollout **不** dropout。 |
| \(\beta\) | **先 1.0**（afengleafs 默认）。与 dropout 一样做成 YAML，不要写死在模块里。 |

### 1.6 \(C\)、执行长度、并行、UTD（原问题 6）

按调研建议锁定：

| 项 | 值 | 说明 |
| --- | --- | --- |
| VLA horizon \(H\) | 50 | SmolVLA `chunk_size` |
| RL chunk \(C\) | 10 | 论文 App. B |
| 每 chunk 执行步数 | **10**，然后重新推理 | **不要**沿用 SFT eval 的 35/50（`--chunk-execution-ratio=0.70`） |
| 训练并行环境 | **YAML 当前 `num_envs=16`**（2026-09-02 用户改判原「V1 单环境」，见 13.2） | 原锁定「V1 单环境，向量化往后放」。现在 `num_envs>1` 可用于正式训练：UTD 按 transition 计账、结束的 env 用 ManiSkill 局部 reset 立即重开、`episode_id` 全局唯一。**并行时必须设 `reconfigure_every_episodes>0`**，否则 peg 几何全程冻结（见 13.2） |
| 评测并行 | 可复用 SFT 的 `num_envs=8` 思路，但 Stage 2 eval 另写，仍要 `benchamrk/rl/` + CPU 测试 |
| max episode steps | 200 | 与 SFT eval 相同 |
| 控制 | `pd_joint_pos` | action 8 维，state / proprio 9 维 |
| UTD | 5 | 每条 **transition** 5 次梯度。实现为 `utd * result.added`：单环境下一个执行 chunk 就是一条 transition，`num_envs=N` 时一次 collect 产出 N 条，梯度数随之放大，否则梯度/数据比会被静默除以 N |
| Critic : Actor | 2 : 1 | 先 Critic 后 Actor |
| \(\gamma\) | 0.99（建议，YAML 可改） | 逐步折扣；chunk backup 用 \(\gamma^n\) |
| \(\tau\) | 0.005 | Polyak |
| Actor \(\sigma\) | 0.05 | 固定探索标准差；部署取 \(\mu\) |
| batch | 256 | 与论文 / afengleafs 同量级；buffer 不够时跳过 update |
| warmup | **32000 env step**（YAML；须 ≥ `batch_size * chunk_len`，并行时最好是 `num_envs * max_episode_steps` 的整数倍） | 纯 VLA 填 buffer。V1 无 stride ⇒ 一个执行 chunk = 一条 transition。硬约束：`warmup_env_steps >= batch_size * chunk_len`，`--check` 会拒绝 |
| warmup 后离线更新 | **要做** | 串行实现下否则第一个 Actor chunk 来自随机网络。`run_offline_updates` 返回真实执行次数，0 次时训练循环打 WARNING，`did_offline` 不再假报 True |
| 动作空间 | SmolVLA **归一化** 空间里做 Actor/Critic；下环境前走 SFT 同一套 postprocessor 反归一化 | 8 维有效，pad 到 32 是 VLA 内部的事 |
| 动作边界 | **per-dim，从 checkpoint 统计推导**，`action_bound_margin=1.5` | 本 checkpoint 的 `ACTION` 归一化是 **MEAN_STD**（z-score），**不是** ±1 有界空间：演示范围约 ±2~5，固定 ±1 clip 会砍掉约 70% 关节行程（149,055 帧里 86% 至少有一维被截断）。见 `src/smolvla_rltoken/vla/action_bounds.py`；`--check` 打印解析出的边界 |
| clip 作用范围 | **只 clip Actor 输出** | warmup 逐位执行冻结 VLA 的 reference（与 SFT eval 同一条链，22% baseline 才可比）。reference 越界只做诊断计数，不改动作 |

### 1.7 开发顺序（原问题 7）

**锁定：按知识库 M5 → M6 → M7 → M8，一步验收再下一步。**

见第 8 节。禁止先写 `train_online_rl.py` 再补测试。

---

## 2. 核心澄清：论文「精细时 Actor 才发挥作用」是什么

两层不要混：

### 2.1 Algorithm 1：每个 chunk 的动作来源

```text
ã ← 冻结 VLA(obs, language)
z_rl ← 冻结 encoder(VLA tokens)
x ← concat(z_rl, proprio)
a ←
    人类干预     若有（本仓库 V1：无）
    ã            若仍在 warmup
    π_θ(x, ã)    否则
执行 C 步
```

Actor 上场后 **每个 chunk 都出动作**。没有「检测接触/精细再打开 Actor」的网络。

### 2.2 Targeted improvement of critical phases：实验协议

论文真机：先让 base VLA 跑容易段；人选择何时把控制交给 RL；RL 只在该段存数据、收稀疏成功/失败。评测还可再微调 VLA 预测 handover。博客例子：拿起螺丝刀是 VLA，对准螺丝是 RL。

这是 **数据收集范围**，不是 Actor 公式。三个参考仓的代码里都没有自动精细门控：

- afengleafs：真机文档用 `reset()` 直接摆到预插入；部署才 VLA→RLT 切换。
- rlt-openpi：warmup 后 Actor 全程；VR 是人类覆盖。
- Rajat：warmup 后 Actor 全程；拖 leader 是人类覆盖。

本仓库锁定 B：仿真整局都是 RL episode（warmup 后 Actor 全程），用 \(\beta\) 约束，而不是切阶段。

进入 RL 段后 VLA **仍每 chunk 前向**：给 \(z_{rl}\) 和 \(\tilde a\)。Actor 是 reference-conditioned refinement，不是从零控臂。

---

## 3. 论文公式（Stage 2 必须对齐）

冻结 \(\theta_{\mathrm{vla}}\) 与 encoder \(\phi\)。状态 \(x=(z_{\mathrm{rl}}, s^p)\)。

Critic（twin Q，target 取 min）：

\[
\mathcal L_Q = \mathbb E_{\mathcal B}\Big[\big(\hat Q - Q_\psi(x, a^{\mathrm{exec}}_{1:C})\big)^2\Big]
\]

\[
\hat Q = \sum_{t'=1}^{n}\gamma^{t'-1} r_{t'} + (1-d)\,\gamma^{n}\, \mathbb E_{a'\sim\pi_\theta}\big[Q_{\psi'}(x', a')\big]
\]

\(a'=\pi_\theta(x',\tilde a')\)。当前项必须用 replay 里的 **executed** 动作。

Actor：

\[
\mathcal L_\pi = \mathbb E\big[-Q_\psi(x, a^{\mathrm{new}}) + \beta\|a^{\mathrm{new}}-\tilde a\|_2^2\big]
\]

\(a^{\mathrm{new}}\sim\pi_\theta(\cdot\mid x,\tilde a)\)（训练可对输入 dropout \(\tilde a\)）。Actor 更新时冻结 Critic **参数**，但保留 \(\partial Q/\partial a\)。

三句话：

1. Critic 当前项：旧真实 \(a^{\mathrm{exec}}\)
2. TD 下一动作：当前 Actor 在 \((x',\tilde a')\) 上新采样
3. Actor loss：当前 Actor 在 \((x,\tilde a)\) 上新动作

---

## 4. Rollout 细节（V1）

每个 chunk 边界：

1. `obs` → preprocessor（与 SFT 同一 `rename_map`：env/hand/insertion → camera1/2/3）。
2. 冻结 SmolVLA：
   - prefix hidden → encoder → \(z_{rl}\in\mathbb R^{512}\)
   - 采样 reference chunk \(\tilde a_{1:H}\)，\(H=50\)，**只取前 \(C=10\)** 给 Actor。
   - Stage 2 应把 extractor 的 `use_cache=True`，用 prefix KV 采 reference（Stage 1 是 `use_cache=False`）。不要每个 chunk 无 cache 地重跑 expert。
3. \(x=\mathrm{cat}(z_{rl}, s^p)\)，\(s^p\) 为 **9 维** `observation.state`（Panda qpos），不要误用 8 维 action 当 proprio（rlt-openpi 那样切 `action_dim` 对本任务是错的）。
4. 选动作：warmup → \(\tilde a_{1:C}\)（**原样执行，不 clip**）；否则 Actor（训练加 \(\sigma\) 噪声，eval 取 \(\mu\)）。Actor 输出在归一化空间，按 **per-dim** 边界 clip 后再 unnormalize。边界由 checkpoint 的 `action.min/max` + `norm_map` 推导（本 checkpoint 是 MEAN_STD，范围约 ±2~5），**不要**写死 ±1。
5. 环境逐步 `step`，最多 \(C\) 步。每步记录 `r_t=float(success)`、terminated/truncated。成功或 timeout 提前结束 chunk，记下 `n_steps`。
6. **晚一拍写入**：到 `obs'` 后再跑一次 VLA+encoder，得到 \(x'\) 和 \(\tilde a'\)，补全 transition 再 `replay.add`。最后一步成功/截断时：`terminated` 则 bootstrap 目标为 0，`next_*` 可占位；`truncated` 仍要尽量算 \(x'\)（或按知识库 truncated 不清零，需要真实 next 状态——timeout 时 next 就是最后一帧 obs，仍应 encode）。

不要学 rlt-openpi：用当前 \(\tilde a\) 冒充 \(\tilde a'\)。

语言指令与 SFT eval 相同：`Insert the peg into the hole from the side.`

---

## 5. Replay 字段（V1，无 stride）

建议 dataclass（名字可改，语义不能少）：

| 字段 | shape / 类型 | 用途 |
| --- | --- | --- |
| `z_rl` / `proprio` 或拼好的 `x` | `[512]` + `[9]` | Critic / Actor 当前状态 |
| `reference_action` | `[C, 8]` 归一化 | Actor 条件与 BC 目标 |
| `executed_action` | `[C, 8]` 归一化 | **仅** Critic 当前 Q |
| `reward_sequence` | `[C]`，未执行步填 0 | 算 \(R^{(n)}\) |
| `n_steps` | int | \(\gamma^n\) |
| `next_z_rl` / `next_proprio` 或 `x_next` | 同 `x` | TD bootstrap 状态 |
| `next_reference_action` | `[C, 8]` | 当前 Actor 在下一状态出 \(a'\) |
| `terminated` | 0/1 | success → 1，bootstrap 清零 |
| `truncated` | 0/1 | time limit → 1，**不要**与 terminated 混成一个 `done` |
| `episode_id`, `chunk_id` | int | 调试：成功前最后 chunk 的 Q |

`sample` 默认均匀随机。正式训分层上采样：`success_sample_frac=0.25` 抽成功局（`terminated=1` 之后该 `episode_id` 仍在 buffer 里的全部 chunk），`reward_sample_frac=0.05` 抽 `reward_sequence` 有正值的 chunk，其余来自非成功槽。还没有成功样本时退回均匀。V1 **没有 PER**。capacity YAML（如 1e5–2e5）。checkpoint 可另存 buffer。

V1 **没有** `offset` / 跨 chunk 拼接 / `ref_full[:, o:o+C]`。注释写明 stride 未做。

---

## 6. Actor / Critic 结构（V1）

均为 2 层 MLP，hidden 256，**LayerNorm+ReLU**（已锁定；测试断言该结构）。无 target smoothing。

**Actor**

- 输入：`cat(x, flatten(ã_in))`，训练时 `ã_in` 以 0.5 概率整段置零。
- 输出：`[B, C, 8]` 的 \(\mu\)；采样 \(\mu+\sigma\epsilon\)。
- **直接输出最终动作**，不加 \(\tilde a\)。
- 部署：`eval()`，无噪声、无 dropout。

**Critic**

- Twin Q：`cat(x, flatten(a)) → scalar`。
- Target 网络 deepcopy + `requires_grad=False`，Polyak \(\tau=0.005\)。
- Target：`min(Q1', Q2')`。可选 TD3 target smoothing（rlt-openpi 有；论文写 TD3 风格 twin Q，smoothing 不是必须）。V1 建议：**先不加** target smoothing，少一个超参；若 Q 过估计再加。

**更新**

```text
batch = replay.sample()
# critic：Q(x, a_exec) vs stopgrad(R + (1-term)*gamma^n * Q'(x', actor(x', ã')))
# 每 2 次 critic：actor_loss = -min(Q(x, a_new)) + beta * MSE(a_new, ã)
# 禁止 no_grad 包住 actor 用的 Q
# polyak
```

Actor 用的 reference 可 dropout；BC 的 \(\tilde a\) 不 dropout。

---

## 7. 三仓对照（写代码时跟谁、不跟谁）

| 点 | 论文 | 本仓库 V1（已锁定） | afengleafs `smollvla_rltoken` | rlt-openpi | Rajat `RL-Token-SmolVLA` |
| --- | --- | --- | --- | --- | --- |
| Hidden | VLA final embeddings | **已有 Stage 1**：prefix 最终层+RMSNorm，image-only，`D=960`，`M=192` | 同左 | 全 prefix，2048 | VLM+expert，576，不要用 |
| Actor | \(\pi(x,\tilde a)\) | **同论文** | 同论文 | **残差，不要用** | 直接 MLP |
| 残差冷启动 | 无 | **离线 warmup 更新** | 文档建议离线更新 | 末层零初始化 | 无 |
| Stride=2 | 有 | **不做** | 有（跨 chunk） | 有 API，主循环弱 | 无 |
| `next_ã` | 要 | **晚一拍必存** | 存 `ref_next` | **用当前 ã 近似，禁止抄** | 有 next_ref |
| 人类干预 | 可选 | **不做** | 接口 | VR | leader |
| dropout / \(\beta\) | 50% / 任务相关 | **0.5 / 1.0** | 0.5 / 1.0 | 0.5 / 0.5 | 0.5 / 0.1 |
| 奖励 | 人标稀疏 +1 | **`info["success"]` → 0/1** | mock / 真机 | 键盘 | 干预当成功 |
| 训练循环 | 异步 rollout/learner | **同步单环境** | 串行 | 串行 | 真机串行 |
| 阶段切换 | 人手 / 后训 VLA | **整局 B** | 真机 reset 到插入段 | 无 | 无 |

实现时优先读：

- 公式与梯度：本仓库 `docs/knowledge-base/详解/`
- buffer 晚一拍、`ref_next`：afengleafs `rlt/replay_buffer.py` 的 **无 stride 子集**（`o=0` 那条），不要把 `_emit_pair` 整段搬过来。
- 不要抄 rlt-openpi `online_rl_trainer.py` 里 `next_a_tilde=a_tilde`。
- Rajat 的 intervention detector、expert suffix、256-d token：**不用于本仓库。**

---

## 8. 编码切片（M5–M8）与验收

每步：**新模块 + CPU 测试**。测试不加载真 VLA、不启仿真 GPU。假 tensor 即可。通过后再做下一步。

建议目录（可微调，但模块要分开）：

```text
src/smolvla_rltoken/rl/
  actor.py
  critic.py
  agent.py          # 更新顺序、dropout、polyak
  replay.py
src/smolvla_rltoken/rollout/
  transition.py
  collector.py      # 晚一拍；warmup 开关
src/smolvla_rltoken/envs/
  chunk_env.py      # 逐步 step，映射 sparse r / terminated / truncated
configs/rl/actor_critic.yaml
scripts/train_online_rl.py   # 薄 CLI，YAML 真源；正式训拒绝 agent
tests/test_rl_actor.py
tests/test_rl_critic.py
tests/test_rl_replay.py
tests/test_rl_transition.py
tests/test_rl_agent_update.py
tests/test_chunk_env_reward.py
```

正式输出父目录：`outputs/online_rl/`，每次 `run_YYYYMMDD_HHMMSS/`（与 Stage 1 相同约定）。目录未创建前不要写进 `AGENTS.md` 资源表；创建并跑通后再登记。

### M5 — Replay + 纯 VLA 灌数据（先关 Actor）

**做：**

- Transition dataclass 与 ReplayBuffer（add / sample / capacity）。
- Chunk env wrapper：`step` 执行单步 8 维反归一化动作；`r=float(success)`；成功 `terminated`；200 步 `truncated`。
- Collector：`use_actor=False` 时动作= VLA 前 \(C\) 步；晚一拍补 `x'`,`ã'`。
- 可用 mock VLA（固定 `z_rl`、固定 `ã`）+ mock env。

**验收（CPU）：**

- sample 形状正确。
- 当前 Q 用的 action 是 `executed_action`，不是重新 `actor(state)`（本步还没有 Actor 更新，可先测字段）。
- 成功步 `n_steps < C` 时 reward 只在该步为 1，其后为 0。
- truncated 与 terminated 分字段。
- **断言没有 stride 字段 / 一次 add 只增加 1。**

**不要：** Actor 优化、真 GPU rollout（M5 的 mock 即可）。真环境灌 warmup 放到 M8 前的 GPU smoke。

### M6 — 只训 Critic

**做：** Twin Q、TD target、polyak。next action 来自 **当前 Actor 前向**（此时 Actor 可随机或 BC 初始化，但 **不** `actor_optimizer.step`）。

**验收：**

- `Q(x, a_exec)` 对 `a_exec` 有梯度；对 Actor 参数无更新。
- target 用 `a' = actor(x', ã')`，`ã'` 来自 batch 的 `next_reference`，不是 `reference`。
- `terminated=1` 时 bootstrap 为 0。
- `n_steps=6` 时折扣是 \(\gamma^6\) 不是 \(\gamma^{10}\)。
- 合成「最后 chunk reward=1、前序 0」时，多步 TD 后前序 Q 上升（可用假转移，不必上仿真）。

### M7 — Actor 更新（仍可用 replay 里已有转移，不必真控环境）

**做：** Actor MLP、dropout 0.5、\(\beta=1\)、critic 参数冻结但 Q 对 action 可导。warmup buffer 上先跑若干 update（离线 warm-start）。

**验收：**

- Actor 参数变、Critic 参数不变。
- `torch.no_grad()` 包 Q 时应 **测出 Actor 无梯度**（反例测试）。
- dropout 只改输入；`MSE(a, ã)` 的 `ã` 未被置零。
- 输出是绝对动作，`μ` 与 `ã` 没有强制 `μ=ã+Δ` 的加法结构。

### M8 — Actor 真正 rollout + 完整循环

**做：**

- `use_actor=True` 的 collector。
- 循环：`if steps < warmup: VLA else: actor` → 写入 buffer → 若已过 warmup 且 buffer≥batch：先可选一次性离线更新 → 之后每 chunk `UTD` 次 `agent.update`。
- YAML + `--check`（不加载 VLA）+ `--smoke`（极少 step，可 mock 或 1 个真 env）。
- 启动器：正式训拒绝 Cursor/Codex agent（抄 `train_rltoken.sh` / `train_sft.sh` 的检查）。
- 评测入口后续再做：须 `benchamrk/rl/<run>.md` + CPU 测试；执行 \(C=10\) 再规划，success 仍是 `info["success"]`。

**验收：**

- warmup 期间 executed == reference（数值上**逐位相等，不允许经过任何 clip**；提前成功时只比已执行的前 `n_steps` 步，因为未执行的尾部会被补零。由 `CollectResult.executed_matches_reference` 给出）。
- 过 warmup 后 executed 来自 Actor。
- 无人类干预分支被调用。
- smoke 不写正式 `outputs/online_rl/` 父目录覆盖；`outputs/online_rl_smoke/` 下每次新建 `<tag>_YYYYMMDD_HHMMSS/`。
- 离线冷启动真的跑了（`OnlineRLTrainResult.offline_updates > 0`），不是只置了标志位。
- episode 级指标可见：warmup 期（纯 VLA）与 Actor 期成功率分开统计，逐 episode 记录写进 run 目录的 `episodes.jsonl`。

**M8 之后（本文件不要求一次做完）：** GPU 短 rollout（`--gpu-smoke`，并行 env 资源探测，非正式评测）、Q 是否向成功 chunk 回传、success rate vs SFT 22% baseline、完整 `benchamrk/rl/` 评测记录。

---

## 9. 本仓库 PegInsertion 工程约束（写 env/collector 时对照）

| 项 | 值 |
| --- | --- |
| 环境 id | `PegInsertionSideThreeCamera-v1` |
| 相机 | `environment_camera`, `hand_camera`, `insertion_camera`，512×512 |
| rename | `SFT_IMAGE_RENAME_MAP`（`src/smolvla_rltoken/paths.py`） |
| `control_mode` | `pd_joint_pos` |
| action | 8 维；SmolVLA `max_action_dim=32` 内部 pad |
| state / proprio | 9 维 qpos |
| 任务文本 | `Insert the peg into the hole from the side.` |
| `sim_backend` | 正式 `physx_cuda` |
| `max_episode_steps` | 200（任务默认 100，本仓库 eval 已改 200） |
| SFT eval 执行 | 预测 50、执行 35 — **仅 baseline 评测**；RL 执行 10 |
| 成功 | `info["success"]`，见 1.4 |
| Stage 1 encoder | `d_model=512`，image-only，2 层 8 头；checkpoint 只取 encoder 权重 |
| Stage 2 开 cache | extractor 对 reference 采样 `use_cache=True`；prefix-only 仍需 `fill_kv_cache=True` |

归一化：与 checkpoint 自带 preprocessor / postprocessor 一致，和 `PolicyRunner`（`evaluation.py`）同一条链。Actor 看到的 action 必须是这条链上的归一化值。

---

## 10. 明确不做（V1）

- 残差 Actor、末层零初始化当默认。
- stride=2 子采样。
- 人类干预 / 干预 BC 替换 reference。
- 学 handover 或启发式切「插入段」。
- 用 dense ManiSkill reward 当正式 TD 目标。
- 训练时更新 VLA 或 RL Token encoder（含 `alpha L_vla`）。
- 加载 Stage 1 decoder。
- 磁盘 embedding cache。
- ~~多环境并行 collect（V1）~~ —— 已解锁，见 13.2。仍**不做**异步 actor worker。
- 异步 actor worker。
- 从 Cursor/Codex agent 启动正式 GPU 长训。
- 把评测结果只写在 W&B / JSON，不写 `benchamrk/rl/`。

---

## 11. 建议超参（YAML 真源，CLI 显式才覆盖）

```yaml
# configs/rl/actor_critic.yaml 草稿，实现时以此为起点
vla_checkpoint: /root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/last/pretrained_model
rl_token_checkpoint: /root/autodl-tmp/smolvla-rltoken/outputs/rl_token/run_20260901_145328/rl_token_best.pt
output_dir: /root/autodl-tmp/smolvla-rltoken/outputs/online_rl

chunk_len: 10            # C
vla_horizon: 50          # H；只执行前 C
action_dim: 8
proprio_dim: 9
rl_token_dim: 512
max_episode_steps: 200
reward: sparse_success   # info["success"] → {0,1}；dense 仅 debug 开关

hidden_dim: 256
n_layers: 2
action_std: 0.05
action_bound_margin: 1.5 # per-dim clip = 演示归一化范围 x 该系数；不要用固定 ±1
ref_dropout: 0.5
bc_beta: 1.0             # BC 项是 mean 归约，等于论文 ‖·‖² 的 1/80
gamma: 0.99
tau: 0.005
actor_lr: 3.0e-4
critic_lr: 3.0e-4
utd: 5
critic_updates_per_actor: 2
batch_size: 256
success_sample_frac: 0.25   # 成功轨迹分层；无 PER
reward_sample_frac: 0.05    # 正奖励 chunk；与上一行之和须 <= 1
buffer_capacity: 200000
warmup_env_steps: 32000             # >= batch_size * chunk_len；16 环境时取 16*200 的倍数
offline_updates_after_warmup: 1000  # 串行冷启动
total_env_steps: 1000000
stride: 1                # V1 锁定；不要改成 2 除非用户再拍板
use_residual_actor: false
human_intervention: false
num_envs: 16             # 并行已支持（13.2）；>1 时必须一起设 reconfigure_every_episodes
reconfigure_every_episodes: 16  # 并行时必填且 >= num_envs，否则 peg 几何全程冻结

task: "Insert the peg into the hole from the side."
device: cuda
```

若之后有更好的 `val_loss_ro` run，把 `rl_token_checkpoint` 换成那个 `rl_token_best.pt`，仍只加载 encoder。当前默认是 `run_20260901_145328/rl_token_best.pt`。

---

## 12. 参考文件（只读对照，不要 fork）

本仓库：

- `docs/knowledge-base/SmolVLA_RLT_plan.md` Part 3–6
- `docs/knowledge-base/详解/RLT_actor_critic协同工作.md`
- `docs/knowledge-base/详解/RLT_Actor.md`
- `docs/knowledge-base/详解/RLT_Critic.md`（已按无 stride 写）
- `src/smolvla_rltoken/vla/evaluation.py`（success 真值、200 step、相机、processor）
- `src/smolvla_rltoken/vla/extractor.py`（Stage 2 要扩展 cache）
- `src/smolvla_rltoken/rlt/module.py`（`encode` / `rl_token`）

外部（`/root/autodl-tmp/`）：

- `smollvla_rltoken/rlt/actor_critic.py`、`replay_buffer.py`、`rlt_policy.py`、`train_online.py`、`real_train.md`
- `rlt-openpi/src/rlt_openpi/models/actor.py`（**反例：残差**）、`training/online_rl_trainer.py`（**反例：next_ã 近似**）
- 论文 PDF：https://www.pi.website/download/rlt.pdf
- 博客：https://pi.website/research/rlt

ManiSkill：`mani_skill/envs/tasks/tabletop/peg_insertion_side.py` 的 `has_peg_inserted` / `evaluate` / `compute_dense_reward`。

---

## 13. 与知识库的关系

本文件 **覆盖** 计划里未锁死的工程选项。已锁定且与草稿一致的：非残差 Actor（**已被 13.3 推翻，改为残差**）、V1 无 stride、chunk \(C=10\)、晚一拍、twin Q、先 Critic 后 Actor、模块分离、M5–M8。

相对知识库草稿的明确选择：

| 草稿 | 本文件锁定 |
| --- | --- |
| reference dropout 默认 0，再 ablation | ~~**默认 0.5**~~ → 13.3 改回 **0** |
| \(\gamma\) 未写死 | **0.99** |
| 可 dense debug | 默认 **sparse success**；dense 仅开关 |
| Actor 随机初始化直接控环境有风险 | ~~**warmup 后离线更新**，仍非残差~~ → 13.3 改为 **残差 + BC 预训练 + 交接门槛** |
| 评测 chunk 比例未区分 | SFT 仍 35/50；**RL 执行 C=10** |
| Actor MLP 草稿纯 ReLU | **LayerNorm+ReLU**（与 afengleafs 一致，测试锁定） |

实现完成后把上述默认值写进 `docs/knowledge-base/算法实现.md` 的 Stage 2 工程约定（本调研提交时已加入口，细节以本文件为准直到代码落地）。M5–M8 核心闭环已按本文件落地；评测脚本仍后置。

### 13.1 首轮 code review 后的三处修正（2026-09-02）

M5–M8 落地后做了一次完整 review，本文件早期表述被以下实测结果覆盖：

1. **归一化动作空间不是 ±1。** checkpoint 的 `policy_postprocessor.json` 里 `ACTION` 是 `MEAN_STD`。原第 4 节写的「clip 到 SmolVLA 动作范围」被实现成固定 `action_clip=1.0`，而 z-score 空间的演示范围是 ±2~5：训练集 149,055 帧里 31.5% 的动作分量 \(|z|>1\)、86% 的帧至少有一维被截断，七个臂关节只剩约 30% 行程。后果是 warmup 执行的不是冻结 VLA，成功率会塌到 0，稀疏奖励永远进不了 buffer。现在改为 per-dim 边界（`vla/action_bounds.py`，`action_bound_margin=1.5`）且**只 clip Actor 输出**。
2. **warmup 步数必须够一个 batch。** V1 无 stride ⇒ 一个执行 chunk = 一条 transition。原 `warmup_env_steps=2000` / `chunk_len=10` 只产出 200 条 < `batch_size=256`，`run_offline_updates` 直接 break，`did_offline` 仍报 True，第一次梯度出现在 env step 2580 —— Actor 用随机权重控了约 580 步。现在 YAML 是 32000（16 环境时取 `num_envs * max_episode_steps` 的整数倍），`--check` 硬性校验 `warmup_env_steps >= batch_size * chunk_len`。
3. **训练循环必须暴露 episode 级成功率。** 原实现只 log critic/actor loss，`CollectResult.success` 算完就丢，无法判断 Actor 是否在改善、也拿不到与 SFT 22% baseline 的对比。现在 warmup 期与 Actor 期成功率分开统计，逐 episode 落 `episodes.jsonl`。

另外两条会静默污染数据的次要项也一并修了：`BatchedChunkCollector` 结束的 env 现在冻结（不再把成功后仍为 True 的 `info["success"]` 写成垃圾 transition，与 `run_episode_batch` 同一约定，不用 ManiSkill 局部 reset）；训练收尾 `flush_pending()` 补写最后一个中段 chunk。

当时判定「有意不改」的两条，已被 13.3 推翻：BC 项的 `mean` 归约（等于论文 \(\|\cdot\|_2^2\) 的 1/80）现在改成 `sum`；`ref_dropout` 默认从 0.5 改成 0，TD target 与 actor loss 里仍走同一条 dropout 路径，只是默认不触发。

### 13.2 并行环境解锁（2026-09-02，用户改判 1.6）

原锁定第 1.6 条是「V1 单环境，向量化往后放」。用户看到 `benchamrk/rl/` 的资源数据后改判：**正式训练允许 `num_envs>1`**。同一张 4090、同一约 35 秒墙钟里，32 环境记录到 960 个 env step 而 4 环境只有 120，显存 4.5→7.3 GB（共 24.6 GB），GPU util 39%→95%。稀疏奖励任务的瓶颈就是等成功样本，所以并行收益大。

解锁前修了三处，缺一处并行就是错的：

1. **UTD 被静默稀释。** 训练循环原本每次 collect 固定做 `utd` 次梯度；`num_envs=N` 时一次 collect 产出 N 条 transition，等于把梯度/数据比除以 N（N=32、UTD=5 时实际只有 0.16）。现在是 `utd * result.added`，单环境行为逐位不变（每轮 added=1）。
2. **`episode_id` 在并行下不唯一。** 原来每个 env 维护自己的计数器，env0 与 env1 的第一局都是 0，`episodes.jsonl` 和 replay 里分不清。现在由 collector 在每局开始时分配全局递增 id，`EpisodeOutcome` / `episodes.jsonl` 另带 `env_index`。
3. **整批同步空转。** 原实现让结束的 env 冻结陪跑到全批结束。现在结束的 env 在当前 chunk 结束时用 ManiSkill 局部 reset（`options={"env_idx": ...}`）立即重开，空转最多 `chunk_len-1` 步。

**并行必须同时设 `reconfigure_every_episodes>0`**（`--check` 会拒绝，smoke 目录只警告）。原因是实测发现的坑：ManiSkill 的 `PegInsertionSideEnv` 在 `num_envs>1` 时把 `reconfiguration_freq` 设成 0，而 peg 长度/半径是在 `_load_scene`（即 reconfigure 时）随机的。实测：普通全量 reset 和局部 reset 都**不会**改变 `peg_half_sizes`，只有 `reconfigure=True` 会。也就是说不加周期性 reconfigure，整个 run 只会见到 `num_envs` 种 peg 几何。`num_envs=1` 不受影响（`reconfiguration_freq=1`，每局自动 reconfigure）。

局部 reset 的安全性已实测确认：`TimeLimitWrapper` 的 `truncated` 读的是 ManiSkill **逐 env** 的 `elapsed_steps`，局部 reset 只清零选中的那几个（`[7,7,7,7]` → `[7,0,7,7]` → `[8,1,8,8]`），reset 返回的 obs 仍是完整 batch。一次带 reconfigure 的全量 reset 约 0.55 s（N=4），不带约 0.02 s，所以周期性 reconfigure 很便宜；代价是它会作废进行中的 episode，因此那一刻的 pending chunk 被丢弃而不是跨 reconfigure 做 bootstrap。

GPU 实测（8 环境、真实冻结 VLA、`max_episode_steps=40`、`reconfigure_every_episodes=8`）：24 局全部记录、24 个 id 全不重复、8 个 `env_index` 都出现、`reconfigures=4`，且 8 个 env 的 peg 半长在周期性 reconfigure 后全部变化。

**并行下仍需注意**：episode 在 warmup 边界跨越时按 Actor 计（混合局归 Actor），所以并行时 `vla_episodes` 可能为 0；想拿到纯 VLA 基线要让 `warmup_env_steps` 是 `num_envs * max_episode_steps` 的整数倍。另外 PegInsertion 的 env 只在成功时才提前结束，失败一律跑到 200 步，所以初期各 env 几乎同步结束，局部 reset 的收益随成功率上升才变明显。

### 13.3 Actor 改残差：两次正式 run 交接即塌方（2026-09-02）

两次正式 run 的结论是一致的，而且和「RL 学不快」无关——**Actor 一接手就比冻结 VLA 差一个数量级**：

| run | env steps | warmup（纯 VLA） | Actor 全程 | Actor 最后 500 局 |
| --- | --- | --- | --- | --- |
| `run_20260902_023144` | 997,748 | 24/160 = **15.0%** | 118/4832 = **2.4%** | **5.8%** |
| `run_20260902_060320` | 310,231 | 24/160 = **15.0%** | 12/1392 = **0.86%** | **1.4%** |

100 万 env step 之后 Actor 也没回到 warmup 的 15%。这不是探索不足或者 Critic 学得慢，是**交接那一刻的策略就已经不是 SFT 策略了**。同一时刻的日志给出了直接证据：`bc_dist=0.0486`。V1 的 BC 项是 `mean` 归约，所以这是每个动作分量的均方差，换算成 RMS 是 \(\sqrt{0.0486}\approx0.22\) 个归一化单位；`MEAN_STD` 空间里各臂关节的 std 是 0.14–0.22 rad，也就是 **每个关节偏离 SFT 参考约 2–4 度、连续 10 步**。PegInsertion 的插入间隙只有毫米级，这个幅度足以让绝大多数局失败。

同时 Critic 已经退化成常数：`critic_loss=8e-4`、`q_mean=0.4571`、`target_mean=0.4563`，`actor_loss=-0.4547` 几乎等于 `-q_mean`。也就是说 Q 对动作没有区分度，Actor 的梯度里只剩噪声，而 BC 这一侧因为 `mean` 归约又被削弱到 1/80——Q 项和 BC 锚定的实测量级比约 10:1。Actor 于是自由漂移到离参考 2–4 度的地方，再也回不来。

三仓对照后确认这是我们独有的问题：rlt-openpi 用 **残差 Actor + 末层零初始化**（交接时恒等于参考），afengleafs 与 RajatDandekar 虽然是非残差，但他们的 Actor 只需要拟合更平滑的动作空间、且没有我们这种 15% 的强 baseline 要保住。我们既要保 baseline 又要非残差，等于让一个随机初始化的 MLP 在几千次梯度里重新学会 SFT 策略——`bc_dist` 的数字说明它没学会。

本轮改动（用户决定：改残差，范围 phase 0/1/2）：

1. **残差 Actor（`use_residual_actor: true`，`rl/actor.py`）。** \(a=\tilde a+\Delta\)，`net` 末层 weight/bias 零初始化，所以 step 0 的 Actor **逐位等于**冻结 VLA chunk，交接不可能低于 SFT 基线。同时把 `ref_dropout` 的作用点从「残差相加」挪到「网络输入」：dropout 只遮蔽送进 MLP 的那份 reference，残差相加永远用真 reference，否则 dropout 会直接把动作打到 \(\Delta\) 本身。
2. **BC 项改 `sum` 归约 + 确定性 \(\mu\)（`bc_reduction: sum`）。** 现在等于论文的 \(\|\cdot\|_2^2\)（在 \(C\times\text{action\_dim}=80\) 个元素上求和），`bc_beta=1.0` 的语义随之变化；BC 距离用确定性 \(\mu\) 而不是采样动作，避免把 `action_std` 的噪声算进「偏离参考」。
3. **交接硬门槛（`bc_pretrain_updates: 2000`，`handover_bc_threshold: 1e-4`）。** warmup 结束后先做纯 BC 蒸馏，再测确定性 \(\mu\) 与参考的均方距离；不达标就延长一轮 warmup，连续 5 次不达标直接中止（此时只跑过 warmup，中止很便宜）。残差 Actor 本来就该一次通过，这条是回归护栏。
4. **`ref_dropout` 默认 0。** rollout 永远带 reference，训练时随机丢掉它只是在制造 train/inference 不一致，还削弱了 reference 条件化的精度。0.5 留作第一个 ablation 旋钮，残差下是安全的。
5. **探索噪声与训练噪声解耦（`explore_std: 0.02`，`action_std: 0.05`）。** 原来 rollout 和 TD backup 共用 `action_std=0.05`；0.05 归一化单位约 0.6 度/关节，连续 10 步足以毁掉插入。`explore_std=0.02`（约 0.35 度）只作用于 rollout，`action_std` 仍是训练图里的平滑噪声。
6. **collector 不再清零未执行的动作尾巴（`rollout/collector.py`）。** 成功在 chunk 中段发生时，原实现把剩余步的动作写成 0 存进 replay。稀疏奖励下这些恰好是唯一的高价值转移，Critic 于是学到「零动作 = 高 Q」，直接解释了 Q 的退化。现在只清零奖励尾巴，动作保留原值。
7. **Actor batch 与 Critic batch 分离采样（`actor_success_sample_frac: 0.0`）。** Critic 仍按 `success_sample_frac=0.25` 上采样成功转移（它需要这些样本才有 TD 信号），Actor 改为独立均匀采样。首次开启 Actor 侧上采样的那个 run 反而更差（0.86% vs 1.5%，同一 env step），原因是把策略梯度对准了当前 Actor 根本不会到达的 warmup-VLA 状态。
8. **周期性 reference 探针（`reference_probe_every_episodes: 320`）。** 原来 `vla_success_rate` 在 warmup 结束后就冻结在 15%，后面 100 万步没有活的基线可比。探针会先 reset 整批、跑 `reference_probe_env_steps`（0 = `num_envs * max_episode_steps` = 3200）步纯参考 chunk、再 reset，所以探针局不会和 Actor 局混在一条 episode 里；320 局一次约占 5% 的 env step。
9. **诊断指标（`rl/agent.py`、`rl/train.py`）。** 新增 `q_std`、`q_success_mean`、`q_rest_mean`（Q 有没有区分度）、`bc_dist_det`（确定性 \(\mu\) 的偏离）、窗口化的 `actor_success_rate_20` / `vla_success_rate_20`、replay 里的 `replay_success_slots` / `replay_reward_slots`。

**尚未做、留到下一轮**：\(\gamma\) 与 chunk 级折扣的重新标定（200 步 × \(\gamma=0.99\) 的有效视界只有 100 步，而稀疏奖励往往在 150 步后才出现）、\(C\) 是否该缩短、以及 stride。这些属于 phase 3，等残差交接确认能守住 15% 基线之后再动。

**验证状态**：186 个 CPU 测试全绿；`--check` 通过（残差已不再被拒绝，per-dim 边界仍正常导出）；`--smoke` 覆盖 warmup → 交接门槛 → 离线 warm-start → Actor 控制 → 探针 → 回到 Actor 控制的完整路径。

`--gpu-smoke`（4 环境、真实冻结 VLA + ManiSkill，`benchamrk/rl/gpu_smoke_env4_20260902_105716.md`）确认了残差交接的关键性质：

- `handover attempt=1 bc_dist_det=0.000e+00 passed=True`——零初始化残差在交接时**逐位**复现冻结 VLA chunk，一次过门槛。
- `post-warm-start bc_dist_det=6.66e-05`，4 次离线更新后仍在 `1e-4` 阈值内。
- `bc_dist=0.0031`（`sum` 归约、确定性 \(\mu\)）⇒ 每分量均方差 3.9e-5、RMS 约 0.006 归一化单位，折合 **每关节约 0.08 度**；对照改动前的 2–4 度，量级差了约 40 倍。
- Critic 不再退化成常数：`q_std=0.045 → 0.071`（改动前是 0.0000）。

资源：4 环境 `nvidia_peak=4523 MiB`、`torch_peak=1130 MiB`、`rss_peak=5605 MiB`、`gpu_util_peak=51%`，正式训练的 16 环境仍在 24.6 GB 之内。

（题外记录：这次 GPU smoke 一开始跑不起来，原因是实例迁移后系统盘缺了 `libEGL.so.1`（apt `libegl1`）——`libGLX_nvidia.so.0` 的 ICD 初始化最后一步会 dlopen 它，缺了就返回 `VK_ERROR_INITIALIZATION_FAILED`，表现为 `vk::createInstanceUnique: ErrorIncompatibleDriver`，而 CUDA/torch 完全正常。装回 `libegl1` 并把 `nvidia_icd.json` 补到 `/usr/share/vulkan/icd.d/` 后恢复，已记进 `AGENTS.md`。）

### 13.4 残差交接成功，但 Actor 被 BC 钉死（2026-09-02）

13.3 的残差改动按设计生效了：`run_20260902_110210` 一次通过交接门槛（`bc_dist_det=0.000e+00`），同一 env step 上 Actor 成功率相对旧 run 提升约 12 倍（≤67k step 时 13/176 = 7.4%，旧 run 是 1/160 = 0.6%）。但跑到 17.5 万 step 后暴露出**新的失败模式：Actor 根本没有偏离冻结 VLA，RL 在策略层面等于没有生效**。

证据有三条：

1. `bc_dist`（即 \(\|\Delta\|^2\)，80 个元素求和）从交接后一路平在 0.0004–0.0010，没有任何增长趋势。换算成每分量 RMS 是 0.0027 归一化单位，按 checkpoint 的 action std（`[0.202, 0.202, 0.069, 0.302, 0.070, 0.141, 0.441, 0.958]`）折合 **joint0 上约 0.03 度**。而 rollout 的 `explore_std=0.02` 对应 0.08°（j2/j4）到 0.51°（j6），比 Actor 自己的修正大了一个数量级——执行出去的就是「VLA chunk 加噪声」。
2. Actor 成功率 14 个窗口（每窗 48 局）在 4.2%–22.9% 之间纯随机跳动，累计 71/672 = 10.6%，**13 万 step 没有趋势**。两次 reference 探针刷新后的实时 VLA 基线是 29/192 = 15.1%。
3. 残差末层权重从零初始化长到 rms 0.022、bias 只有 4e-4，说明 \(\Delta\) 是状态相关的、梯度确实在流，不是代码 bug。

根因是 13.3 把 BC 从 `mean` 改成 `sum` 时过头了。Actor loss \(L=-Q(\mu)+\beta\sum_i(\mu_i-\text{ref}_i)^2\) 的稳定点是

\[\mu_i-\text{ref}_i=\frac{\partial Q/\partial\mu_i}{2\beta}\]

拿两次 run 反解 Critic 的动作梯度，结果几乎完全一致：

| run | 归约 | 有效 \(\beta\) | 实测 \(\Delta\) 的 RMS | 反解 \(\|\partial Q/\partial\mu\|\) |
| --- | --- | --- | --- | --- |
| `run_20260902_023144` | `mean` | 1/80 = 0.0125 | 0.220（2.5°） | 0.00551 |
| `run_20260902_110210` | `sum` | 1.0 | 0.0027（0.03°） | 0.00548 |

也就是说 **Critic 的动作梯度强度前后没变，80 倍的 \(\Delta\) 差异 100% 来自 \(\beta\)**。`mean` 下的 0.0125 太弱（2.5° 崩溃），`sum` 下的 1.0 太强（0.03° 等于没动），合理值在中间；用上式可以直接预测：\(\beta=0.1\) 对应 0.32°，\(\beta=0.05\) 对应 0.65°。

还有一条容易误读的指标：`q_gap=0.58` 比较的是成功轨迹转移与其余转移的 Q，衡量的是**状态**区分度；Actor 需要的是同一状态下的**动作**敏感度，那个量是 0.0055，小两个数量级。Critic 学会了「哪些局面好」，但因为 `explore_std=0.02` 只让它见过参考动作周围约 0.2 度的范围，它基本没学到「这里换个动作会不会更好」。

**本轮只补测量，不动 \(\beta\)**（用户决定）。盲目调小 \(\beta\) 有可能只是把 2.5° 的崩溃重演一遍，因为我们完全不知道 \(\partial Q/\partial a\) 指的方向是否有用。新增：

1. **梯度天平（`rl/agent.py::_actor_diagnostics`）。** `grad_q_rms` 用 `new_action` 上的 backward hook 捕获（乘回 batch size 抵消 `.mean()`），`grad_bc_rms` 是解析值 \(2\beta\,\mathrm{rms}(\mu-\text{ref})\)，`grad_ratio` 是两者之比。稳定点上比值趋近 1；远大于 1 说明 Actor 还在被推离参考。这让 `bc_beta` 从「猜」变成「读日志」。
2. **Critic 的动作意见。** `q_ref_mean` = \(Q(x,\text{ref})\)，`q_adv_det` = \(Q(x,\mu)-Q(x,\text{ref})\)，`q_adv_exec` = \(Q(x,a_{\text{exec}})-Q(x,\text{ref})\)。`q_adv_det` 长期为 0 就意味着策略梯度无话可说，此时调小 \(\beta\) 只会放大噪声。零初始化残差在 step 0 上 `q_adv_det` 恒等于 0，这一点有测试锁定。
3. **确定性 Actor 探针（`probe_deterministic_actor: true`）。** reference 探针结束后立刻在同一批 reconfigure 过的环境上再跑一段 `explore_std=0` 的 Actor 探针，成绩记进独立的 `det_*` 计数器。这是把「残差有害」和「探索噪声很贵」拆开的唯一办法——`run_20260902_110210` 的 10.6% 对 15.1% 只有 1.7σ，两种解释都说得通。代价是探针开销从约 5% 翻倍到约 10%。
4. **`episodes.jsonl` 新增 `mode` 字段**（`warmup` / `actor` / `probe_ref` / `probe_det`）。`probe_det` 的局虽然 `use_actor=True`，但不进 `actor_*` 计数器，否则两条曲线就没法比。

**验证**：193 个 CPU 测试全绿；`--check` / `--smoke`（四种 mode 全覆盖）通过；默认 `--gpu-smoke` 资源与上一版一致（`nvidia_peak=4523 MiB`）；探针链路另做了一次真实环境验证（4 环境、`max_episode_steps=20` 以便 episode 能结束）：2 次 reference 探针 + 2 次确定性探针，`episode modes {'warmup': 4, 'actor': 8, 'probe_ref': 8, 'probe_det': 8}`，相位切换的 reconfigure 正常。那次验证成功率全 0 是因为 20 步远不够 PegInsertion 完成，它只验证链路。

**下一个 run 要读的三个数**：`grad_ratio`（Actor 是否已停在稳定点）、`q_adv_det`（Critic 是否真的想让 Actor 离开参考）、`det_success_rate` 对 `vla_success_rate`（去掉噪声后 Actor 到底比不比参考差）。这三个数出来之后再定 \(\beta\)，以及是否需要把残差改成 \(\Delta=\delta\cdot\tanh(\cdot)\) 的结构性有界形式。

### 13.5 三个诊断的读数：Critic 对动作是盲的（2026-09-02）

`run_20260902_115329`（16 环境，读到 16.9 万 / 100 万 env step，交接后 13 万步）把 13.4 埋的三个数全测出来了，结论一致指向**同一个根因，而且不是 \(\beta\)**。

**1. `grad_ratio` = 1.41–1.47，13 万步无趋势 → Actor 确实已经停在稳定点。**

`grad_q_rms=0.0081`、`grad_bc_rms=0.0056`；反解稳定半径 \(\mathrm{rms}(\Delta)=\text{grad\_q}/(2\beta)=0.0041\)，实测 `bc_dist=0.00066` 折合 \(\mathrm{rms}(\Delta)=0.00287\)。Actor 停在稳定半径的 70% 处不再外移。顺带验证了 13.4 的反解方法：那里从旧 run 反推的 \(\|\partial Q/\partial\mu\|=0.0055\)，现在直接测出 0.0081，同一量级。

**2. `q_adv_det` = 0.0010–0.0015，同样无趋势 → Critic 对动作没有任何可信的意见。**

| 对比量 | 数值 | `q_adv_det` 占比 |
| --- | --- | --- |
| Critic 自身 TD 残差 \(\sqrt{\text{critic\_loss}}\) | 0.0181 | **1/14** |
| `q_std` | 0.318 | 0.40% |
| `q_gap`（状态区分度） | 0.60 | 0.21% |

`q_adv_det` 比 Critic 自己的拟合误差还小一个数量级，是纯噪声。`q_adv_exec`（随机探索扰动相对参考）也是正的 0.0004，一个随机方向也「看起来更好」，进一步说明梯度方向无意义。按 `_actor_diagnostics` 自己的判据，**此刻调小 \(\beta\) 只会放大噪声**。

**3. `det_success_rate` 对 `vla_success_rate`：没测出来，样本量不够。**

探针只触发过 1 次（episode ~480 / step ~97k）。`episodes.jsonl` 分模式：`warmup` 24/160 = 15.0%、`actor`（带 `explore_std`）61/608 = 10.0%、`probe_ref` 2/16、`probe_det` 2/16。n=16 的 95% CI 约 [2%, 38%]，判不了任何东西；参考基线合并 26/176 = 14.8% 对带噪 Actor 10.0% 是 z=1.76（p≈0.08），和上一个 run 一样的 1.7σ。但机制上可以定性：\(\mathrm{rms}(\Delta)=0.00287\)（各关节 0.011°–0.157°）比 `explore_std=0.02`（0.08°–0.51°）小一个数量级，所以确定性 Actor 实质就是参考策略，那 4.8 个点几乎只能来自探索噪声。

**根因：Critic 的输入把动作量化掉了。**

`a = ref(x) + explore_std` 噪声，而 `ref` 本身近似是 x 的确定函数。V1 Critic 的输入是 `cat([x(521), a(80)])`，动作那 80 维在 `MEAN_STD` 空间里绝对值是 O(1)–O(5)，**变化幅度只有 0.02**——250:1 的动态范围。经过第一层 `Linear → LayerNorm`，动作对激活的贡献被 521 维状态淹没，于是 Q 完全靠 \(V(x)\) 就能把 TD loss 压到 3e-4。这正是观测到的：状态区分度 0.60，动作区分度 0.0013。动作梯度是噪声 → Actor 停在 BC 稳定点 → rollout 的动作支撑永远只有参考周围 0.02 → Critic 永远学不到动作依赖。闭环自锁。

这个失败模式已经复现成单元测试（`tests/test_rl_critic.py::test_raw_action_input_is_blind_to_explore_std_sized_actions`）：回归目标只依赖动作落在参考的哪一侧、幅度恰好一个 `explore_std`，其余为 521 维随机状态与 O(2) 随机参考。原始 `(x, a)` 输入的 Critic 最终 MSE = 1.006（等于目标方差，即学到的关于动作的信息**恰好是零**），\(Q(\text{ref}+d)-Q(\text{ref}-d)=0.0001\)；残差坐标下 MSE = 0.015、分离度 1.96（真值 2.0）。

**本轮改动（用户决定：只做 P0-1，不动 \(\beta\)）**

**Critic 换残差坐标**（`critic_residual_input: true`，`rl/critic.py`）。输入从 `cat([x, a])` 改为 `cat([x, ref, (a - ref) / critic_residual_scale])`。因为 \(a=\text{ref}+\Delta\)，信息量严格不减（还多给了 ref 本身），关键是**缩放**：把 0.02 尺度的变化变成 O(1) 的输入通道。`critic_residual_scale: 0` 表示取 `explore_std`，于是「一个 `explore_std` 的扰动」恰好落在该通道的 1.0 上。`reference_chunk` 是 `ChunkCritic.forward` / `min_q` 的必填位置参数（即使 ablation 关掉也必填），漏传是签名错误而不是静默变成 \(a/\text{scale}\)。`critic_residual_input: false` 可回到 V1，`--check` 会给出 action-blind 警告。

**验证**：202 个 CPU 测试全绿；`--check` 打印 `critic_residual_input` 与解析后的 `critic_residual_scale`；`--smoke` 四种 mode 全覆盖；`--gpu-smoke`（4 环境，`benchamrk/rl/gpu_smoke_env4_20260902_124328.md`）资源与上一版完全一致（`nvidia_peak=4523 MiB`），交接门槛仍 `bc_dist_det=0.000e+00` 逐位通过。

**⚠️ 必须在正式 run 之前处理：\(\beta\) 的标定已经失效。**

残差通道除以了 `scale`，所以链式法则给出 \(\partial Q/\partial a=(1/\text{scale})\cdot\partial Q/\partial(\text{residual})\)——**动作梯度被机械放大约 \(1/0.02=50\) 倍**。而 BC 稳定点是 \(\mathrm{rms}(\Delta)=\text{grad\_q}/(2\beta)\)，随之同比放大。GPU smoke 里（Critic 只有 16 步梯度、12 条转移，量级不可外推，只说明通道通了）`grad_q_rms` 从旧 run 的 0.0081 变成 0.24，`bc_dist_det` 从 6.66e-05 变成 1.06e-03。

用旧 run 的 \(\beta=1.0\) 直接开正式训，稳定点可能落到 \(\mathrm{rms}(\Delta)\approx0.05\text{--}0.1\)，折合每关节 0.5°–5°——**正好是 13.3 记录的 2°–4° 塌方区**。换句话说这次修复有可能在不动 \(\beta\) 的情况下自己走进旧的失败模式。真实的 \(\text{grad\_q}\) 要等 Critic 在真环境上训起来才知道。

**护栏（用户决定：`actor_drift_ceiling`，\(\beta\) 先不动）**

正式 YAML 默认 `actor_drift_ceiling: 1.6e-3`（RMS 0.04 归一化单位，约 0.46°/臂关节）：是交接门槛的 16 倍，压在 13.4 \(\beta=0.1\) 预测之上所以不挡预定调参带，是 13.3 实测塌方点 `bc_dist_det=4.86e-2` 的 1/30。每个 `log_freq` 用 `reference_fidelity` 重测一遍（和交接门槛同一量），超阀就存 checkpoint 后 abort，报错里带 `grad_q_rms` 反解出的「把 Actor 钉在天花板上的 \(\beta\)」，所以一次失败的 run 仍然给出测量。`--smoke` / `--gpu-smoke` 关掉这条（GPU smoke 在 16 步梯度上已经测到 `bc_dist_det=3.5e-3`，那是 Q 噪声不是塌方）。0 关闭；`--check` 对 `<= handover_bc_threshold` 报错，对 0 给警告。

**尚未做、留到下一轮**：P0-2 探索噪声改 chunk 级相关（现在是 `[B, C, 8]` 上 `randn_like`，80 个独立分量，对毫米级插入是抖动：足以毁掉成功率，方向却在 80 维里几乎正交于任何有用方向）、P0-3 部分环境用大 `explore_std` 的混合探索、P1 ref/det 探针配对同 seed（现在 320 局一次 × 16 局，跑满 1M 也只累计 ~240 局/臂）、P2 \(\Delta=\delta\cdot\tanh(\cdot)\) 有界残差与 \(\beta\) 阶梯。
