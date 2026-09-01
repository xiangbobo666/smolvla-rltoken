# 三仓库：从 VLA hidden 提取 RL Token 调研

对照论文 Eq.1–2（`docs/knowledge-base/详解/RL Token loss公式.md`）和本仓库计划（`docs/knowledge-base/SmolVLA_RLT_plan.md`），逐行核对三个开源复现：

| 仓库 | 路径 | 基座 | 代码入口 |
| --- | --- | --- | --- |
| **afengleafs / smollvla_rltoken** | `/root/autodl-tmp/smollvla_rltoken` | LeRobot SmolVLA（README：lerobot 0.5.1，SmolVLM2-500M） | `rlt/rl_token.py`，`rlt/train_rl_token.py` |
| **RajatDandekar / RL-Token-SmolVLA** | `/root/autodl-tmp/RL-Token-SmolVLA` | LeRobot SmolVLA v0.4.4 + patch | `smolvla_rlt/modeling_smolvla_rlt.py`，`scripts/train_rlt_stage1.py`，`lerobot_patches/modeling_smolvla.patch` |
| **yknxh / rlt-openpi** | `/root/autodl-tmp/rlt-openpi` | OpenPI PI0 / PI0.5（PaliGemma，hidden 2048） | `src/rlt_openpi/vla/embedding_extractor.py`，`src/rlt_openpi/models/rl_token.py`，`src/rlt_openpi/training/rl_token_trainer.py` |

本仓库本地栈（`AGENTS.md`）：`lerobot[smolvla]==0.4.4`，`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，**只用 VLM 前 16 层**，`text_config.hidden_size = 960`，action expert 宽 `0.75 × 960 = 720`。下面所有 SmolVLA 形状以这份本地代码为准；Rajat 仓库里写死的 576/432 **与当前 smolvla_base 不一致**。

论文目标（压缩成一句话）：

\[
z_{\mathrm{rl}} = g_\phi([z_{1:M}, e_{\mathrm{rl}}])_{M+1},\qquad
\bar z_i = \mathrm{sg}(z_i),\qquad
\mathcal L_{\mathrm{ro}} = \mathbb E_{\mathcal D}\Big[\sum_{i=1}^{M}\big\|h_\phi(d_\phi([z_{\mathrm{rl}},\bar z_{1:i-1}]))_i - \bar z_i\big\|_2^2\Big].
\]

---

## 0. 一页对照

| 项目 | 论文 / 本仓库计划 | afengleafs | RajatDandekar | yknxh / rlt-openpi |
| --- | --- | --- | --- | --- |
| **抠哪一层** | VLA 最后一层 hidden | VLM prefix **最终层**（16 层后 + final RMSNorm） | VLM prefix **最终层** + expert suffix **最终层** | PaliGemma LM **最终层** `last_hidden_state` |
| **抠哪些 token** | \(z_{1:M}\)；脚注：任务指令固定时可只留图像 token | 默认 **仅图像 connector token**（每相机 64，3 相机 \(M=192\)）；可选保留 lang+state | **全部 prefix** + **全部 action-expert token**（chunk_size=50） | **全部 prefix**（图像 + 语言）；**不含 state** |
| **shape \([B,M,D]\)** | \([B,M,D_{\mathrm{VLA}}]\) | \([B,M,960]\)；image-only 时 \(M=64\times N_{\mathrm{cam}}\)；全 prefix 实测 \(M=241=192+48+1\) | 两路：`vlm [B,L_p,576]`、`expert [B,50,432]`（配置写死；对 SmolVLM2 应为 960/720） | \([B,M,2048]\)，\(M\) 随相机数和 prompt 变长，`pad_mask` 标有效位 |
| **磁盘 embedding cache** | 计划有（避免反复 VLA forward） | **无** | **无** | **无** |
| **内存 KV cache** | 未规定 | **有**：prefix `past_key_values`，复用于 flow-matching 采样 \(\tilde a_{1:H}\) | Stage 1 `use_cache=False`；推理时另走 `sample_actions` | 抽取时 `use_cache=False`；参考动作另跑完整 diffusion |
| **Encoder 投影** | \(u_i=W_{\mathrm{in}}z_i\) | `Linear(960 → 512)` | 两路：`Linear(576→256)`、`Linear(432→256)` | **无**，直接在 2048 维上做 attention |
| **Learnable token** | \(e_{\mathrm{rl}}\) 拼到序列末尾 | `e_rl: Parameter(d)` | `rl_token: Parameter(1,1,d)` | `e_rl: Parameter(1,1,2048)` |
| **位置编码** | 计划有 learnable pos | **有**：`enc_pos [max_recon+1, d]`、`dec_pos [max_recon+1, d]` | **无**（decoder 注释写了 pos，代码未实现） | **无**（依赖 VLA RoPE 已写进 \(z\) 的内容） |
| **层数 / 头数** | 计划未锁死 | 2 / 8，`d_model=512`，`mlp_ratio=4`，`norm_first` | 4 / 8，`rlt_hidden=256`，dropout 0.1 | 2 / 8，`d_model=2048`，FFN \(4D\) |
| **\(z_{\mathrm{rl}}\) 取位** | 最后一个位置 \(M+1\) | **是** `out[:, -1]`，shape `[B, 512]` | **是** `out[:, -1]` + LayerNorm，`[B, 256]` | **是** `out[:, -1]`，`[B, 2048]` |
| **Decoder 形态** | causal + teacher forcing，\(z_{\mathrm{rl}}\) 当第 0 个 token | `TransformerEncoder` + **causal mask**（decoder-only） | `TransformerDecoder`：**learnable query + 对 \(z_{\mathrm{rl}}\) 的 cross-attn**，**不是** teacher forcing | `TransformerDecoder`：teacher forcing **且** extra cross-attn 到 \(z_{\mathrm{rl}}\) |
| **Causal mask** | 必须有 | **有** `triu(-inf, diagonal=1)` | **无** | **有** `generate_square_subsequent_mask` |
| **Loss** | \(\sum_i\|\cdot\|_2^2\) | 有效 token 上的 **masked mean-MSE** | `mse(vlm)+mse(expert)`，`F.mse_loss` 全元素平均 | 有效 token 上的 **masked mean-MSE** |
| **sg / 冻 VLA** | \(\bar z=\mathrm{sg}(z)\)，默认冻 VLA | `z.detach()` + `@torch.no_grad()` 抽取；`alpha=0` 时 `requires_grad_(False)` | 抽取 `@torch.no_grad()` + 再 `.detach()`；全部 VLA `requires_grad=False` | `z.detach()`；`alpha=0` 时 VLA freeze；joint 时 prefix **仍 detach**，只 `L_vla` 更新 VLA |
| **训练数据从哪来** | 计划：先 cache 再训 Encoder | **每 step 在线 VLA prefix forward**，LeRobot demo + SmolVLA preprocessor | **每 step 在线 prefix+suffix forward**（需要 GT action） | **每 step 在线 prefix forward**，OpenPI transform 链 |
| **Optimizer / lr / batch / steps** | 未锁死 | AdamW，`lr=1e-4`，wd=0.01，clip=1，**B=16**（CLI；dataclass 默认 8），**5000** 步 | AdamW，`lr=1e-4`，wd=1e-5，warmup 200 + cosine，clip=1，**B=16**，**5000** | AdamW，`lr=1e-4`，wd=1e-5，warmup 500（linear），clip=1，**B=32**，**5000** |
| **Checkpoint 存什么** | encoder（进 Stage 2）；decoder 可丢 | `rl_token.pt`：encoder+decoder 整模 + config + step；`alpha>0` 另存 `smolvla_sft.pt` | `checkpoint_stepN.pt` + `best_checkpoint.pt`：enc/dec/opt/config/loss；另 `loss_history.json` | `rl_token_stepN.pt`：model+opt+scheduler+step+config；joint 时再存 VLA 权重与 VLA optimizer |

**和论文最对齐的 SmolVLA 实现是 afengleafs**；**和论文最对齐的 π 系实现是 rlt-openpi**。Rajat 的 decoder **不是**论文的自回归重建，且 VLM 维数写死 576，不能直接接到当前 `smolvla_base`。

---

## 1. VLA hidden 从哪抠

三仓都抠 **Transformer 堆叠之后、final norm 之后的 last hidden**，没有人抠中间层、没有人抠 vision encoder 的 patch token（未过 VLM）。差别在：**prefix-only vs prefix+suffix**，以及 **哪些 prefix token 留下**。

### 1.1 afengleafs — prefix-only，默认只要图像 token

抽取类：`SmolVLAPrefixExtractor.extract`（`rlt/rl_token.py`）。

流程与 `VLAFlowMatching.sample_actions` 的 prefill 相同，但把 prefix 最终 hidden 留下来：

1. `prepare_images` / `prepare_state` / 语言 token。
2. `model.embed_prefix(images, img_masks, tokens, masks, state=state)`。
3. `vlm_with_expert.forward(..., inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True)`。
   - `inputs_embeds[1] is None` → **只跑 VLM，不跑 action expert**。
   - **必须** `fill_kv_cache=True`：本地 SmolVLA 默认 `attention_mode=cross_attn`，`fill_kv_cache=False` 会走 `forward_cross_attn_layer` 并对 `None` expert 取 `.dtype`。本仓库 Stage 1 用 `use_cache=False, fill_kv_cache=True`（不存 KV，但仍走 self-attn prefill）。
   - 本地 `smolvlm_with_expert.py`：循环 `num_vlm_layers=16` 层，最后 `models[i].norm(hidden)`，返回的就是 **第 16 层后的 RMSNorm 输出**。
4. `prefix_out = outputs_embeds[0]`，cast `float32`。

Prefix 布局（与本地 `embed_prefix` 一致）：

```
[ cam0 connector tokens | cam1 | ... | language tokens | 1 × state token ]
```

- 图像：SigLIP → **connector（pixel shuffle）**，512×512 时 **每相机 64 token**（smoke test 实测 3 相机 `n_img=192`）。
- 语言：`tokenizer_max_length=48`（pad_language_to 依 preprocessor）。
- state：`state_proj(s)`，**1 个 token**。
- 全 prefix 实测：`M_total = 192 + 48 + 1 = 241`，`D = 960`。

默认 `use_image_tokens_only=True`（论文脚注 1：固定任务指令可丢掉语言 token；state 已在 RL 状态 \(x=(z_{\mathrm{rl}}, s^p)\) 里再拼一次，所以这里也丢掉）：

```
z, mask = z[:, :n_img], mask[:, :n_img]   # [B, 192, 960]
```

若 `M > max_recon_tokens=256`，`linspace` 均匀下采样到 256。image-only 的 192 不会触发。

**KV cache（内存，不是文件）**：`past_key_values` 与 `prefix_pad_masks` 一并返回，给 `sample_reference_chunk` 做 10 步 flow matching，避免再跑一遍 prefix。这是部署/Stage 2 的优化，Stage 1 重建 **不用** 这份 cache。

**不需要 GT action。Stage 1 只看观察。

### 1.2 RajatDandekar — prefix + noisy suffix，两路 hidden

LeRobot 本体没有 `extract_embeddings`。仓库用 patch 加到 `VLAFlowMatching`：

`lerobot_patches/modeling_smolvla.patch`：

- 与 `forward()` 同一条训练前向：对 GT action 采样 `noise, time`，构造 \(x_t = t\cdot\varepsilon + (1-t)a\)。
- `embed_prefix` + `embed_suffix(x_t, time)` 拼接。
- `vlm_with_expert.forward(inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)`。
- 返回：
  - `prefix_out`：`[B, L_prefix, vlm_hidden]`
  - `suffix_out[:, -chunk_size:]`：`[B, 50, expert_hidden]`

也就是说 **\(z\) 里混进了当前噪声水平的 action-expert token**，不是纯观察表征。同一条观察、不同 `(noise, time)` 会得到不同 expert hidden。Stage 1 每个 step 重新采样，encoder 必须对这条随机性鲁棒。

配置写死：

```python
vlm_hidden_dim = 576      # 第一代 SmolVLM-500M
expert_hidden_dim = 432   # 576 * 0.75
```

当前 `lerobot==0.4.4` + `smolvla_base` 是 **SmolVLM2，960 / 720**。若按仓库默认 config 接本地 checkpoint，投影层会维度对不上。测试里 `VLM_SEQ_LEN=120` 只是假数据，不是真实 prefix 长度。

Stage 1 循环（`scripts/train_rlt_stage1.py`）：

```python
with torch.no_grad():
    vlm_emb, expert_emb = vla_model.extract_embeddings(
        images, img_masks, lang_tokens, lang_masks, state, actions,
    )
```

**必须有 `actions`**。数据集要能 `prepare_action`。语言侧直接读 `batch[OBS_LANGUAGE_TOKENS]`，但 DataLoader **没有**挂 SmolVLA preprocessor（afengleafs 有 `make_smolvla_pre_post_processors`）。若原始 LeRobot 集只有 `task` 字符串、没有现成 token，这里会 KeyError。这是管线缺口，不是论文差异。

### 1.3 yknxh / rlt-openpi — PI0 prefix-only，全部 prefix token

`EmbeddingExtractor.extract_embeddings`：

1. `_preprocess_observation`
2. `embed_prefix(images, img_masks, lang_tokens, lang_masks)` — **不把 state 塞进 prefix**
3. `paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=False)`
4. `z = prefix_out.float()` → **`[B, M, 2048]`**

`M` = 各相机图像 token + 语言 token（padding 由 `prefix_pad_masks` 标，`True=有效`）。PI0/PaliGemma 常见每图 256 token（224 分辨率）；2 相机约 \(512+L_{\mathrm{lang}}\)，3 相机 DROID 覆盖 `three_camera_droid` 时再加一路。仓库 **没有** image-only 过滤，语言 token 始终进 encoder。

联合微调（`vla_finetune_alpha > 0`）走 `forward_joint`：monkey-patch `paligemma_with_expert.forward`，从完整 prefix+suffix 训练前向里抓 `prefix_out`，并立刻 `.detach()`。作者注明 PI0 的 attention 使 prefix 不看 suffix，因此这份 prefix 与 prefix-only 前向数值相同，但只做一次 forward。

参考动作 **不复用** 抽取时的 KV：`sample_actions` 再跑完整 diffusion。

### 1.4 层与 token：和本仓库计划的关系

本仓库计划写 \(Z\in\mathbb R^{M\times D_{\mathrm{VLA}}}\)，未规定是否含 expert、是否丢语言。

对 **SmolVLA + PegInsertion 这类固定指令任务**，afengleafs 的选择更贴论文脚注和本地 hidden 维：

- 层：VLM **第 16 层 + final norm**（不是 32 层满栈，也不是 vision tower）。
- token：默认图像 connector；`D=960`。
- 不要把 noisy expert suffix 算进 \(z\)（那是 Rajat 的额外选择，观察不再是函数）。

---

## 2. Encoder：投影、\(e_{\mathrm{rl}}\)、位置、层、\(z_{\mathrm{rl}}\)

三仓都实现了论文 Eq.1 的骨架：**序列末尾拼 learnable token，取最后一个位置**。差在宽度、要不要投影、有没有位置编码和 padding mask。

### 2.1 afengleafs `RLTokenModule`

```
z [B,M,960]
  → enc_in_proj Linear(960,512)
  → cat e_rl [B,1,512]
  → + enc_pos[:M+1]
  → TransformerEncoder × 2, 8 heads, FFN=2048, gelu, dropout=0, norm_first, 末尾 LayerNorm
  → z_rl = out[:, -1]     # [B, 512]
```

- `e_rl`：`nn.Parameter(torch.randn(d)*0.02)`，无单独 LayerNorm。
- padding：`src_key_padding_mask = cat(~mask, False)`，RL 位永不 mask。
- `d_model=512` 是相对论文「\(z_{\mathrm{rl}}\) 与 VLA 同宽」的缩减（~15M 参数）；`--d-model 960` 可对齐 1×VLA 宽。

### 2.2 Rajat `RLTokenEncoder`

```
vlm [B,L_v,576] → vlm_proj → 256
expert [B,L_e,432] → expert_proj → 256
cat [vlm, expert, rl_token]     # 无位置编码、无 pad mask
TransformerEncoder × 4, 8 heads, FFN=1024, dropout=0.1, norm_first
z_rl = LayerNorm(out[:, -1])   # [B, 256]
```

- 输入是 **两条不同宽度的流**，靠两个投影对齐，再在 token 维拼接。论文是单一 \(z_{1:M}\)。
- 无 pad mask：变长 prefix 若右侧 padding，padding 向量也会进 attention。
- `rlt_hidden_dim=256` 更激进的瓶颈。

### 2.3 rlt-openpi `RLTokenEncoder`

```
z [B,M,2048]  (已 detach)
  → cat e_rl [B,1,2048]          # 无输入投影
  → TransformerEncoder × 2, 8 heads, FFN=8192, norm_first
  → src_key_padding_mask = ~pad_mask
  → z_rl = out[:, -1]            # [B, 2048]
```

- \(z_{\mathrm{rl}}\) **与 VLA 同宽**，最接近论文图示的 1×2048。
- 无位置编码：token 顺序信息只存在于 \(z\) 自身（PaliGemma RoPE 已经写进内容）。Encoder 对 token 排列近似置换等变，但 \(e_{\mathrm{rl}}\) 永远在最后，读出位置固定。
- Stage 2 的 RL 状态是 `x = cat(z_rl, s^p)`，`state_dim = 2048 + action_dim`。

### 2.4 是否「最后一个位置当 \(z_{\mathrm{rl}}\)」

三仓都是 **是**。没有 mean-pool、没有单独的 attention pooling 头。论文公式下标 \(_{M+1}\) 就是这个实现。

---

## 3. Decoder + loss：teacher forcing、causal、MSE、stop-gradient

这是三仓分叉最大的地方。论文要求：**Decoder 输入真实前缀 \(\bar z_{1:i-1}\)**（teacher forcing），**causal mask**，一次前向并行出 \(\hat z_{1:M}\)；梯度 **不得** 经 \(z_i\) 回 VLA。

### 3.1 afengleafs — 最接近 Eq.2

`reconstruction_loss`：

1. `z_bar = z.detach()`，再 subsample。
2. `z_rl = encode(z_bar, mask)`（encoder 也吃 stop-grad 后的 \(z\)）。
3. Decoder 输入：`[z_rl, dec_in_proj(z_bar[:, :-1])]`，长度 \(M\)，加 `dec_pos`。
4. `TransformerEncoder` + causal mask（实现成 decoder-only，没有 cross-attn）。
5. `out_proj: Linear(d, 960)` = 论文 \(h_\phi\)。
6. `err = (pred - z_bar).pow(2).mean(dim=-1)`，再按 `mask` 做加权平均。

对应关系：

| 位置 | 输入 | 预测目标 |
| --- | --- | --- |
| 0 | \(z_{\mathrm{rl}}\) | \(\bar z_1\) |
| 1 | \(\bar z_1\) | \(\bar z_2\) |
| … | … | … |
| \(M-1\) | \(\bar z_{M-1}\) | \(\bar z_M\) |

与知识库「第一个是 rltoken，前 i 个是真实 embedding」一致。推理时 **不用 decoder**（`rl_token()` 只 encode）。

Loss 是 **mean over D、再 mean over 有效 token**，不是论文的 \(\sum_i\|\cdot\|_2^2\)。\(M,D\) 固定时差一个常数，被 lr 吸收；变长 / mask 时与「对 token 求和」不完全等价。

`alpha>0` 时：抽取仍 `@torch.no_grad()`，`L_ro` **永远不回 VLA**；另一次 `policy.forward(batch)` 算 flow-matching `L_vla`，`loss = L_ro + alpha * L_vla`。两条计算图分离，符合 Algorithm 1 的 \(\mathrm{sg}\)。

### 3.2 Rajat — 非自回归 query decoder（偏离论文）

`RLTokenDecoder`：

- 不把 \(\bar z_{1:M-1}\) 喂进去。
- `vlm_queries`、`expert_queries` 都是 **`(1, 1, d)` 一个向量**，`expand` 到目标长度。所有 VLM 位置共享同一 query，所有 expert 位置共享另一 query。
- `memory = Linear(z_rl).unsqueeze(1)`，标准 `TransformerDecoder` **非因果** cross-attn。
- 注释写 “Add positional information via simple learned position encoding”，**代码没有 pos embedding**。
- 输出两路 `Linear` 回到 576 / 432。

无 teacher forcing、无 causal mask。更关键：相同 query + 同一 memory + 无位置差 → **同一路所有位置的 decoder 输出相同**（置换等变 + 输入恒等）。重建目标实质上塌缩成「预测 VLM embedding 的均值」和「预测 expert embedding 的均值」。\(z_{\mathrm{rl}}\) 不必保存序列结构。这与 Eq.2 的信息瓶颈训练目标不一致。

Loss：`F.mse_loss(vlm_recon, vlm_target) + F.mse_loss(expert_recon, expert_target)`，两路等权平均，不按 token 数加权。`L_vlm` 与 `L_expert` 因 `L_prefix`（~100+）和 `chunk_size=50` 不同，隐式偏向更长的那一路。

Stop-grad：`forward_rlt_training` 与 Stage 1 脚本都 `.detach()`；VLA 参数全冻。**冻得住**。缺的是论文形态的 decoder，不是 freeze。

### 3.3 rlt-openpi — teacher forcing + 额外 cross-attn

`RLTokenDecoder`：

```
tgt = cat(z_rl.unsqueeze(1), z[:, :-1])   # [B, M, 2048]
memory = z_rl.unsqueeze(1)               # [B, 1, 2048]
TransformerDecoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=~pad)
z_hat = h_phi(out)                     # Linear(2048, 2048)
```

- Teacher forcing、causal mask、\(h_\phi\) 都在。
- 比论文多一路 **对 \(z_{\mathrm{rl}}\) 的 cross-attention**（self-attn 里位置 0 已经是 \(z_{\mathrm{rl}}\)，这条 memory 是冗余的，但不破坏因果）。
- 无 decoder 输入投影、无位置编码。
- Masked MSE：`(z_hat-z).pow(2).mean(-1)` × `pad_mask`，再 `/ num_valid`。

`RLTokenModel.forward` 开头 `z = z.detach()`。测试 `test_model_stop_gradient` 断言：即使 `z.requires_grad=True`，`loss.backward()` 后 `z.grad is None`。

Joint 模式：`L = L_ro(φ) + α L_vla(θ)`，两个 optimizer；`L_ro` 图上的 \(z\) 已 detach，**不会**通过重建去改 VLA。α 只打开 `L_vla` 那条图。

### 3.4 「stop-gradient 是否真的冻住 VLA」

| 机制 | afengleafs | Rajat | rlt-openpi |
| --- | --- | --- | --- |
| 抽取无梯度 | `@torch.no_grad()` | `@torch.no_grad()` | freeze 时 `@torch.no_grad()`；joint 时 hook 里 `.detach()` |
| 重建目标 detach | `z_bar = z.detach()` | `.detach()` 两次 | `z = z.detach()` |
| VLA `requires_grad` | `alpha==0` 时 False | 始终 False | `alpha==0` freeze；`alpha>0` unfreeze **仅对 L_vla** |
| 结论 | 默认真正冻住 | 真正冻住 | 默认真正冻住；joint 时 VLA **只被 flow-matching 更新** |

没有一仓让 \(\mathcal L_{\mathrm{ro}}\) 回流进 VLA。这与知识库「在 rltoken 的 transformer 看来，vla 是冻结的」一致。

---

## 4. 训练循环：数据、优化器、checkpoint

### 4.1 数据：cache 还是在线 forward

**三个仓库都没有磁盘 embedding cache。** 每一 training step 都对当前 batch 的观察（Rajat 还要 GT action）做一次 VLA forward。

本仓库计划（`SmolVLA_RLT_plan.md`「数据 B」）希望：

> 先把 \(z_{1:M}\) 存成 embedding cache，Stage 1 不再反复 forward 大 VLA。

三仓都没做。代价：Stage 1 的时间/显存几乎全在 VLA 上；好处：不用维护 cache 与 checkpoint、preprocessor、图像增强的一致性。

若本仓库要做 cache，需要 **新设计文件格式**（三仓无可抄）。建议最小字段：

```
sample_id / episode_index / frame_index
z: float16/float32  [M, D]     # 已选好的 token（例如 image-only 192×960）
pad_mask: bool      [M]
# 可选：task string、相机数、smolvla revision、image_only 标志
```

不要存 HuggingFace `past_key_values`：那是推理 KV，按层/头/dtype 绑死实现，不能当训练 cache。

afengleafs 的「cache」只是 **同一次 prefix 前向的 KV**，给采样 \(\tilde a\) 用，生命周期是一次 `extract()` 调用。

### 4.2 数据管线细节

**afengleafs**

- `LeRobotDataset` + `delta_timestamps`（obs 当前帧，action 按 `action_delta_indices` 取 chunk）。
- `make_smolvla_pre_post_processors`：图像、state pad、语言 tokenize，与 policy 训练一致。
- 无限循环 DataLoader，`drop_last=True`。
- 默认数据集 README 写 `lerobot/libero_10`（验证管线）；正式应对齐本体/任务。

**Rajat**

- 裸 `LeRobotDataset(repo_id)`，无 processor、无 delta_timestamps。
- 依赖 dataset 里已经有 SmolVLA 需要的键；语言 token 尤其可疑。
- 需要 `prepare_action` → 必须有动作 chunk。

**rlt-openpi**

- 完全交给 OpenPI：`create_torch_dataset` → `transform_dataset`（Normalize、Resize、TokenizePrompt、Pad）。
- 无限 iterator 产出 `(Observation, actions)`。
- freeze 模式 step **只用 Observation**；joint 才用 `actions`。
- `repo_id` + `HF_LEROBOT_HOME` 定位本地集。

### 4.3 Optimizer、lr、batch、steps

| | afengleafs | Rajat | rlt-openpi |
| --- | --- | --- | --- |
| Opt | AdamW(enc+dec；α>0 再加 VLA 参数) | AdamW(enc+dec) | AdamW(RLT)；joint 时 **第二个** AdamW(VLA, lr=1e-5) |
| lr | 1e-4 | 1e-4 | 1e-4（VLA 1e-5） |
| wd | 0.01 | 1e-5 | 1e-5 |
| schedule | 无 | 200 warmup + cosine 到 0 | 500 linear warmup → 恒定 |
| clip | 1.0 | 1.0 | 1.0 |
| batch | CLI 16（config 默认 8） | 16 | 32 |
| steps | 5000（论文区间 2k–10k） | 5000 | 5000 |
| log / save | 20 / 500 | 50 / 500 | wandb 每步；save 1000 |
| 其它 | 无 resume | 无 resume（有 opt state 但未实现 load） | `--resume-checkpoint` |

Rajat 的 cosine 把 lr 退火到 0；另两仓后期仍是峰值 lr。这会影响 5000 步末的重建质量，复现时不要混用。

### 4.4 Checkpoint 里有什么、Stage 2 读什么

**afengleafs** `outputs/rl_token/rl_token.pt`（覆盖写，无 step 文件名）：

```python
{"rl_token": state_dict,   # Encoder+Decoder+投影+e_rl+pos
 "config": vars(cfg),
 "step": step}
# 若 alpha>0：另存 smolvla_sft.pt = policy.state_dict()
```

**不存 optimizer**。Stage 2 `load_rl_token` 只恢复 `RLTokenModule`，`eval + requires_grad_(False)`。Decoder 权重在文件里但 online 不用。

**Rajat** `checkpoints/rlt_stage1/checkpoint_step{N}.pt` 与 `best_checkpoint.pt`：

```python
{"step", "encoder_state_dict", "decoder_state_dict",
 "optimizer_state_dict", "config": rlt_config.__dict__, "loss"}
```

Stage 2 / 推理 **只 `load encoder_state_dict`**，decoder 丢弃。`best` 按 **当前 step 的 batch loss** 更新，不是滑动平均，噪声大。

**rlt-openpi** `checkpoints/rl_token/<run_name>/rl_token_step{N}.pt`：

```python
{"model", "optimizer", "scheduler", "step", "config": RLTokenTrainConfig,
 # joint 才有：
 "vla_model", "vla_optimizer", "vla_scheduler"}
```

`load_rl_token_model` 从 `config` 重建 `RLTokenModel` 再 `load_state_dict(ckpt["model"])`。Stage 2 冻整模（含 decoder），只用 `.encode()`。Joint 时 Stage 2 应加载同文件里的 `vla_model`，否则会拿到未微调的 OpenPI 权重。

---

## 5. 和本仓库知识库的差异（需要记住的）

1. **磁盘 cache**：计划有，三仓都没有。若 Stage 1 要在 PegInsertion 上反复扫 demo，cache 仍值得做，格式需自定（见 §4.1）。
2. **Decoder**：计划 / 知识库是 teacher forcing + causal。afengleafs、rlt-openpi 做到了；Rajat **没有**，不要当参考实现。
3. **Loss 归约**：计划示例写 `.pow(2).mean()`；论文是对 token **求和**。afengleafs / rlt-openpi 用 masked mean。实现时选一种并在 log 里注明，避免和论文数字横比。
4. **Hidden 维**：对本仓库 SmolVLA，**D=960 不是 576**。Rajat README/config 的 576/432 会误导。Expert 宽是 720，但 **不建议** 把 noisy suffix 放进 \(z\)。
5. **图像 token 数**：connector 后每相机 **64**（不是 PaliGemma 的 256）。3 相机 image-only：`[B, 192, 960]`。
6. **KV cache**：只有 afengleafs 在「一次 prefix 同时服务 \(z_{\mathrm{rl}}\) 和 \(\tilde a\)」上做成了；这是 Stage 2 吞吐关键，与 Stage 1 的 embedding 文件 cache 是两件事。
7. **Val loss**：三仓 Stage 1 都只记 train reconstruction loss。本仓库按 episode holdout（默认 90/10）另记 `val_loss_ro`，公式与 `loss_ro` 相同。

---

## 6. 对本仓库 Stage 1 的可执行结论

若目标是「论文 Eq.1–2 + 本地 SmolVLA 0.4.4」：

1. **抽取**：抄 afengleafs 的 prefix-only + 最终层 + 默认 image-only。`extract_embeddings` 不要走 Rajat 那条 noisy suffix。
2. **Encoder**：`Linear(960 → d_model)` + `e_rl` 末尾 + learnable pos + 最后一位；`d_model` 用 512 或 960，层数 2、头 8 即可（与 afengleafs / rlt-openpi 一致）。Rajat 的 4 层 256 维是另一套超参。
3. **Decoder / loss**：必须 causal teacher forcing + `h_φ` + `z.detach()`。Decoder 用 EncoderLayer+mask（afengleafs）或 TransformerDecoder（rlt-openpi）都可以；不要用 Rajat 的共享 query。
4. **冻 VLA**：抽取 `no_grad` 或 `detach` 二选一即可；不要让 `L_ro` 进 VLA。联合 SFT 另开 `α L_vla`。
5. **循环**：先可以像三仓一样在线 forward；若 VLA 太慢再加 cache。Checkpoint 至少存 encoder（+ config + step）；decoder 可同文件，Stage 2 不加载。本仓库另按 episode holdout 记 `val_loss_ro`（三仓都没有）。
6. **不要** 依赖 Rajat 的 576 维 config 和 Stage 1 裸 Dataset（缺 preprocessor）。

关键文件（便于回查）：

- `/root/autodl-tmp/smollvla_rltoken/rlt/rl_token.py`
- `/root/autodl-tmp/smollvla_rltoken/rlt/train_rl_token.py`
- `/root/autodl-tmp/RL-Token-SmolVLA/smolvla_rlt/modeling_smolvla_rlt.py`
- `/root/autodl-tmp/RL-Token-SmolVLA/lerobot_patches/modeling_smolvla.patch`
- `/root/autodl-tmp/rlt-openpi/src/rlt_openpi/vla/embedding_extractor.py`
- `/root/autodl-tmp/rlt-openpi/src/rlt_openpi/models/rl_token.py`
- `/root/autodl-tmp/rlt-openpi/src/rlt_openpi/training/rl_token_trainer.py`
