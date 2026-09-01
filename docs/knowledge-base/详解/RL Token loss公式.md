## RL Token 的重建损失

RL Token 表征学习使用的 reconstruction loss 为：

$$
\mathcal L_{\mathrm{ro}}
=
\mathbb E_{\mathcal D}
\left[
\sum_{i=1}^{M}
\left\|
h_\phi
\left(
d_\phi
\left(
[z_{\mathrm{rl}},\bar z_{1:i-1}]
\right)
\right)_i
-
\bar z_i
\right\|_2^2
\right].
$$

这个损失的核心目标是：

> 使用 RL Token $z_{\mathrm{rl}}$ 和前面的真实 VLA embeddings，预测第 $i$ 个 VLA embedding，并计算预测结果与真实 embedding 之间的误差。

整个过程可以理解为：

$$
\text{VLA embeddings}
\rightarrow
z_{\mathrm{rl}}
\rightarrow
\text{Decoder 重建 VLA embeddings}
\rightarrow
\text{Reconstruction Loss}.
$$

---

### 1. $\mathcal L_{\mathrm{ro}}$

$$
\mathcal L_{\mathrm{ro}}
$$

表示 **RL Token representation objective**，也就是 RL Token 的重建损失。

它不是强化学习中的 reward loss、Q loss 或 actor loss。

这个阶段还没有使用 reward。

它的作用是训练 RL Token Encoder，让生成的 $z_{\mathrm{rl}}$ 尽可能保留原始 VLA embeddings 中的信息。

---

### 2. $\mathcal D$

$$
\mathcal D
$$

表示用于训练 RL Token 的 demonstration dataset。

对于数据集中的每一个 observation：

$$
o_t
$$

首先经过 VLA，获得一系列 hidden embeddings：

$$
z_1,z_2,\ldots,z_M.
$$

然后再利用这些 embeddings 训练 RL Token Encoder 和 Decoder。

---

### 3. $\mathbb E_{\mathcal D}$

$$
\mathbb E_{\mathcal D}[\cdot]
$$

表示对训练数据集 $\mathcal D$ 中的样本取期望。

实际代码训练时一般就是：

$$
\text{一个 batch 中所有样本 loss 的平均值}.
$$

因此可以简单理解为：

> 先计算每一个训练样本的 reconstruction loss，再对 batch 中的样本取平均。

---

### 4. $M$

$$
M
$$

表示 VLA 输出的 embedding/token 数量。

例如 VLA 最后一层得到：

$$
z_1,z_2,z_3,\ldots,z_M.
$$

这里的 $M$ 就是这一整串 VLA embeddings 的长度。

---

### 5. $i$

$$
i=1,2,\ldots,M
$$

表示当前正在预测第几个 VLA embedding。

例如：

- $i=1$：预测 $z_1$
- $i=2$：预测 $z_2$
- $i=3$：预测 $z_3$
- ...
- $i=M$：预测 $z_M$

---

### 6. $z_{\mathrm{rl}}$

$$
z_{\mathrm{rl}}
$$

是 RL Token Encoder 输出的 RL Token。

它由整串 VLA embeddings 压缩得到：

$$
z_{\mathrm{rl}}
=
g_\phi
\left(
[z_{1:M},e_{\mathrm{rl}}]
\right)_{M+1}.
$$

其中 $g_\phi$ 是 RL Token Encoder。

因此：

$$
z_1,\ldots,z_M
\rightarrow
g_\phi
\rightarrow
z_{\mathrm{rl}}.
$$

$z_{\mathrm{rl}}$ 是希望最终保留下来的紧凑状态表示。

---

### 7. $\bar z_i$

公式中并没有直接写 $z_i$，而是：

$$
\bar z_i.
$$

论文定义：

$$
\bar z_i=\operatorname{sg}(z_i),
$$

其中：

$$
\operatorname{sg}
$$

表示 **stop gradient**。

因此：

$$
\bar z_i
$$

在数值上与 $z_i$ 完全相同，但是反向传播时梯度不会通过 $\bar z_i$ 更新原始 VLA。

可以理解为：

$$
z_i
\rightarrow
\operatorname{stop-gradient}
\rightarrow
\bar z_i.
$$

因此 reconstruction loss 主要用于训练 RL Token Encoder 和 Decoder，而不是通过这个 loss 更新 VLA。

---

### 8. $\bar z_{1:i-1}$

$$
\bar z_{1:i-1}
$$

表示第 $i$ 个 embedding 之前的所有**真实 VLA embeddings**：

$$
\bar z_{1:i-1}
=
[\bar z_1,\bar z_2,\ldots,\bar z_{i-1}].
$$

需要特别注意：

这里使用的是 **真实 embedding（ground-truth embedding）**，而不是 Decoder 前面预测出来的 embedding。

例如：

#### 预测 $z_1$

输入：

$$
[z_{\mathrm{rl}}]
$$

目标：

$$
\bar z_1.
$$

#### 预测 $z_2$

输入：

$$
[z_{\mathrm{rl}},\bar z_1]
$$

目标：

$$
\bar z_2.
$$

#### 预测 $z_3$

输入：

$$
[z_{\mathrm{rl}},\bar z_1,\bar z_2]
$$

目标：

$$
\bar z_3.
$$

因此整体上是：

$$
z_{\mathrm{rl}}
\rightarrow
\hat z_1,
$$

$$
[z_{\mathrm{rl}},\bar z_1]
\rightarrow
\hat z_2,
$$

$$
[z_{\mathrm{rl}},\bar z_1,\bar z_2]
\rightarrow
\hat z_3,
$$

一直到：

$$
[z_{\mathrm{rl}},\bar z_1,\ldots,\bar z_{M-1}]
\rightarrow
\hat z_M.
$$

这属于 **Teacher Forcing 的自回归训练方式**。

---

### 9. $[z_{\mathrm{rl}},\bar z_{1:i-1}]$

方括号：

$$
[z_{\mathrm{rl}},\bar z_{1:i-1}]
$$

表示把 RL Token 放在整个 Decoder 输入序列的最前面。

也就是：

$$
[z_{\mathrm{rl}},
\bar z_1,
\bar z_2,
\ldots,
\bar z_{i-1}].
$$

因此 RL Token 相当于整个 reconstruction sequence 的第一个条件 token。

---

### 10. $d_\phi(\cdot)$

$$
d_\phi
$$

表示 RL Token reconstruction 使用的 **Transformer Decoder**。

它的输入是：

$$
[z_{\mathrm{rl}},\bar z_{1:i-1}],
$$

然后通过 causal self-attention 产生 decoder hidden states。

可以写成：

$$
[z_{\mathrm{rl}},\bar z_1,\ldots,\bar z_{i-1}]
\xrightarrow{d_\phi}
[y_1,y_2,\ldots,y_i].
$$

这里的 $y_i$ 是 Decoder 在第 $i$ 个位置产生的 hidden representation。

---

### 11. 下标 $(\cdot)_i$

公式中的：

$$
\left(
d_\phi([z_{\mathrm{rl}},\bar z_{1:i-1}])
\right)_i
$$

表示：

> 取 Transformer Decoder 输出序列中的第 $i$ 个位置。

例如：

$$
d_\phi([z_{\mathrm{rl}},\bar z_1,\bar z_2])
=
[y_1,y_2,y_3].
$$

如果当前 $i=3$，就取：

$$
y_3.
$$

这个 hidden state 用来预测：

$$
z_3.
$$

---

### 12. $h_\phi(\cdot)$

$$
h_\phi
$$

表示一个 **linear output projection（线性输出投影层）**。

Decoder 输出的是内部 hidden state：

$$
y_i.
$$

然后通过：

$$
h_\phi(y_i)
$$

映射回 VLA embedding 所在的空间。

因此预测出来的第 $i$ 个 embedding 为：

$$
\hat z_i
=
h_\phi
\left(
d_\phi
\left(
[z_{\mathrm{rl}},\bar z_{1:i-1}]
\right)
\right)_i.
$$

所以可以把原公式简化为：

$$
\hat z_i
=
h_\phi
\left(
d_\phi
\left(
[z_{\mathrm{rl}},\bar z_{1:i-1}]
\right)
\right)_i.
$$

---

### 13. $\bar z_i$

$$
\bar z_i
$$

是 Decoder 当前需要预测的 **真实目标 embedding**。

因此：

$$
\hat z_i
$$

是预测值，

而：

$$
\bar z_i
$$

是真实值。

---

### 14. $\hat z_i-\bar z_i$

于是公式中的：

$$
h_\phi
\left(
d_\phi
\left(
[z_{\mathrm{rl}},\bar z_{1:i-1}]
\right)
\right)_i
-
\bar z_i
$$

其实就是：

$$
\hat z_i-\bar z_i.
$$

也就是：

> Decoder 预测出来的第 $i$ 个 VLA embedding 与真实第 $i$ 个 VLA embedding 之间的误差。

---

### 15. $\|\cdot\|_2^2$

$$
\left\|
\hat z_i-\bar z_i
\right\|_2^2
$$

表示预测 embedding 和真实 embedding 之间的 **平方 L2 距离**。

假设：

$$
z_i=
[z_{i,1},z_{i,2},\ldots,z_{i,D}]
$$

是一个 $D$ 维 embedding，那么：

$$
\|\hat z_i-\bar z_i\|_2^2
=
\sum_{j=1}^{D}
(\hat z_{i,j}-\bar z_{i,j})^2.
$$

预测越准确：

$$
\|\hat z_i-\bar z_i\|_2^2
\rightarrow 0.
$$

预测越差，loss 越大。

---

### 16. $\sum_{i=1}^{M}$

因为一共有：

$$
M
$$

个 VLA embeddings，所以每一个 embedding 都需要计算 reconstruction error：

$$
L_1
=
\|\hat z_1-\bar z_1\|_2^2,
$$

$$
L_2
=
\|\hat z_2-\bar z_2\|_2^2,
$$

一直到：

$$
L_M
=
\|\hat z_M-\bar z_M\|_2^2.
$$

最后全部相加：

$$
L_{\mathrm{sample}}
=
\sum_{i=1}^{M}
\|\hat z_i-\bar z_i\|_2^2.
$$

因此一个 observation 的 loss 就是整串 VLA embeddings 的重建误差之和。

---

## 整个公式可以简化理解为

首先定义 Decoder 对第 $i$ 个 embedding 的预测：

$$
\hat z_i
=
h_\phi
\left(
d_\phi
\left(
[z_{\mathrm{rl}},\bar z_{1:i-1}]
\right)
\right)_i.
$$

那么原来的复杂公式就可以直接写成：

$$
\mathcal L_{\mathrm{ro}}
=
\mathbb E_{\mathcal D}
\left[
\sum_{i=1}^{M}
\|\hat z_i-\bar z_i\|_2^2
\right].
$$

也就是说：

> 对 demonstration dataset 中的每一个样本，让 Decoder 使用 $z_{\mathrm{rl}}$ 和前面的真实 VLA embeddings 预测下一个 VLA embedding；计算所有 $M$ 个 embedding 的平方 L2 重建误差，再对数据集中的样本取平均。

---

## 训练时的实际形式

虽然数学公式写成：

$$
i=1,2,\ldots,M
$$

逐个预测，但训练时并不需要真的运行 $M$ 次 Decoder。

因为使用的是 Teacher Forcing，所以可以一次性输入：

$$
[z_{\mathrm{rl}},\bar z_1,\bar z_2,\ldots,\bar z_{M-1}],
$$

然后利用 causal attention mask，一次 Transformer forward 同时得到：

$$
[\hat z_1,\hat z_2,\ldots,\hat z_M].
$$

因此可以表示为：

$$
\begin{array}{cccccc}
\text{Input:}
&
z_{\mathrm{rl}}
&
\bar z_1
&
\bar z_2
&
\cdots
&
\bar z_{M-1}
\\
&
\downarrow
&
\downarrow
&
\downarrow
&
&
\downarrow
\\
\text{Target:}
&
\bar z_1
&
\bar z_2
&
\bar z_3
&
\cdots
&
\bar z_M
\end{array}
$$

所以：

- 自回归关系仍然存在；
- 每个位置只能看到前面的 token；
- 训练时使用真实的前序 embedding，而不是预测 embedding；
- 因此所有位置可以在一次 Transformer forward 中并行计算。

---

## Loss 最终更新哪些模块？

Reconstruction loss 的梯度路径大致为：

$$
\mathcal L_{\mathrm{ro}}
\rightarrow
h_\phi
\rightarrow
d_\phi
\rightarrow
z_{\mathrm{rl}}
\rightarrow
g_\phi.
$$

因此它会训练：

- RL Token Encoder $g_\phi$
- RL Token embedding $e_{\mathrm{rl}}$
- Reconstruction Decoder $d_\phi$
- Output projection $h_\phi$

但是因为：

$$
\bar z_i=\operatorname{sg}(z_i),
$$

VLA embeddings 在 reconstruction objective 中被 stop-gradient，因此这个 loss 不通过 $z_i$ 更新 VLA。

---

## 最核心的一句话

RL Token reconstruction loss 可以理解为：

$$
\boxed{
\text{让 }z_{\mathrm{rl}}\text{ 帮助 Decoder 尽可能恢复原始 VLA embeddings}
}
$$

如果 $z_{\mathrm{rl}}$ 丢失了大量重要信息，Decoder 就难以恢复：

$$
z_1,\ldots,z_M,
$$

于是：

$$
\mathcal L_{\mathrm{ro}}
\uparrow.
$$

这个误差会反向传播到 RL Token Encoder：

$$
\mathcal L_{\mathrm{ro}}
\rightarrow
z_{\mathrm{rl}}
\rightarrow
g_\phi,
$$

从而迫使 Encoder 学习一个更有信息量的：

$$
z_{\mathrm{rl}}.
$$

需要特别注意：

> Decoder 是训练 RL Token 表征时使用的辅助模块。训练完成进入真正的 Online RL 后，Decoder 不再需要；保留下来的是 RL Token Encoder，它继续根据 VLA embeddings 产生 $z_{\mathrm{rl}}$，供后续 Actor 和 Critic 使用。

---

## 训练损失与验证损失

公式里的 $\mathbb E_{\mathcal D}$ 在工程上要拆成两份互斥的 demonstration 子集。

- **训练 `loss_ro`**：对 $\mathcal D_{\mathrm{train}}$ 的一个 batch 求 masked mean-MSE，反传更新 encoder/decoder。
- **验证 `val_loss_ro`**：对 $\mathcal D_{\mathrm{val}}$ 用**同一条** reconstruction loss，`eval()` + stop-gradient，**不更新**参数。

划分必须按 **episode**，不能按帧随机切：同一条轨迹里相邻观测高度相关，frame split 会把几乎相同的 $z_{1:M}$ 同时放进 train 和 val，验证数字会虚低。本仓库默认 `val_ratio=0.1`（1000 episode → 900 / 100，`seed=1000`）。

Val 不是仿真成功率，只回答「held-out 观测上重建还好不好」。过拟合时 `loss_ro` 继续降、`val_loss_ro` 走平或回升。中途 val 会 cap batch 数（省冻结 VLA 的 prefix forward）；最后一个 step 跑完整 val split。