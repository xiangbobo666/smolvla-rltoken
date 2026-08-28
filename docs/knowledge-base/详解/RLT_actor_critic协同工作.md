# RLT Actor-Critic 协同工作：Replay Buffer、TD Target 与完整循环

本文说明 RLT 中 VLA、Actor、Critic 与 replay buffer 如何形成完整的 off-policy 训练闭环。为简洁起见，以下写一个 Critic $Q_\psi$ 和它的 target Critic $Q_{\psi'}$；若实现使用 twin critics，可将相关 Q 值替换为两个 Critic 中较保守的估计。

## 1. 三个模块的职责

### 冻结的 VLA：提供动作参考

在状态 $\mathbf x_t$ 下，冻结的 VLA 根据观测和语言任务输出 reference action chunk：

$$
\tilde{\mathbf a}_t
= \pi_{\mathrm{VLA}}(\text{observation}_t, \text{language}).
$$

它提供动作先验，但本身不由 RL loss 更新。

### Actor：提出要执行或要评估的动作

Actor 根据 RL 状态和 VLA reference 产生动作：

$$
\mathbf a_t
\sim
\pi_\theta(
\cdot
\mid \mathbf x_t, \tilde{\mathbf a}_t).
$$

它回答的是：“在当前状态、且以 VLA proposal 为参考时，应该输出哪个 action chunk？”

### Critic：评价一个状态—动作对的长期价值

Critic 估计

$$
Q_\psi(\mathbf x_t, \mathbf a_t),
$$

即从 $\mathbf x_t$ 执行 action chunk $\mathbf a_t$，之后继续依照当前策略行动时的预期折扣回报。

三者的关系是：

```text
VLA:      (observation, language) → reference action ã
Actor:    (RL state x, ã)          → new action a
Critic:   (x, a)                   → value Q
```

Critic 学习“什么动作好”；Actor 利用 Critic 对 action 的梯度，学习“怎样输出这些高价值动作”。

## 2. Chunk-level transition 与 replay buffer

令每个 action chunk 含 $C$ 个控制步。令

$$
\mathbf x_t = (\mathbf z_{\mathrm{rl},t}, \mathbf s^p_t)
$$

为 RL 状态，其中 $\mathbf z_{\mathrm{rl},t}$ 是冻结 VLA 导出的 RL token，$\mathbf s^p_t$ 是 proprioception。

在 rollout 中，机器人位于状态 $A$ 时：

$$
(\mathbf x_A, \tilde{\mathbf a}_A)
\xrightarrow{\pi_\theta}
\mathbf a_A^{\mathrm{exec}}
\xrightarrow{\text{execute }C\text{ steps}}
\mathbf x_B.
$$

这里 $\mathbf a_A^{\mathrm{exec}}$ 是 rollout 当时真正下发给环境并导致 $A\rightarrow B$ 的动作。环境给出 chunk 内 reward

$$
\mathbf r_A = (r_{A,1}, \ldots, r_{A,C}),
$$

以及终止标记 $d_A$。

一条建议保存到 replay buffer 的完整 transition 为

$$
\boxed{
\left(
\mathbf x_A,
\tilde{\mathbf a}_A,
\mathbf a_A^{\mathrm{exec}},
\mathbf r_A,
\mathbf x_B,
\tilde{\mathbf a}_B,
d_A
\right).
}
$$

字段含义如下：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| $\mathbf x_A$ | 状态 A 的 RL token 与 proprioception | Critic 当前项；Actor update 的状态输入 |
| $\tilde{\mathbf a}_A$ | 状态 A 的 VLA | Actor 输入；reference regularization |
| $\mathbf a_A^{\mathrm{exec}}$ | rollout 时旧 Actor 实际执行的 action chunk | Critic 当前项 |
| $\mathbf r_A$ | 这 $C$ 步环境 reward | 构造 TD target |
| $\mathbf x_B$ | 执行后真实到达的下一个状态 | TD target 的 bootstrap 状态 |
| $\tilde{\mathbf a}_B$ | 状态 B 的 VLA reference | 当前 Actor 在 target 中重新生成未来动作 |
| $d_A$ | 环境是否在该 chunk 后终止 | 终止时截断 bootstrap |

保存 $\tilde{\mathbf a}_B$ 很重要：训练时可以直接让当前 Actor 在 $B$ 状态产生动作，无须重新运行大型 VLA。工程上可在得到状态 $B$ 的 VLA 输出后，再把 transition $A\rightarrow B$ 补完整，即“晚一拍”写入 buffer。

chunk 折扣回报为

$$
R_A^{(C)}
= \sum_{i=1}^{C}\gamma^{i-1}r_{A,i}.
$$

如果环境或 replay buffer 已直接提供 chunk reward，也可以直接存储 $R_A^{(C)}$。

## 3. Critic update：学习真实旧动作的价值

从 replay buffer 采样一个 mini-batch：

$$
(\mathbf x_A, \tilde{\mathbf a}_A, \mathbf a_A^{\mathrm{exec}},
\mathbf r_A, \mathbf x_B, \tilde{\mathbf a}_B, d_A).
$$

Critic 当前项必须是

$$
\boxed{
Q_\psi(\mathbf x_A, \mathbf a_A^{\mathrm{exec}}).
}
$$

原因是我们观察到的环境 transition 和 reward，确实由这一个历史动作造成：

$$
\mathbf x_A
\xrightarrow{\mathbf a_A^{\mathrm{exec}}}
\mathbf x_B.
$$

即使当前 Actor 已更新过很多次，也不能把这条历史转移的当前项改成“当前 Actor 在 $A$ 新生成的动作”，因为该新动作没有真实导致 buffer 中的 $\mathbf r_A$ 与 $\mathbf x_B$。

### 3.1 TD target：在下一状态使用当前 Actor 的新动作

先在下一个状态 $B$ 让当前 Actor 重新产生未来动作：

$$
\boxed{
\mathbf a_B^{\mathrm{new}}
\sim
\pi_\theta(
\cdot
\mid \mathbf x_B, \tilde{\mathbf a}_B).
}
$$

然后 target Critic 计算未来 bootstrap 价值：

$$
Q_{\psi'}(\mathbf x_B, \mathbf a_B^{\mathrm{new}}).
$$

因此 TD target 为

$$
\boxed{
y_A
= R_A^{(C)}
+ (1-d_A)\gamma^C
Q_{\psi'}
\left(
\mathbf x_B,
\pi_\theta(\mathbf x_B, \tilde{\mathbf a}_B)
\right).
}
$$

在构造 $y_A$ 时，应停止梯度：Actor 和 target Critic 不接收这次 Critic-loss 反向传播。

这里最易混淆的两个 action 必须严格区分：

$$
\boxed{
\begin{aligned}
&\text{Critic 当前项：}
&&Q_\psi(\mathbf x_A, \underbrace{\mathbf a_A^{\mathrm{exec}}}_{\text{buffer 中真实旧动作}}),\\
&\text{TD target 未来项：}
&&Q_{\psi'}(\mathbf x_B, \underbrace{\mathbf a_B^{\mathrm{new}}}_{\text{当前 Actor 新动作}}).
\end{aligned}
}
$$

之所以未来项使用当前 Actor，是因为我们要估计：“抵达 $B$ 后，从现在开始按当前策略继续执行，未来能获得多少回报？”这正是 off-policy Actor-Critic bootstrap 的含义。

### 3.2 Critic loss 与更新

单 Critic 的平方 TD error 可写为

$$
\boxed{
\mathcal L_Q(\psi)
=
\mathbb E_{\mathcal D}
\left[
\left(
Q_\psi(\mathbf x_A, \mathbf a_A^{\mathrm{exec}})
- y_A
\right)^2
\right].
}
$$

更新 Critic 参数：

$$
\psi
\leftarrow
\psi - \eta_Q\nabla_\psi\mathcal L_Q.
$$

target Critic 通常通过 Polyak averaging 缓慢跟随在线 Critic：

$$
\psi'
\leftarrow
\tau\psi + (1-\tau)\psi',
\qquad 0 < \tau \ll 1.
$$

慢更新的 target Critic 能降低 bootstrap target 随训练剧烈变化而带来的不稳定性。

## 4. Actor update：用当前 Critic 改善当前策略

Actor update 仍从同一类 replay batch 取 $\mathbf x_A$ 与 $\tilde{\mathbf a}_A$，但它不复用旧的 executed action 作为 policy 输出。

当前 Actor 重新生成候选动作：

$$
\mathbf a_A^{\mathrm{new}}
\sim
\pi_\theta(
\cdot
\mid \mathbf x_A, \tilde{\mathbf a}_A).
$$

随后通过当前 Critic 计算 Actor loss：

$$
\boxed{
\mathcal L_\pi(\theta)
=
\mathbb E_{\mathcal D}
\left[
-Q_\psi(\mathbf x_A, \mathbf a_A^{\mathrm{new}})
+ \beta
\left\|
\mathbf a_A^{\mathrm{new}}
- \tilde{\mathbf a}_A
\right\|_2^2
\right].
}
$$

更新规则为

$$
\theta
\leftarrow
\theta - \eta_\pi\nabla_\theta\mathcal L_\pi.
$$

Actor update 阶段，$\psi$ 固定而且不执行 Critic optimizer step；但计算图必须从 $Q_\psi(\mathbf x,\mathbf a)$ 保留到 $\mathbf a$，这样 Critic 才能向 Actor 传递 $\partial Q/\partial\mathbf a$。

## 5. 完整训练循环

完整闭环可按如下顺序理解：

```text
1. Rollout
   VLA 在状态 A 生成 ã_A。
   当前 Actor 根据 (x_A, ã_A) 生成并执行 a_A^exec。
   环境经过 C 步返回 reward、状态 B 和 done。
   VLA 在 B 生成 ã_B；将完整 transition 写入 replay buffer。

2. Critic update
   从 buffer 取 batch。
   用真实旧动作 a_A^exec 计算 Q_ψ(x_A, a_A^exec)。
   用 B 状态和 ã_B，经当前 Actor 得到 a_B^new。
   用 target Critic 构造 TD target y_A，最小化 TD error，更新 ψ。
   软更新 target Critic ψ'。

3. Actor update
   从 buffer 取 (x_A, ã_A)。
   当前 Actor 重新生成 a_A^new。
   用固定 Critic 计算 -Q_ψ(x_A, a_A^new) + β||a_A^new - ã_A||²。
   反向传播到 θ，更新 Actor。

4. 重复
   参数更新后的 Actor 用于后续 rollout，产生更多经验；buffer 中的新旧经验共同支持 off-policy 学习。
```

用公式把 Critic 与 Actor 的核心闭环串起来：

$$
\boxed{
\begin{aligned}
\text{rollout:}\quad
&\mathbf a_A^{\mathrm{exec}}
\sim \pi_\theta(\cdot\mid\mathbf x_A,\tilde{\mathbf a}_A),\\
\text{Critic:}\quad
&Q_\psi(\mathbf x_A,\mathbf a_A^{\mathrm{exec}})
\leftarrow
R_A^{(C)}+(1-d_A)\gamma^C
Q_{\psi'}(\mathbf x_B,\pi_\theta(\mathbf x_B,\tilde{\mathbf a}_B)),\\
\text{Actor:}\quad
&\theta\ \text{is updated to maximize}\ 
Q_\psi(\mathbf x_A,\pi_\theta(\mathbf x_A,\tilde{\mathbf a}_A))
-\beta
\left\|
\pi_\theta(\mathbf x_A,\tilde{\mathbf a}_A)
-\tilde{\mathbf a}_A
\right\|_2^2.
\end{aligned}
}
$$

## 6. 最常见的混淆

### 混淆一：buffer 里的旧动作在哪用？

$\mathbf a_A^{\mathrm{exec}}$ 用在 Critic 当前项，因为它是实际发生的环境转移的动作。它不作为 Actor update 的新输出。

### 混淆二：TD target 的下一动作从哪里来？

它不是 buffer 中状态 $B$ 当时的旧 Actor 动作。正确的未来动作是由当前 Actor 重新计算：

$$
\mathbf a_B^{\mathrm{new}}
= \pi_\theta(\mathbf x_B,\tilde{\mathbf a}_B).
$$

### 混淆三：为什么还要保存 $\tilde{\mathbf a}$？

因为 reference action 同时服务于：

- Actor 的输入；
- Actor loss 中的 $\|\mathbf a-\tilde{\mathbf a}\|_2^2$；
- TD target 中，当前 Actor 在下一状态重新产生动作。

没有保存 $\tilde{\mathbf a}_B$ 时，训练阶段必须重跑 VLA 才能得到这一条件输入。

### 混淆四：Actor 与 Critic 在同一步同时更新吗？

它们通常交替更新，也可以有不同更新频率。Critic 先通过真实经验改善 Q 的估计；Actor 再利用当前 Critic 的梯度改善策略。无论具体调度如何，Actor update 时应固定 Critic 参数，Critic update 时 TD target 应停止梯度。

## 7. 一句话总结

$$
\boxed{
\text{Critic 用 replay 中真实执行过的旧动作学习价值；}
\text{Actor 则在 replay 提供的状态与 VLA reference 上重新产生新动作，}
\text{并在 Critic 引导下、但不远离 VLA 的前提下持续改进。}
}
$$
