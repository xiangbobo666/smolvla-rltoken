# RLT Actor 公式详解：输入、输出、Loss、梯度与实现

本文说明 RLT（以 RL token 表示状态、以 VLA action chunk 为参考动作的 Actor-Critic 设定）中的 Actor：它接收什么、输出什么、为什么这样训练，以及一次实现中参数如何真正被更新。

## 1. Actor 在系统中的位置

冻结的 VLA 先根据当前观测和语言指令给出一个 action chunk 作为建议。Actor 以这个建议为条件，结合 RL 状态输出新的 action chunk；环境真正执行 Actor 的输出。

```text
当前观测、语言、机器人状态
              ↓
         冻结的 VLA
              ↓
 VLA reference action chunk ã_{1:C}
              ↓
     Actor（结合 RL 状态 x）
              ↓
     最终 action chunk a_{1:C}
              ↓
            环境执行
```

因此，Actor 的功能不是从零开始替代 VLA，而是：

> 在 VLA 提供的可靠动作建议附近，利用 Critic 学到的长期回报信息，选择更好的 action chunk。

从数学结构上，Actor 直接生成最终动作，而不是显式预测 residual。它不是预先规定的

$$
\mathbf a = \tilde{\mathbf a} + \Delta\mathbf a.
$$

更准确的写法是

$$
\mathbf a_{1:C}
= \pi_\theta(\mathbf x, \tilde{\mathbf a}_{1:C}).
$$

但由于 reference regularization 会把输出拉回 $\tilde{\mathbf a}_{1:C}$ 附近，其效果可理解为对 VLA 动作做局部编辑。

## 2. 符号与输入

令一个 action chunk 含有 $C$ 个控制步：

$$
\mathbf a_{1:C} = [\mathbf a_1, \ldots, \mathbf a_C].
$$

其中每个 $\mathbf a_i$ 可以是多维机器人控制量，例如末端执行器位姿增量、关节命令和夹爪命令。

RL 状态定义为

$$
\mathbf x = (\mathbf z_{\mathrm{rl}}, \mathbf s^p),
$$

其中：

- $\mathbf z_{\mathrm{rl}}$：从冻结 VLA 的 token embedding 中得到的 RL token 表征；
- $\mathbf s^p$：proprioception，例如关节位置、速度、夹爪状态等；
- $\tilde{\mathbf a}_{1:C}$：VLA 在当前状态下生成的 reference action chunk。

所以 Actor 的完整输入为

$$
\boxed{(\mathbf x, \tilde{\mathbf a}_{1:C})}.
$$

注意 $\tilde{\mathbf a}_{1:C}$ 不是训练标签，而是 Actor 的条件输入，也是后续约束项的参照物。

## 3. Actor 的输出：条件高斯策略

Actor 表示一个条件策略：

$$
\pi_\theta
\left(
\mathbf a_{1:C}
\mid
\mathbf x, \tilde{\mathbf a}_{1:C}
\right)
=
\mathcal N
\left(
\mu_\theta(\mathbf x, \tilde{\mathbf a}_{1:C}),
\sigma^2 \mathbf I
\right).
$$

其中 $\theta$ 是 Actor 的可训练参数，$\mu_\theta$ 是网络给出的 action chunk 均值，$\sigma$ 是固定或可设定的较小探索标准差。

训练或 rollout 时可用重参数化写成

$$
\mathbf a_{1:C}
= \mu_\theta(\mathbf x, \tilde{\mathbf a}_{1:C})
+ \sigma\boldsymbol\epsilon,
\qquad
\boldsymbol\epsilon \sim \mathcal N(\mathbf 0, \mathbf I).
$$

部署时通常使用确定性动作，即

$$
\mathbf a_{1:C}
= \mu_\theta(\mathbf x, \tilde{\mathbf a}_{1:C}).
$$

高斯噪声的作用是让采集数据时在当前策略附近进行小范围探索；Actor 本身的核心学习目标仍是移动均值 $\mu_\theta$。

## 4. Actor 的损失函数

从 replay buffer 采样状态和 reference action 后，当前 Actor 重新生成动作：

$$
\mathbf a^{\mathrm{new}}_{1:C}
\sim
\pi_\theta(
\cdot
\mid \mathbf x, \tilde{\mathbf a}_{1:C}).
$$

Actor loss 为

$$
\boxed{
\mathcal L_\pi(\theta)
=
\mathbb E
\left[
- Q_\psi(\mathbf x, \mathbf a^{\mathrm{new}}_{1:C})
+ \beta
\left\|
\mathbf a^{\mathrm{new}}_{1:C}
- \tilde{\mathbf a}_{1:C}
\right\|_2^2
\right].
}
$$

它由两个相互制衡的目标组成：

$$
\mathcal L_\pi
=
\underbrace{-Q_\psi(\mathbf x, \mathbf a^{\mathrm{new}})}_{\text{提高长期价值}}
+
\underbrace{
\beta
\left\|
\mathbf a^{\mathrm{new}}
- \tilde{\mathbf a}
\right\|_2^2
}_{\text{保持接近 VLA reference}}.
$$

### 4.1 价值项 $-Q$

优化器最小化 $\mathcal L_\pi$，因此最小化 $-Q$ 等价于最大化 $Q$：

$$
\min_\theta[-Q_\psi(\mathbf x, \mathbf a)]
\quad\Longleftrightarrow\quad
\max_\theta Q_\psi(\mathbf x, \mathbf a).
$$

Critic 的含义是“在状态 $\mathbf x$ 执行 action chunk $\mathbf a$，并此后按照当前策略继续，预期的折扣回报是多少”。因此该项促使 Actor 产出被 Critic 评为长期价值更高的动作。

### 4.2 Reference regularization 项

第二项

$$
\beta
\left\|
\mathbf a - \tilde{\mathbf a}
\right\|_2^2
$$

会惩罚远离 VLA proposal 的动作。其作用是把策略搜索限制在 VLA 已经较可信的局部动作区域，减少 Critic 在训练数据覆盖不足区域高估 Q 值时造成的动作漂移。

超参数 $\beta$ 控制两者的权衡：

- $\beta$ 较大时，$\mathbf a \approx \tilde{\mathbf a}$，Actor 更接近复制 VLA；
- $\beta$ 较小时，Actor 更积极地追随 Critic 的价值信号，也更依赖 Critic 估计的可靠性。

## 5. Critic 如何把“动作应如何改变”传给 Actor

Actor 的输出可简写为

$$
\mathbf a = f_\theta(\mathbf x, \tilde{\mathbf a}).
$$

对 action 求导，Actor loss 的局部梯度是

$$
\boxed{
\frac{\partial \mathcal L_\pi}{\partial \mathbf a}
=
-\frac{\partial Q_\psi(\mathbf x, \mathbf a)}{\partial \mathbf a}
+ 2\beta(\mathbf a - \tilde{\mathbf a}).
}
$$

第一项 $-\partial Q/\partial\mathbf a$ 指向能提高 Q 的方向；第二项 $2\beta(\mathbf a-\tilde{\mathbf a})$ 把动作往 VLA reference 拉回。再利用链式法则，可得 Actor 参数的梯度：

$$
\boxed{
\frac{\partial \mathcal L_\pi}{\partial \theta}
=
\left[
-\frac{\partial Q_\psi(\mathbf x, \mathbf a)}{\partial \mathbf a}
+ 2\beta(\mathbf a - \tilde{\mathbf a})
\right]
\frac{\partial \mathbf a}{\partial \theta}.
}
$$

这条公式的意义是：Critic 不直接给 Actor 一个“正确动作标签”，也不直接改环境中的 action；它构建可微的 $Q(\mathbf x, \mathbf a)$ 地形，并通过 $\partial Q/\partial\mathbf a$ 告诉 Actor 在当前动作附近往哪个方向移动会提高长期回报。

梯度路径如下：

```text
Actor parameters θ
       ↓
Actor: a = f_θ(x, ã)
       ↓
Critic: Q_ψ(x, a)
       ↓
Actor loss L_π

反向传播：L_π → Q → a → θ
```

Actor update 时，Critic 参数 $\psi$ 应固定：反向传播可以经过 Critic 来计算 $\partial Q/\partial\mathbf a$，但不对 Critic optimizer 执行更新。

## 6. 参数更新

采用学习率 $\eta_\pi$ 的梯度下降，Actor 参数更新为

$$
\boxed{
\theta
\leftarrow
\theta
- \eta_\pi
\nabla_\theta \mathcal L_\pi.
}
$$

$\theta$ 包含 Actor 网络中所有可训练参数，例如 MLP 或 Transformer 的权重与 bias、注意力层参数、动作头参数等。更新后，下一次即使输入同一个 $(\mathbf x,\tilde{\mathbf a})$，Actor 输出也会发生变化。

## 7. 实现时的一次 Actor update

从 replay buffer 中取出 mini-batch 的 $\mathbf x$ 和 $\tilde{\mathbf a}$。不要把 buffer 中的历史 executed action 当作 Actor 的新输出；Actor update 必须使用当前参数 $\theta$ 重新产生动作。

```text
batch: (x, ã, ...)
    ↓
a_new = actor(x, ã)              # 当前 Actor 重新产生动作
q = critic(x, a_new)             # Critic 参数固定，但保留对 a_new 的梯度
reference_loss = mean(||a_new - ã||²)
actor_loss = mean(-q) + β · reference_loss
    ↓
仅更新 Actor 参数 θ
```

对应的伪代码如下：

```python
# critic 的参数不参与本次 optimizer 更新
freeze(critic.parameters())

a_new = actor.sample_or_mean(x, a_ref)
q_new = critic(x, a_new)
loss_actor = (-q_new).mean() + beta * ((a_new - a_ref) ** 2).mean()

actor_optimizer.zero_grad()
loss_actor.backward()
actor_optimizer.step()

unfreeze(critic.parameters())
```

`freeze` 的目的是避免累积或更新 $\psi$ 的梯度；它不能切断从 $Q$ 到 $\mathbf a^{\mathrm{new}}$ 的计算图，否则 Actor 无法得到 $\partial Q/\partial\mathbf a$。

## 8. 一句话总结

$$
\boxed{
\text{Actor 以 }(\mathbf x,\tilde{\mathbf a})\text{ 为条件产生新动作，}
\text{沿 Critic 的 }\partial Q/\partial\mathbf a\text{ 改善长期价值，}
\text{并由 }\beta\|\mathbf a-\tilde{\mathbf a}\|^2\text{ 保持在 VLA 动作附近。}
}
$$
