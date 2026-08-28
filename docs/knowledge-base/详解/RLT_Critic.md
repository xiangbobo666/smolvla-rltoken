# RLT / RL Token：Critic 损失与奖励价值计算详解

> 说明：这是一份面向 VS Code Markdown Preview 的版本。
>
> 为了避免再次出现公式被当作普通文本的问题，本文所有公式都使用 VS Code / KaTeX 更稳定的块级公式格式：单独一行的 `$$` 包住公式。
>
> 本文暂时不讨论 stride 切分，默认一次执行一个长度为 `C` 的 RL action chunk，就形成一条 replay transition。

---

## 1. Critic 到底在预测什么？

Critic 的输出写成：

$$
Q_\psi(x, a_{1:C})
$$

其中：

- `x`：当前 RL 状态；
- `a_{1:C}`：当前真正执行的长度为 `C` 的 action chunk；
- `Q_ψ(x, a_{1:C})`：从状态 `x` 执行这段 action chunk 后，未来累计折扣回报的估计；
- `ψ`：Critic 的参数。

在 RLT 中，RL 状态可以理解为：

$$
x = (z_{\mathrm{rl}}, s^p)
$$

其中：

- `z_rl`：RL token；
- `s^p`：机器人 proprioception。

### 一个重要纠正

`Q = 0.55` 不应该直接解释成“成功概率 55%”。

更准确地说：

> `Q = 0.55` 表示当前 Critic 估计，从这个状态执行这段 action chunk 后，未来能够获得的期望折扣累计回报约为 0.55。

只有在非常特殊的设定下，例如奖励严格是终点二值成功奖励、没有其他奖励、并且不考虑折扣时间差异时，Q 才可能和成功概率接近。

---

## 2. Critic 的损失函数

原始 Critic loss 可以写成：

$$
L_Q
=
\mathbb{E}_{(x,a_{1:C},x')\sim\mathcal{B}}
\left[
\left(
\hat Q
-
Q_\psi(x,a_{1:C})
\right)^2
\right]
$$

这里：

- `B`：Replay Buffer；
- `Q_ψ(x,a_{1:C})`：Online Critic 当前的预测；
- `Q_hat`：为这一条 replay sample 构造出来的 TD target；
- `L_Q`：预测值和 TD target 之间的平方误差。

核心关系就是：

$$
\text{Critic Loss}
=
(\text{TD Target}-\text{Current Q Prediction})^2
$$

例如当前 Critic 预测：

$$
Q_\psi(x,a_{1:C})=0.55
$$

而 TD target 为：

$$
\hat Q=0.90
$$

那么：

$$
L_Q=(0.90-0.55)^2=0.1225
$$

训练时对 `ψ` 反向传播，使当前 Critic 的预测逐渐靠近 TD target。

---

## 3. 最关键的公式：TD target 怎么计算？

Critic 的 target 为：

$$
\hat Q
=
\sum_{t'=1}^C
\gamma^{t'-1}r_{t'}
+
\gamma^C
\mathbb{E}_{a'\sim\pi_\theta}
\left[
Q_{\psi'}(x',a')
\right]
$$

它由两部分组成：

$$
\hat Q
=
\text{当前 chunk 内真实获得的折扣奖励}
+
\text{执行完当前 chunk 后的未来价值}
$$

---

## 4. 第一部分：当前 chunk 内的真实奖励

定义：

$$
R_{\mathrm{chunk}}
=
\sum_{t'=1}^C
\gamma^{t'-1}r_{t'}
$$

如果：

$$
C=5
$$

那么：

$$
R_{\mathrm{chunk}}
=
r_1
+
\gamma r_2
+
\gamma^2 r_3
+
\gamma^3 r_4
+
\gamma^4 r_5
$$

这里的 reward 是环境真实返回的，不是 Critic 预测出来的。

---

## 5. 例子：第 5 步成功

假设一个 chunk 有 5 个 low-level step，并且：

$$
r_1=r_2=r_3=r_4=0
$$

第 5 步插入成功：

$$
r_5=1
$$

再假设：

$$
\gamma=0.99
$$

那么当前 chunk 内的折扣奖励为：

$$
R_{\mathrm{chunk}}
=
0.99^4
\approx
0.960596
$$

如果第 5 步成功后 episode 立刻结束，则不存在后续未来价值。

因此：

$$
\hat Q
=
0.960596
$$

---

## 6. 如果当前 chunk 完全没有成功呢？

假设当前 chunk 内所有 reward 都是 0：

$$
r_1=r_2=\cdots=r_C=0
$$

那么：

$$
R_{\mathrm{chunk}}=0
$$

但只要 episode 没有结束，TD target 仍然不一定是 0。

此时：

$$
\hat Q
=
\gamma^C
\mathbb{E}_{a'\sim\pi_\theta}
\left[
Q_{\psi'}(x',a')
\right]
$$

也就是说：

> 当前 chunk 自己没有拿到 reward，但如果它把机器人带到了一个未来很有希望成功的状态，那么当前 chunk 依然可以具有较高 Q 值。

因此：

$$
r=0
\;\not\Rightarrow\;
Q=0
$$

---

## 7. 未来价值到底怎么计算？

当前状态记为：

$$
x_t
$$

Actor 输出并实际执行一个长度为 `C` 的 action chunk：

$$
a_{t:t+C-1}
$$

执行完 `C` 个 low-level step 后，环境进入下一 RL 决策状态：

$$
x_{t+C}
$$

为了计算当前 sample 的 future value，需要估计：

> 如果从 `x_{t+C}` 开始，按照当前策略继续行动，后面还能得到多大的累计回报？

因此会计算：

$$
Q_{\psi'}(x_{t+C},a')
$$

这里的 `Q_ψ'` 是 Target Critic。

---

## 8. 下一动作 `a'` 不是凭空产生的

RLT 的 Actor 仍然依赖 VLA 给出的 reference action。

到了下一状态 `x_{t+C}` 后，首先根据新的真实观测重新运行 VLA：

$$
\tilde a'_{1:H}
=
\pi_{\mathrm{VLA}}(o')
$$

VLA 可以输出长度为 `H` 的 action horizon。

然后只取前 `C` 步作为 RL Actor 的 reference：

$$
\tilde a'_{1:C}
$$

Actor 再根据下一状态和这份 reference 产生新的 RL action chunk：

$$
a'
\sim
\pi_\theta
\left(
\cdot
\mid
x',
\tilde a'_{1:C}
\right)
$$

最后 Target Critic 评价：

$$
Q_{\psi'}(x',a')
$$

所以 future value 的完整数据流是：

```text
下一状态 x'
    ↓
冻结 VLA
    ↓
新的 VLA action chunk
    ↓
取前 C 步 reference
    ↓
当前 Actor
    ↓
下一段 RL action chunk a'
    ↓
Target Critic
    ↓
Q_ψ'(x', a')
```

---

## 9. 为什么未来价值前面是 `γ^C`？

因为当前一条 RL transition 跨过的不是 1 个 low-level step，而是完整的 `C` 步。

也就是：

$$
x_t
\xrightarrow{a_{t:t+C-1}}
x_{t+C}
$$

所以未来价值要经过 `C` 次折扣：

$$
\gamma^C
Q_{\psi'}(x_{t+C},a')
$$

因此这里是 `γ^C`，而不是单独一个 `γ`。

---

## 10. 数值例子：当前 chunk reward 全是 0，但未来价值很高

假设：

$$
C=5
$$

$$
\gamma=0.99
$$

当前 chunk 内：

$$
r_1=r_2=r_3=r_4=r_5=0
$$

所以：

$$
R_{\mathrm{chunk}}=0
$$

执行完以后，到达下一状态 `x'`。

假设当前 Actor 在 `x'` 下产生下一段动作 `a'`，Target Critic 预测：

$$
Q_{\psi'}(x',a')=0.8
$$

那么：

$$
\hat Q
=
0
+
0.99^5\times0.8
$$

因为：

$$
0.99^5\approx0.95099
$$

所以：

$$
\hat Q\approx0.7608
$$

虽然当前 chunk 没拿到任何 reward，但它仍然有约 0.7608 的 target value。

如果 Online Critic 当前只预测：

$$
Q_\psi(x,a_{1:5})=0.4
$$

那么：

$$
L_Q
=
(0.7608-0.4)^2
\approx
0.1302
$$

于是这次更新会把当前 Q 预测往更高的方向推。

---

## 11. A → B → C → D：成功奖励怎么向前传播？

假设一条 episode 是：

```text
chunk A
  ↓ reward 全 0

chunk B
  ↓ reward 全 0

chunk C
  ↓ reward 全 0

chunk D
  ↓ 第 5 步成功，reward = 1

episode 结束
```

继续假设：

$$
C=5
$$

$$
\gamma=0.99
$$

---

### 11.1 先学习最后的 chunk D

D 的第 5 步成功：

$$
\hat Q_D
=
0.99^4
\approx
0.960596
$$

因为 D 后面已经 terminal：

$$
Q_{\mathrm{future}}=0
$$

所以经过训练后：

$$
Q(D)
\rightarrow
0.960596
$$

---

### 11.2 再学习 chunk C

C 自己的 reward 全为 0：

$$
R_C=0
$$

但 C 之后进入 D 附近的状态。

如果 Target Critic 已经学到：

$$
Q(D)
\approx
0.960596
$$

那么：

$$
\hat Q_C
=
0.99^5
\times
0.960596
$$

因此：

$$
\hat Q_C
\approx
0.913517
$$

所以：

$$
Q(C)
\rightarrow
0.913517
$$

---

### 11.3 再学习 chunk B

同理：

$$
\hat Q_B
=
0.99^5
Q(C)
$$

得到：

$$
\hat Q_B
\approx
0.868746
$$

于是：

$$
Q(B)
\rightarrow
0.868746
$$

---

### 11.4 最后传播到 chunk A

同理：

$$
\hat Q_A
=
0.99^5
Q(B)
$$

得到：

$$
\hat Q_A
\approx
0.826169
$$

于是：

$$
Q(A)
\rightarrow
0.826169
$$

---

## 12. 奖励传播的本质

整个过程可以概括为：

```text
最终成功 reward = 1
        ↓
最后一个 chunk 先学到高 Q
        ↓
前一个 chunk 通过 future Q 得到学习信号
        ↓
继续向更早的 chunk 传播
```

数值上大致是：

$$
1
\rightarrow
Q(D)\approx0.960596
\rightarrow
Q(C)\approx0.913517
\rightarrow
Q(B)\approx0.868746
\rightarrow
Q(A)\approx0.826169
$$

这就是 TD bootstrap。

---

## 13. 为什么一开始前面的 chunk 学不到？

训练刚开始时，Target Critic 可能对所有状态动作都预测接近 0：

$$
Q_{\psi'}(x',a')\approx0
$$

如果当前 chunk reward 又是 0：

$$
R_{\mathrm{chunk}}=0
$$

那么：

$$
\hat Q\approx0
$$

因此前面的 chunk 一开始确实可能学不到成功价值。

只有当 rollout 中真正出现成功 transition 后，最后几个 chunk 才先获得非零 target。

之后通过 Replay Buffer 中的反复采样和 TD bootstrap，成功价值才逐渐向前传播。

所以核心机制是：

$$
\mathrm{Replay\ Buffer}
+
\mathrm{TD\ Bootstrap}
+
\mathrm{Repeated\ Critic\ Updates}
$$

---

## 14. 三种情况必须分清楚

### 情况 1：当前 chunk 成功并终止

如果当前 chunk 获得成功 reward，并且 episode 结束：

$$
\hat Q
=
R_{\mathrm{chunk}}
$$

因为 terminal 后没有 future value。

---

### 情况 2：当前 chunk 没成功，但 episode 继续

如果：

$$
R_{\mathrm{chunk}}=0
$$

但 episode 没结束：

$$
\hat Q
=
\gamma^C
Q_{\psi'}(x',a')
$$

当前 chunk 的价值来自未来。

---

### 情况 3：当前 chunk 没成功，而且失败终止

如果当前 reward 为 0，并且 episode 已终止：

$$
R_{\mathrm{chunk}}=0
$$

同时：

$$
Q_{\mathrm{future}}=0
$$

因此：

$$
\hat Q=0
$$

所以一定要区分：

$$
\mathrm{not\ successful\ yet}
\neq
\mathrm{terminal\ failure}
$$

---

## 15. 实际代码中更方便的 terminal mask 写法

定义：

- `d = 1`：当前 transition 后 episode 结束；
- `d = 0`：episode 继续。

那么 TD target 可以统一写成：

$$
\hat Q
=
\sum_{t'=1}^C
\gamma^{t'-1}r_{t'}
+
(1-d)
\gamma^C
\mathbb{E}_{a'\sim\pi_\theta}
\left[
Q_{\psi'}(x',a')
\right]
$$

如果 `d = 1`：

$$
1-d=0
$$

future value 自动被清零。

如果 `d = 0`：

$$
1-d=1
$$

保留 bootstrap 项。

---

## 16. Online Critic 和 Target Critic 的区别

Online Critic：

$$
Q_\psi
$$

主要负责：

- 对当前 replay action 做预测；
- 计算 loss；
- 通过梯度下降更新参数 `ψ`。

Target Critic：

$$
Q_{\psi'}
$$

主要负责：

- 为下一状态提供相对稳定的 future Q；
- 不直接用当前 Critic loss 快速更新。

常见做法是让 Target Critic 缓慢跟随 Online Critic：

$$
\psi'
\leftarrow
\tau\psi
+
(1-\tau)\psi'
$$

其中 `τ` 是较小的 soft-update 系数。

---

## 17. 不做 stride 切分时，一条 replay sample 保存什么？

第一版实现可以保存：

```text
x_t
vla_ref_chunk_t
executed_action_chunk_t
reward_seq_t
x_next
next_vla_ref_chunk
done
episode_id
chunk_id
```

其中：

- `x_t`：当前 RL 状态；
- `vla_ref_chunk_t`：当前 VLA 输出的前 `C` 步 reference；
- `executed_action_chunk_t`：真正执行的 `C` 步动作；
- `reward_seq_t`：这 `C` 个 low-level step 的 reward 序列；
- `x_next`：执行完这 `C` 步后的下一 RL 状态；
- `next_vla_ref_chunk`：下一状态对应的新 VLA reference；
- `done`：episode 是否结束。

---

## 18. Critic 更新时实际使用哪些数据？

首先从 Replay Buffer 采样一条 transition。

Online Critic 评价 rollout 时真正执行过的动作：

$$
Q_{\mathrm{pred}}
=
Q_\psi(x_t,a_t)
$$

然后计算当前 `C` 步真实折扣奖励：

$$
R_t^{(C)}
=
\sum_{i=0}^{C-1}
\gamma^i r_{t+i}
$$

如果没有 terminal，则利用下一状态和下一份 VLA reference，让当前 Actor 产生下一动作：

$$
a'
\sim
\pi_\theta
\left(
\cdot
\mid
x_{t+C},
\tilde a_{t+C}
\right)
$$

再计算：

$$
Q_{\mathrm{future}}
=
Q_{\psi'}(x_{t+C},a')
$$

构造：

$$
Q_{\mathrm{target}}
=
R_t^{(C)}
+
(1-d)
\gamma^C
Q_{\mathrm{future}}
$$

最后：

$$
L_Q
=
\left(
Q_{\mathrm{target}}
-
Q_{\mathrm{pred}}
\right)^2
$$

然后只更新 Online Critic 参数 `ψ`。

---

## 19. Warmup 阶段有什么不同？

Warmup 阶段 Actor 暂时不负责实际控制。

VLA 输出一个长度为 `H` 的 action chunk：

$$
\tilde a_{1:H}
$$

取前 `C` 步：

$$
\tilde a_{1:C}
$$

直接执行。

因此 warmup 阶段：

$$
a_{1:C}
=
\tilde a_{1:C}
$$

也就是：

```text
VLA reference 的前 C 步
        =
实际执行的 C 步
```

正式 RL 阶段则通常变成：

```text
VLA reference 的前 C 步
        ↓
      Actor
        ↓
实际执行的 RL action chunk
```

此时一般：

$$
a_{1:C}
\neq
\tilde a_{1:C}
$$

但 Critic 的 Bellman / TD 学习方式不变。

---

## 20. Critic 完整训练流程

```text
从 Replay Buffer 采样
        ↓
拿到：
x_t
真实执行的 action chunk
当前 C 步 reward
x_next
next VLA reference
done
        ↓
Online Critic
        ↓
Q_pred = Q_ψ(x_t, action_t)
        ↓
计算当前 chunk 真实折扣奖励 R_chunk
        ↓
如果没有 terminal：
    next VLA reference
            ↓
          Actor
            ↓
      next action a'
            ↓
      Target Critic
            ↓
      future Q
        ↓
构造 TD target
        ↓
Q_target = R_chunk + γ^C future_Q
        ↓
Critic loss
        ↓
L_Q = (Q_target - Q_pred)^2
        ↓
反向传播，只更新 Online Critic
```

---

## 21. 最核心的两个公式

带 terminal mask 的 TD target：

$$
\hat Q
=
\sum_{t'=1}^C
\gamma^{t'-1}r_{t'}
+
(1-d)
\gamma^C
\mathbb{E}_{a'\sim\pi_\theta}
\left[
Q_{\psi'}(x',a')
\right]
$$

Critic loss：

$$
L_Q
=
\mathbb{E}_{(x,a_{1:C},x')\sim\mathcal{B}}
\left[
\left(
\hat Q
-
Q_\psi(x,a_{1:C})
\right)^2
\right]
$$

这两个公式就是 Critic 训练的核心：

```text
真实奖励
+
下一状态未来价值
        ↓
      TD target
        ↓
与当前 Critic 预测比较
        ↓
      MSE loss
        ↓
更新 Critic
```
