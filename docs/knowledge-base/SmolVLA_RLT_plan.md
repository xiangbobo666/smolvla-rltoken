# SmolVLA + RL Token (RLT) 复现项目实施计划（完整合并版）

本文档由以下 7 份文档按逻辑顺序合并整理而成：

1. SmolVLA_RLT_full_export_plan.md —— 文档结构说明
2. SmolVLA_RLT_full_plan_01_project_and_data.md —— Part 1 项目定位、整体架构、框架选择与数据方案
3. SmolVLA_RLT_full_plan_02_rl_token.md —— Part 2 RL Token 表征学习
4. SmolVLA_RLT_full_plan_03_rollout_replay.md —— Part 3 Rollout、Chunk Environment 与 Replay Buffer
5. SmolVLA_RLT_full_plan_04_critic.md —— Part 4 Critic 完整实现
6. SmolVLA_RLT_full_plan_05_actor.md —— Part 5 Actor 完整实现
7. SmolVLA_RLT_full_plan_06_training_pipeline.md —— Part 6 训练系统与开发路线

整体逻辑：项目定位与数据（Part 1）→ 状态表示（Part 2）→ 经验收集（Part 3）→ 价值学习（Part 4）→ 策略优化（Part 5）→ 完整训练系统（Part 6）。

--------------------------------------------------------------------------------

## SmolVLA + RL Token 复现项目 Markdown 导出说明

上一版导出的文件存在问题：内容被错误压缩成摘要，没有保留原回答中的完整技术细节。

后续重新生成时将采用以下规则：

-   保留原回答全部章节结构；
-   保留所有公式、流程图文字描述、代码框架、超参数说明；
-   不再压缩成简版；
-   按章节拆分为多个 Markdown 文件，避免单文件长度限制导致截断。

计划文件：

1.  SmolVLA_RLT_full_plan_01_project_and_data.md
    -   项目定位
    -   技术栈选择
    -   ManiSkill / LeRobot / SmolVLA
    -   数据集方案
    -   Stage 0 baseline
2.  SmolVLA_RLT_full_plan_02_rl_token.md
    -   RL Token Encoder
    -   Decoder
    -   Transformer实现
    -   Reconstruction Loss
    -   checkpoint设计
3.  SmolVLA_RLT_full_plan_03_rollout_replay.md
    -   chunk定义
    -   rollout流程
    -   transition设计
    -   replay buffer字段
    -   next state补全
4.  SmolVLA_RLT_full_plan_04_critic.md
    -   Critic输入输出
    -   TD target
    -   Q loss
    -   Twin critic
    -   terminal处理
5.  SmolVLA_RLT_full_plan_05_actor.md
    -   Actor结构
    -   reference action条件
    -   policy loss
    -   reference regularization
    -   gradient flow
6.  SmolVLA_RLT_full_plan_06_training_pipeline.md
    -   Actor-Critic循环
    -   Trainer设计
    -   仓库结构
    -   开发路线
    -   实验规划

--------------------------------------------------------------------------------


## Part 1：项目定位、整体架构、框架选择与数据方案

------------------------------------------------------------------------

## 一、项目定位

本项目目标不是简单复刻某一个开源仓库，而是基于 RL Token
论文思想，构建一个独立、模块化、可开源的 SmolVLA + RL Token
强化学习框架。

项目定位：

> A modular reproduction of RL Token for SmolVLA in simulation, with
> online off-policy chunk-level reinforcement learning.

核心目标：

-   使用 SmolVLA 作为基础 VLA；
-   使用 RL Token 将大型 VLA 的高维 token embedding
    压缩为强化学习状态表示；
-   冻结 VLA 与 RL Token encoder；
-   使用轻量 Actor-Critic 在仿真环境中进行 online off-policy
    reinforcement learning；
-   验证 RL 是否能够在 VLA 已有能力基础上进一步提升精细操作能力。

整体路线：

    ManiSkill
        |
        v
    LeRobot Dataset
        |
        v
    SmolVLA SFT
        |
        v
    VLA Hidden Token Extraction
        |
        v
    RL Token Transformer Training
        |
        v
    Frozen VLA + Frozen RL Token Encoder
        |
        v
    Chunk-level Actor-Critic RL
        |
        v
    Improved Robot Manipulation Policy

------------------------------------------------------------------------

## 二、总体技术栈选择

### 2.1 深度学习框架

选择：

PyTorch

原因：

-   RL Token Transformer 需要自定义；
-   Actor/Critic 需要自定义；
-   Replay Buffer 与 rollout pipeline 需要完全控制；
-   不建议依赖 Stable-Baselines3，因为标准 RL interface 无法表达 VLA
    reference action、action chunk 和 RL token 状态。

------------------------------------------------------------------------

## 三、VLA 框架选择

### LeRobot + SmolVLA

作为 VLA 基础。

原因：

-   SmolVLA 是轻量级 VLA；
-   LeRobot 已经提供：
    -   dataset format
    -   training pipeline
    -   normalization
    -   policy interface

项目结构中：

    SmolVLA
        |
        | provides
        |
        +-- reference action chunk
        +-- hidden token embeddings

RL 部分不修改 SmolVLA 内部。

设计原则：

    VLAAdapter
            |
            +---- SmolVLA
            |
            +---- Future OpenVLA
            |
            +---- Future pi models

通过 adapter 层隔离。

------------------------------------------------------------------------

## 四、仿真环境选择

### ManiSkill

作为第一阶段主要环境。

原因：

-   面向机器人 manipulation；
-   支持 GPU simulation；
-   提供标准 RL benchmark；
-   提供 demonstration；
-   支持 LeRobot 数据转换。

------------------------------------------------------------------------

### 第一阶段任务选择

推荐顺序：

    PegInsertionSide-v1
            |
            v
    PlugCharger-v1

------------------------------------------------------------------------

## 为什么不是 LIBERO 作为第一任务？

LIBERO：

优点：

-   VLA 社区使用广泛；
-   多任务语言条件丰富。

缺点：

-   更偏 manipulation benchmark；
-   与 ManiSkill 的物理控制接口不同；
-   action chunk 与低层控制迁移需要额外适配。

因此：

第一版重点验证：

    RL Token
    +
    Chunk Actor-Critic
    +
    真实 reward propagation

ManiSkill 更适合。

------------------------------------------------------------------------

## 五、数据设计

项目需要三类数据。

------------------------------------------------------------------------

## 数据 A：SmolVLA demonstration 数据

来源：

ManiSkill demonstration。

用途：

1.  SmolVLA SFT；
2.  RL Token representation training。

流程：

    ManiSkill trajectory

            |

            v

    RGB observation
    +
    proprioception
    +
    action
    +
    language instruction

            |

            v

    LeRobot Dataset

------------------------------------------------------------------------

## 数据 B：RL Token representation 数据

不额外采集。

直接使用：

    SmolVLA demonstration

运行：

    observation

          |

          v

    SmolVLA

          |

          v

    VLA token embeddings

保存：

    embedding cache

原因：

RL Token Transformer 训练阶段不应该反复 forward 大 VLA。

------------------------------------------------------------------------

## 数据 C：Online RL replay 数据

来源：

环境 rollout。

流程：

    Frozen SmolVLA

            |

            v

    Actor-Critic interaction

            |

            v

    ManiSkill environment

            |

            v

    Replay Buffer

与 demonstration 不同。

Replay 是强化学习经验。

------------------------------------------------------------------------

## 六、训练阶段划分

整个项目分三个 Stage。

------------------------------------------------------------------------

## Stage 0：SmolVLA baseline

目标：

证明 VLA 本身具有基础能力。

流程：

    ManiSkill demo

            |

            v

    LeRobot Dataset

            |

            v

    SmolVLA SFT

            |

            v

    Evaluation

记录：

-   success rate
-   episode length
-   failure cases

------------------------------------------------------------------------

## Stage 1：RL Token Representation Learning

冻结：

    SmolVLA

训练：

    RL Token Encoder
    +
    Decoder

目标：

学习：

    high dimensional VLA representation

            |

            v

    compact RL state token

------------------------------------------------------------------------

## Stage 2：Online RL

冻结：

    SmolVLA
    RL Token Encoder

训练：

    Actor
    Critic

流程：

    Observation

        |

        v

    SmolVLA

        |

        +------ reference action

        |

        v

    RL Token Encoder

        |

        v

    RL state

        |

        v

    Actor

        |

        v

    Execute chunk

        |

        v

    Replay Buffer

        |

        v

    Critic update

        |

        v

    Actor update

------------------------------------------------------------------------

## 七、项目开发原则

不要：

    一个 train_rlt.py
    包含全部逻辑

应该：

    VLA
    RL Token
    Environment
    Replay
    Actor
    Critic
    Trainer

模块完全分离。

这样：

-   可以单独测试；
-   可以替换 VLA；
-   可以替换环境；
-   可以扩展实验。

--------------------------------------------------------------------------------


## Part 2：RL Token Transformer 表征学习完整实现

------------------------------------------------------------------------

## 一、RL Token 阶段目标

RL Token 阶段的目标不是训练机器人动作策略。

这一阶段解决的问题是：

> 如何把大型 VLA 内部高维 token representation
> 压缩成一个适合强化学习使用的小状态表示。

因此：

输入：

$$ Z_t=\left[z_1,z_2,...,z_M\right] $$

其中：

-   (M)：VLA token 数量
-   ($D_{VLA}$)：VLA embedding dimension

输出：

$$ z_{rl} $$

作为后续 Actor-Critic 的状态。

------------------------------------------------------------------------

## 二、为什么不能直接使用 VLA embedding

SmolVLA 的 hidden representation：

$$ Z \in R^{M \times D_{VLA}} $$

通常：

-   token 数量多；
-   embedding 维度大；
-   包含大量视觉、语言、动作相关信息。

直接输入 Critic：

    VLA embedding
            |
            v
    MLP Critic

存在问题：

1.  状态维度过高；
2.  RL 学习困难；
3.  大量信息对于当前任务无关。

因此需要：

    VLA representation

            |

            v

    compact RL state

            |

            v

    Actor-Critic

------------------------------------------------------------------------

## 三、RL Token Encoder

### 3.1 输入

冻结 SmolVLA：

    observation

        |

        v

    SmolVLA

        |

        v

    token embeddings

得到：

$$ Z=\left[z_1,...,z_M\right] $$

其中：

$$ z_i\in R^{D_{VLA}} $$

------------------------------------------------------------------------

### 3.2 输入投影

由于：

$$ D_{VLA} $$

可能较大。

首先映射：

$$ u_i=W_{in}z_i $$

得到：

$$ u_i\in R^{D_{model}} $$

代码：

``` python
self.input_proj = nn.Linear(
    vla_dim,
    hidden_dim
)
```

------------------------------------------------------------------------

## 四、Learnable RL Token

定义一个可学习 token：

$$ e_{rl} $$

类似 Transformer 中的 CLS token。

拼接：

$$ \left[u_1,u_2,...,u_M,e_{rl}\right] $$

输入 Transformer Encoder。

------------------------------------------------------------------------

## 五、Transformer Encoder

结构：

    VLA tokens

        +

    RL token

        |

        v

    Transformer Encoder

        |

        v

    hidden states

最终：

取最后一个位置：

$$ z_{rl} = g_\phi( \left[u_{1:M},e_{rl}\right] )_{M+1} $$

因此：

RL token 是整个 VLA representation 的摘要。

------------------------------------------------------------------------

## 六、PyTorch 模块设计

建议：

``` python
class RLTokenEncoder(nn.Module):

    def __init__(
        self,
        vla_dim,
        hidden_dim,
        layers,
        heads,
    ):
        ...
```

内部：

    input projection

    learnable rl token

    position embedding

    Transformer encoder

forward:

输入：

``` python
vla_tokens
# [B,M,D_vla]
```

输出：

``` python
z_rl
# [B,D_model]
```

------------------------------------------------------------------------

## 七、Decoder 的作用

非常重要：

Decoder 不是 policy。

Decoder 不参与机器人控制。

它只用于：

> 判断 RL token 是否保留了足够的 VLA 信息。

因此：

Stage 1:

    Encoder

         |

         v

    RL token

         |

         v

    Decoder

         |

         v

    reconstruct VLA embedding

------------------------------------------------------------------------

## 八、Decoder 输入

采用 autoregressive reconstruction。

输入：

$$ \left[z_{rl},z_1,z_2,...,z_{M-1}\right] $$

预测：

$$ \left[z_1,z_2,...,z_M\right] $$

例如：

位置：

    input:
    [z_rl]

    output:
    z1

位置：

    input:
    [z_rl,z1]

    output:
    z2

位置：

    input:
    [z_rl,z1,z2]

    output:
    z3

------------------------------------------------------------------------

## 九、Teacher Forcing

训练阶段：

使用真实 embedding。

不是：

    预测 z1

    ↓

    使用预测 z1

    ↓

    预测 z2

而是：

    真实 z1

    ↓

    真实 z2

    ↓

    真实 z3

原因：

可以并行训练。

Transformer 一次 forward 完成全部位置预测。

------------------------------------------------------------------------

## 十、Causal Mask

Decoder 必须使用 causal attention。

保证：

预测：

$$ z_i $$

时只能看到：

$$ z_{rl},z_1,...,z_{i-1} $$

不能看到：

$$ z_i,z_{i+1} $$

否则模型会直接复制输入。

------------------------------------------------------------------------

## 十一、Reconstruction Loss

目标：

让 decoder 重建：

$$ Z $$

公式：

$$
\mathcal L_{ro} = E_D \left[ \sum_i
\|\hat z_i-\bar z_i\|_2^2\right]
$$

其中：

$$ \bar z_i=stopgrad(z_i) $$

代码：

``` python
target = vla_tokens.detach()

loss = (
    prediction-target
).pow(2).mean()
```

------------------------------------------------------------------------

## 十二、梯度流设计

必须保证：

训练 Stage 1：

更新：

    RL Token Encoder
    Decoder

冻结：

    SmolVLA

梯度：

    Loss

     |

     +---- Decoder

     |

     +---- RL Encoder


    SmolVLA

    gradient = None

------------------------------------------------------------------------

## 十三、Stage 1 checkpoint

训练结束保存：

    checkpoints/

        smolvla.pt

        rl_token_encoder.pt

        decoder.pt

进入 Stage 2：

加载：

    smolvla.pt

    rl_token_encoder.pt

冻结。

不加载：

    decoder

因为：

Decoder 不参与 RL。

------------------------------------------------------------------------

## 十四、建议实现顺序

不要直接训练。

按照：

### Step 1

完成：

    SmolVLA hidden extraction

验证：

shape:

    [B,M,D]

正确。

------------------------------------------------------------------------

### Step 2

实现：

    RLTokenEncoder

验证：

输出：

    [B,D_rl]

------------------------------------------------------------------------

### Step 3

实现：

    Decoder

验证：

causal mask 正确。

------------------------------------------------------------------------

### Step 4

训练：

    reconstruction loss

观察：

loss 是否下降。

------------------------------------------------------------------------

### Step 5

检查梯度：

    SmolVLA grad = None

    Encoder grad != None

    Decoder grad != None

------------------------------------------------------------------------

## 十五、第一版推荐参数

工程初值：

    RL token dimension:

    512


    Transformer layers:

    4


    Attention heads:

    8


    Training steps:

    5000

后续可以做：

    256
    512
    1024

ablation。

------------------------------------------------------------------------

## 十六、完成后的数据流

最终：

    Observation

          |

          v

    Frozen SmolVLA

          |

          v

    VLA token embeddings

          |

          v

    RL Token Encoder

          |

          v

    z_rl

          |

          v

    Actor-Critic

这就是后续 Online RL 阶段唯一需要的状态表示。

--------------------------------------------------------------------------------


## Part 3：Rollout、Chunk Environment 与 Replay Buffer 完整实现

------------------------------------------------------------------------

## 一、Online RL 阶段整体目标

Stage 2 的目标：

在冻结：

    SmolVLA

    RL Token Encoder

的情况下，只训练：

    Actor

    Critic

通过环境交互提升机器人任务成功率。

整体流程：

    Observation

          |

          v

    Frozen SmolVLA

          |

          +----------------+
          |                |
          v                v

    VLA token          Reference action

          |
          v

    RL Token Encoder

          |

          v

    RL state

          |

          v

    Actor

          |

          v

    Action chunk

          |

          v

    Environment

          |

          v

    Reward + Next observation

          |

          v

    Replay Buffer

          |

          v

    Critic update

          |

          v

    Actor update

------------------------------------------------------------------------

## 二、为什么 RL Token 使用 chunk-level decision

VLA 本身不是单步 policy。

SmolVLA 输出：

$$ \tilde a_{1:H} $$

其中：

-   (H)：action horizon

例如：

    未来 50 个动作

但是 RL 不应该每一步修改。

原因：

机器人操作存在：

-   temporal consistency
-   smoothness
-   contact dynamics

如果每个 timestep 都修改：

    step1:
    Actor 修改

    step2:
    Actor 修改

    step3:
    Actor 修改

动作会抖动。

因此：

RLT 使用：

    action chunk

作为 RL decision unit。

------------------------------------------------------------------------

## 三、区分 H 和 C

非常重要。

### VLA horizon

$$ H $$

表示：

VLA 一次预测多少动作。

例如：

    H=50

------------------------------------------------------------------------

### RL chunk length

$$ C $$

表示：

RL 实际优化多少动作。

第一版：

$$ C=10 $$

流程：

    SmolVLA

          |

          v

    50 step action chunk


          |

          v

    取前10步


          |

          v

    Actor 修改


          |

          v

    执行10步

------------------------------------------------------------------------

## 四、为什么第一版不使用 stride

论文后续可能使用：

    stride=2

增加 RL samples。

例如：

    action chunk:

    0-10

    10-20

    20-30

变成：

    0-10

    2-12

    4-14

    ...

但是第一版不做。

原因：

需要首先保证：

    state

    action

    next state

    reward

语义清晰。

因此：

定义：

$$ \text{1 executed chunk = 1 replay transition} $$

------------------------------------------------------------------------

## 五、Chunk Transition 定义

普通 RL transition：

$$ (s_t,a_t,r_t,s_{t+1}) $$

这里扩展：

$$ (x_t,a_t,r_t,x_{t+C}) $$

其中：

状态：

$$ x_t=\left[z_{rl},s_t^p\right] $$

包含：

-   RL Token
-   proprioception

------------------------------------------------------------------------

动作：

不是：

    VLA output

而是：

    environment executed action

即：

$$ a_t^{exec} $$

原因：

环境真实转移：

$$ x_t \xrightarrow{a_t^{exec}} x_{t+C} $$

------------------------------------------------------------------------

## 六、Transition 数据结构设计

推荐：

``` python
@dataclass
class Transition:

    # current state

    z_rl:
        np.ndarray

    proprio:
        np.ndarray


    # VLA reference

    reference_action:
        np.ndarray


    # real executed action

    executed_action:
        np.ndarray


    # reward during chunk

    reward_sequence:
        np.ndarray


    # next state

    next_z_rl:
        np.ndarray

    next_proprio:
        np.ndarray


    next_reference_action:
        np.ndarray


    # episode information

    terminated:
        bool

    truncated:
        bool


    # actual chunk length

    n_steps:
        int


    episode_id:
        int


    chunk_id:
        int
```

------------------------------------------------------------------------

## 七、为什么需要保存 episode_id 和 chunk_id

因为一个 episode：

可能：

    chunk0

    chunk1

    chunk2

    chunk3

    success

Replay Buffer 随机采样：

    chunk2

    chunk0

    chunk3

训练时不依赖顺序。

但是：

调试和分析需要知道：

    这个 transition 属于哪个 episode

    在 episode 哪个阶段

例如：

分析：

    成功前最后一个chunk

    Q value最高

必须依赖这些信息。

------------------------------------------------------------------------

## 八、Rollout Collector 设计

建议独立模块：

    rollout/

        collector.py

        transition_builder.py

------------------------------------------------------------------------

collector负责：

    environment interaction

transition_builder负责：

    把环境结果整理成Replay格式

------------------------------------------------------------------------

## 九、完整 Rollout 流程

### Step 1：reset

``` python
obs = env.reset()
```

------------------------------------------------------------------------

### Step 2：提取 VLA 信息

冻结：

``` python
with torch.no_grad():
```

运行：

    obs

     |

     v

    SmolVLA

     |

     +------ token embeddings

     |

     +------ reference action

得到：

$$ Z_t $$

和：

$$ \tilde a_t $$

------------------------------------------------------------------------

### Step 3：RL Token

输入：

$$ Z_t $$

得到：

$$ z_{rl,t} $$

构造：

$$ x_t $$

------------------------------------------------------------------------

### Step 4：Actor 输出

Actor 输入：

$$ (x_t,\tilde a_t) $$

输出：

$$ a_t $$

------------------------------------------------------------------------

### Step 5：执行 chunk

环境执行：

    C steps

得到：

    next_obs

    reward sequence

    terminated

    truncated

------------------------------------------------------------------------

## 十、为什么 Replay 需要"晚一拍写入"

这是实现中非常关键的一点。

执行：

    state A

    ↓

    Actor

    ↓

    action

    ↓

    environment

    ↓

    state B

此时：

我们知道：

    A

    action

    reward

但是不知道：

    next_z_rl

    next_reference_action

因为它需要：

    state B

    ↓

    SmolVLA

重新计算。

------------------------------------------------------------------------

因此：

不能立即：

``` python
replay.add()
```

------------------------------------------------------------------------

正确：

保存 pending transition：

    pending:

    state A

    action

    reward

到达 B：

    B

    ↓

    SmolVLA

    ↓

    next reference

    ↓

    next RL token

补全：

    (A,a,r,B)

然后：

    replay.add()

------------------------------------------------------------------------

## 十一、为什么训练 Critic 时不重新运行 VLA

如果没有保存：

    next_z_rl

    next_reference_action

那么每次：

critic update

都需要：

    Replay sample

    ↓

    run SmolVLA

    ↓

    calculate next action

问题：

1.  慢；
2.  VLA 冻结但计算量巨大；
3.  不利于大量 RL update。

因此对于本项目的 reference-conditioned Actor 设计：

Replay 必须保存：

    next state representation
    +
    next reference action

原因：Critic 计算 TD target 时，需要估计下一 chunk 的未来价值：

$$ Q(x_{t+C},\pi_\theta(x_{t+C},\tilde a_{t+C})) $$

Actor 输入包含下一状态下 SmolVLA 输出的 reference action。因此 rollout 时需要慢一拍完成 transition：

1. 执行当前 chunk；
2. 到达 next observation；
3. 运行冻结 SmolVLA 得到 next reference action；
4. 运行 RL Token Encoder 得到 next RL Token；
5. 补全 transition 后写入 replay buffer。

这样 critic update 时无需重复 forward SmolVLA。

------------------------------------------------------------------------

## 十二、Warmup 阶段

不能一开始让随机 Actor 控制机器人。

否则：

    random action

    ↓

    failure

    ↓

    reward=0

    ↓

    critic无法学习

------------------------------------------------------------------------

第一阶段：

执行：

$$ a_t^{exec} = \tilde a_t $$

即：

    SmolVLA reference action

            |

            v

    Environment

            |

            v

    Replay Buffer

------------------------------------------------------------------------

目标：

收集：

    VLA behavior dataset

作为 off-policy replay。

------------------------------------------------------------------------

## 十三、Replay Buffer 实现

结构：

``` python
class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):
        self.buffer=[]


    def add(
        self,
        transition
    ):
        ...


    def sample(
        self,
        batch_size
    ):
        ...

```

------------------------------------------------------------------------

sample 返回：

    batch:

    z_rl

    proprio

    reference_action

    executed_action

    reward_sequence

    next_z_rl

    next_proprio

    next_reference_action

    terminated

    n_steps

------------------------------------------------------------------------

## 十四、Action Normalization

非常重要。

不要直接使用：

    robot physical action

例如：

    translation:
    meter

    rotation:
    radian

    gripper:
    percentage

直接计算：

$$ \|a-\tilde a\|^2 $$

会导致尺度不平衡。

统一：

$$ a\in[-1,1] $$

流程：

    SmolVLA action

            |

            v

    normalized space

            |

            v

    Actor/Critic

            |

            v

    environment adapter

            |

            v

    robot controller

------------------------------------------------------------------------

## 十五、Rollout 完成后的数据流

    Episode

    chunk0

    (state0,
     reference0,
     action0,
     reward0,
     state1)


    chunk1

    (state1,
     reference1,
     action1,
     reward1,
     state2)


    chunk2

    (state2,
     reference2,
     action2,
     reward2,
     state3)

Replay 随机采样：

    chunk0

    chunk2

    chunk1

不依赖 episode 顺序。

------------------------------------------------------------------------

## 十六、下一阶段连接 Critic

Replay 提供：

当前：

$$ (x_t,a_t^{exec}) $$

用于：

$$ Q_\psi(x_t,a_t) $$

下一状态：

$$ (x_{t+C},\tilde a_{t+C}) $$

用于：

Actor 生成：

$$ a' $$

计算：

$$ Q(x',a') $$

这就是后续 Critic TD target 的输入来源。

--------------------------------------------------------------------------------


## Part 4：Critic 完整实现 ------ TD Learning、Q Network 与训练流程

------------------------------------------------------------------------

## 一、Critic 在整个 RLT 中的作用

在 Stage 2 Online RL 阶段：

冻结：

    SmolVLA

    RL Token Encoder

训练：

    Actor

    Critic

其中：

Actor 负责：

> 产生更优的 action chunk。

Critic 负责：

> 判断某个 action chunk 在当前状态下未来能够获得多少长期收益。

数学定义：

$$ Q_\psi(x_t,a_t) $$

表示：

从状态：

$$ x_t $$

开始执行动作：

$$ a_t $$

之后能够获得的累计折扣奖励。

------------------------------------------------------------------------

## 二、为什么需要 Critic

如果只有 Actor：

    state

    ↓

    Actor

    ↓

    action

    ↓

    environment

    ↓

    reward

Actor 不知道：

    哪个动作更好

因为机器人任务：

-   reward 稀疏；
-   成功通常发生在 episode 末尾；
-   当前动作和未来成功之间存在长距离关系。

例如：

    chunk A

    reward=0


    chunk B

    reward=0


    chunk C

    reward=0


    chunk D

    insert success

    reward=1

Critic 的作用：

把最后成功的信息传播回前面的 chunk。

因此：

    chunk D Q ↑

            |

            v

    chunk C Q ↑

            |

            v

    chunk B Q ↑

            |

            v

    chunk A Q ↑

------------------------------------------------------------------------

## 三、为什么使用 Twin Critic

RLT 原论文采用两个独立的 Q 网络进行 value estimation，即 Twin Critic 结构。

本项目保持该设计。Twin Critic 用于稳定 Q 学习，减少过估计问题，并不是 RL Token 表征本身的核心创新。

定义：

$$ Q_{\psi_1}(x,a) $$

和：

$$ Q_{\psi_2}(x,a) $$

原因：

单 Q 网络容易：

> over estimation

例如：

真实：

$$ Q=5 $$

但是：

网络预测：

$$ Q=8 $$

长期累积导致：

Actor 学习错误方向。

因此 target 使用：

$$ \min(Q_1,Q_2) $$

代码结构：

``` python
critic1 = Critic()

critic2 = Critic()
```

------------------------------------------------------------------------

## 四、Critic 输入设计

Critic 输入：

$$ (x_t,a_t) $$

其中：

状态：

$$ x_t= \left[z_{rl,t},s_t^p\right] $$

包含：

### RL Token

$$ z_{rl} $$

来自：

    SmolVLA

    ↓

    token embedding

    ↓

    RL Token Encoder

    ↓

    z_rl

------------------------------------------------------------------------

### proprioception

例如：

    robot joint position

    robot velocity

    gripper state

------------------------------------------------------------------------

### action chunk

$$ a_t = \left[a_t^1,...,a_t^C\right] $$

其中：

C：

RL chunk length。

------------------------------------------------------------------------

最终：

    z_rl

    +

    proprio

    +

    flatten(action chunk)

            |

            v

    MLP

            |

            v

    Q value

------------------------------------------------------------------------

## 五、为什么 Critic 当前动作必须使用 replay 中保存的 executed action

这是实现中非常容易错误的地方。

Replay 中保存：

$$ a_t^{exec} $$

它代表：

环境真实执行过的动作。

因此：

Critic 当前预测：

$$ Q(x_t,a_t^{exec}) $$

原因：

Replay transition:

$$ x_t \xrightarrow{a_t^{exec}} x_{t+C} $$

这个 transition 是真实发生的。

------------------------------------------------------------------------

错误方式：

重新调用 Actor：

$$ a_t=\pi_\theta(x_t) $$

然后：

$$ Q(x_t,a_t) $$

问题：

这个动作可能：

-   没有被环境执行；
-   没有产生当前 reward；
-   与 replay transition 不匹配。

------------------------------------------------------------------------

## 六、Critic Target 计算

目标：

计算：

$$ y_t $$

用于监督：

$$ Q(x_t,a_t^{exec}) $$

------------------------------------------------------------------------

## Step 1：计算 chunk reward

一个 chunk 有：

$$ n_t $$

个实际执行 step。

reward：

$$ r_t^0,r_t^1,...,r_t^{n_t-1} $$

累计：

$$ R_t= \sum_{i=0}^{n_t-1} \gamma^i r_{t+i} $$

代码：

``` python
discount = 1.0

R = 0

for r in reward_sequence:

    R += discount * r

    discount *= gamma
```

------------------------------------------------------------------------

## Step 2：获得 next action

下一状态：

$$ x_{t+C} $$

使用当前 Actor：

$$ a' = \pi_\theta(x_{t+C}, \tilde a_{t+C}) $$

注意：

这里不是 replay 中旧动作。

原因：

这是 off-policy actor-critic。

Target 评价：

> 当前策略未来会怎么做。

------------------------------------------------------------------------

## Step 3：计算未来 Q

两个 target critic：

$$ Q_{\psi'_1}(x',a') $$

$$ Q_{\psi'_2}(x',a') $$

取：

$$
Q_{next} = \min( Q_{\psi'_1}, Q_{\psi'_2}
)
$$

------------------------------------------------------------------------

## Step 4：最终 TD Target

完整：

$$ y_t = R_t + (1-d_t) \gamma^{n_t} Q_{next} $$

其中：

-   (d_t)：episode 是否结束；
-   (n_t)：实际执行步数。

------------------------------------------------------------------------

## 七、为什么不能固定使用 $\gamma^C$

例如：

chunk:

    10 steps

但是：

第 6 步成功。

episode 结束。

实际：

$$ n_t=6 $$

应该：

$$ \gamma^6 $$

而不是：

$$ \gamma^{10} $$

因此 Replay 必须保存：

    n_steps

------------------------------------------------------------------------

## 八、Terminal 与 Truncated

环境返回：

    terminated

    truncated

区别：

### terminated

真正结束：

例如：

-   成功；
-   失败。

此时：

未来价值：

$$ 0 $$

------------------------------------------------------------------------

### truncated

例如：

time limit。

状态本身可能仍然有价值。

所以：

不要简单：

    done=True

混合。

推荐保存：

``` python
terminated

truncated
```

------------------------------------------------------------------------

## 九、Critic Loss

两个 Q：

$$ Q_1,Q_2 $$

loss：

$$ L_Q = (Q_1-y)^2 + (Q_2-y)^2 $$

代码：

``` python
q1 = critic1(
    state,
    action
)

q2 = critic2(
    state,
    action
)


loss = mse(q1,target)+mse(q2,target)


loss.backward()

critic_optimizer.step()
```

------------------------------------------------------------------------

## 十、Target Network

为了稳定 TD：

维护：

    critic

    target critic

参数：

$$ \psi' $$

使用 Polyak update：

$$
\psi' = \tau\psi +
(1-\tau)\psi'
$$

代码：

``` python
for p,target_p in zip(
    critic.parameters(),
    target.parameters()
):

    target_p.data = (
        tau*p.data
        +(1-tau)*target_p.data
    )
```

------------------------------------------------------------------------

## 十一、Critic PyTorch结构

推荐：

``` python
class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):

        self.net = nn.Sequential(

            nn.Linear(
                state_dim+action_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1
            )
        )


    def forward(
        self,
        state,
        action
    ):

        x=torch.cat(
            [state,action],
            dim=-1
        )

        return self.net(x)
```

------------------------------------------------------------------------

## 十一点五、Critic Current Action 与 Target Action 区别

Critic 训练中，当前 Q 使用的 action 与 TD target 中使用的 action 来源不同。

|位置|Action 来源|作用|
|-|-|-|
|Current Q|Replay Buffer 保存的 executed_action|计算真实 transition 对应的 Q：\(Q(x_t,a_t^{exec})\)|
|Target Q|当前 Actor 根据 next state 重新采样|估计当前策略未来价值：\(Q(x_{t+C},a')\)|
|Actor Update|当前 Actor 根据 replay state 重新生成|优化策略参数，使 Q 提升|

因此：

- Critic 当前值不能重新调用 Actor 替代 replay action；
- TD target 不能使用 replay 中旧 action；
- Actor update 需要重新采样 action 并通过 Critic 获得梯度。

------------------------------------------------------------------------

## 十二、训练时的数据流

Replay:

    sample batch

          |

          v


    current:

    x

    executed_action

    reward


    next:

    x'

    reference'

------------------------------------------------------------------------

Critic:

    (x,executed_action)

            |

            v

    Q_current

Target:

    x'

     +

    reference'

            |

            v

    Actor

            |

            v

    new action


            |

            v

    target critic


            |

            v

    TD target

------------------------------------------------------------------------

## 十三、Critic 阶段验证指标

不要只看 loss。

应该观察：

### 1.

成功 episode 最后的 chunk：

$$ Q $$

是否升高。

------------------------------------------------------------------------

### 2.

Q 是否向前传播：

    success chunk

    ↓

    previous chunk

    ↓

    earlier chunk

------------------------------------------------------------------------

### 3.

target Q 是否稳定。

------------------------------------------------------------------------

## 十四、Critic 完成后的下一步

Critic 学会：

    哪些 action chunk 有价值

之后：

Actor 才能利用：

$$ \nabla_a Q(x,a) $$

优化自己的动作。

下一部分：

Part 5：

Actor 完整实现。

--------------------------------------------------------------------------------


## Part 5：Actor 完整实现 ------ Chunk Policy、Reference Conditioning 与梯度优化

------------------------------------------------------------------------

## 一、Actor 在 RLT 中的作用

在 Critic 学会评价动作价值之后，Actor 的目标：

> 学习生成能够获得更高长期回报，同时不要偏离 VLA 原始能力太远的 action
> chunk。

因此 Actor 不是重新学习整个机器人控制。

而是：

    VLA

    提供基础行为

            +

    Actor

    进行 RL refinement

------------------------------------------------------------------------

## 二、Actor 输入设计

Actor 输入：

$$ (x_t,\tilde a_t) $$

其中：

状态：

$$ x_t=\left[z_{rl},s_t^p\right] $$

包含：

### 1. RL Token

$$ z_{rl} $$

提供：

-   视觉理解；
-   任务上下文；
-   VLA 高层语义信息。

------------------------------------------------------------------------

### 2. proprioception

例如：

    joint position

    joint velocity

    gripper state

提供：

机器人当前物理状态。

------------------------------------------------------------------------

### 3. VLA reference action

$$ \tilde a_t $$

这是非常重要的输入。

它表示：

> 如果没有 RL，VLA 认为机器人应该如何行动。

因此 Actor 学习：

    在 VLA 附近优化

而不是：

    从随机动作开始学习

------------------------------------------------------------------------

## 三、Actor 输出

Actor 输出：

$$ a_t $$

shape：

$$ \left[C,A\right] $$

其中：

-   C：RL action chunk length
-   A：action dimension

例如：

    C=10

    action_dim=7

    output:

    [10,7]

------------------------------------------------------------------------

## 四、为什么 Actor 输出 chunk

因为：

VLA 本身预测 chunk。

因此保持：

    VLA chunk

    ↓

    Actor chunk

    ↓

    Environment

一致。

如果：

Actor 单步输出：

    step1

    step2

    step3

会破坏：

-   temporal smoothness；
-   manipulation contact dynamics。

------------------------------------------------------------------------

## 五、Actor 网络结构

第一版：

MLP 即可。

输入：

    z_rl

    +

    proprio

    +

    reference action

拼接：

$$ h= \left[z_{rl},s^p,\tilde a\right] $$

经过：

    Linear

    ReLU

    Linear

    ReLU

    Linear

输出：

$$ \mu_\theta $$

------------------------------------------------------------------------

代码：

``` python
class Actor(nn.Module):

    def __init__(
        self,
        input_dim,
        chunk_size,
        action_dim
    ):

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                chunk_size*action_dim
            )
        )


    def forward(
        self,
        state,
        reference
    ):

        x=torch.cat(
            [
                state,
                reference.flatten(1)
            ],
            dim=-1
        )

        action=self.net(x)

        return action.reshape(
            -1,
            chunk_size,
            action_dim
        )
```

------------------------------------------------------------------------

## 六、为什么不是 Residual Actor

容易产生误解：

因为 Actor 靠近 VLA。

但是默认 RLT：

不是：

$$ a=\tilde a+\Delta a $$

而是：

$$ a=\pi_\theta(x,\tilde a) $$

即：

Actor 直接输出最终动作。

------------------------------------------------------------------------

但是：

通过：

$$ \beta\|a-\tilde a\|^2 $$

约束动作不要偏离 reference。

因此：

它表现类似 residual refinement。

------------------------------------------------------------------------

## 七、Actor Policy Distribution

为了探索：

Actor 可以建模 Gaussian policy。

输出：

$$ \mu_\theta $$

采样：

$$ a= \mu_\theta+\sigma\epsilon $$

其中：

$$ \epsilon\sim N(0,I) $$

------------------------------------------------------------------------

代码：

``` python
noise=torch.randn_like(mu)

action=mu+sigma*noise
```

------------------------------------------------------------------------

## 八、Actor Loss

Actor 最大化：

$$ Q(x,a) $$

同时保持：

reference consistency。

因此：

$$
\mathcal L_\pi = -Q_\psi(x,a) +
\beta \|a-\tilde a\|^2
$$

------------------------------------------------------------------------

第一项：

$$ -Q $$

作用：

让 Actor 生成高价值动作。

因为优化是梯度下降：

最小化：

$$ -Q $$

等价于：

最大化：

$$ Q $$

------------------------------------------------------------------------

第二项：

$$ \beta\|a-\tilde a\|^2 $$

作用：

限制：

Actor 不要离 VLA 太远。

------------------------------------------------------------------------

## 九、Actor 更新流程

Replay sample：

得到：

    x

    reference_action

------------------------------------------------------------------------

重新生成：

$$ a_{new} = \pi_\theta(x,\tilde a) $$

注意：

不是使用：

    replay中的executed_action

------------------------------------------------------------------------

然后：

输入 Critic：

$$ Q(x,a_{new}) $$

计算：

$$ L_\pi $$

------------------------------------------------------------------------

代码：

``` python
new_action = actor(
    state,
    reference
)


q1=critic1(
    state,
    new_action
)

q2=critic2(
    state,
    new_action
)


q=torch.min(q1,q2)


reference_loss=(
    (new_action-reference)**2
).mean()


actor_loss=(
    -q.mean()
    +
    beta*reference_loss
)
```

------------------------------------------------------------------------

## 十、Actor 更新时 Critic 是否更新

不更新。

Actor update：

冻结：

    Critic parameters

但是：

不能：

``` python
torch.no_grad()
```

原因：

需要梯度：

$$ \frac{\partial Q}{\partial a} $$

传播：

    Critic

    ↓

    action

    ↓

    Actor

------------------------------------------------------------------------

正确：

``` python
for p in critic.parameters():

    p.requires_grad=False


actor_loss.backward()


actor_optimizer.step()


for p in critic.parameters():

    p.requires_grad=True
```

------------------------------------------------------------------------

错误：

``` python
with torch.no_grad():

    q=critic(
        state,
        action
    )
```

这样：

Actor 没有梯度。

------------------------------------------------------------------------

## 十一、Actor 与 Critic 梯度关系

Critic update：

目标：

更新：

$$ \psi $$

路径：

    Q(state,executed_action)

            |

            v

    critic parameters

Actor 不参与。

------------------------------------------------------------------------

Actor update：

目标：

更新：

$$ \theta $$

路径：

    Actor

     |

     v

    action

     |

     v

    Q

     |

     v

    gradient back

     |

     v

    Actor

Critic 参数冻结。

------------------------------------------------------------------------

## 十二、Reference Dropout

论文中一个重要技巧：

训练 Actor 时随机 mask reference action。

原因：

防止：

Actor 完全依赖：

$$ \tilde a $$

导致：

没有 RL 能力。

------------------------------------------------------------------------

实现：

概率：

$$ p_{drop} $$

默认配置：

$$ p_{drop}=0 $$

该概率作为实验超参数，根据 ablation 再调整。

例如：

``` python
mask=torch.rand(batch)<0.5

reference[mask]=0
```

------------------------------------------------------------------------

注意：

只修改：

Actor 输入。

不要修改：

reference regularization target。

否则：

$$ \|a-\tilde a\|^2 $$

失去意义。

------------------------------------------------------------------------

## 十三、Actor Warm Start

随机初始化 Actor：

直接控制机器人：

风险：

    random policy

    ↓

    all fail

    ↓

    no useful reward

------------------------------------------------------------------------

因此：

推荐：

warm start：

行为：

$$ a=\tilde a $$

或者：

增加短暂 BC loss：

$$ L_{BC} = \|a-\tilde a\|^2 $$

让 Actor 先靠近 VLA。

------------------------------------------------------------------------

进入正式 RL：

使用：

$$ -Q+\beta L_{ref} $$

------------------------------------------------------------------------

## 十四、Actor 训练验证指标

不能只看 loss。

观察：

### 1.

Actor action 与 reference 距离：

$$ \|a-\tilde a\| $$

是否逐渐合理。

------------------------------------------------------------------------

### 2.

Actor action 的 Q：

$$ Q(x,a_\pi) $$

是否提高。

------------------------------------------------------------------------

### 3.

真实环境 success rate

是否提高。

------------------------------------------------------------------------

## 十五、Actor 完成后的整体关系

最终：

    Observation

          |

          v

    SmolVLA

          |

          +--------------+
          |              |
          v              v

    RL Token       Reference action


          |

          v

    Actor

          |

          v

    Improved action chunk


          |

          v

    Environment

------------------------------------------------------------------------

下一部分：

Part 6：

完整训练循环、Trainer设计、仓库结构、实验规划。

--------------------------------------------------------------------------------


## Part 6：完整训练系统、工程结构、实验规划与开发路线

------------------------------------------------------------------------

## 一、完整 Online RL 训练循环

经过前面的模块：

已经拥有：

    Frozen SmolVLA

    Frozen RL Token Encoder

    Actor

    Critic

    Replay Buffer

    ManiSkill Environment

最终训练：

    environment interaction

            |

            v

    replay buffer

            |

            v

    critic update

            |

            v

    actor update

            |

            v

    better policy

------------------------------------------------------------------------

## 二、整体 Trainer 结构设计

不要写成单文件。

推荐：

    train_online_rl.py

            |

            v

    RLTTrainer

            |

            +----------------+

            |                |

            v                v

    RolloutCollector     Agent


                           |

                           +------ Actor

                           |

                           +------ Critic

------------------------------------------------------------------------

## 三、RLTTrainer 主循环

伪代码：

``` python
for episode in range(num_episodes):


    collect_rollout()


    if replay.size < warmup_size:

        continue


    for update in range(
        updates_per_episode
    ):


        batch=replay.sample()


        update_critic(batch)


        if step % actor_interval == 0:

            update_actor(batch)


        update_target_network()
```

------------------------------------------------------------------------

## 四、Rollout 和 Learning 解耦

推荐逻辑：

    Collector

    负责：

    environment interaction


    Learner

    负责：

    gradient update

第一版：

同步即可：

    collect

    ↓

    update

    ↓

    collect

以后：

可以扩展：

    Actor workers

            |

            v

    Replay

            |

            v

    Learner GPU

------------------------------------------------------------------------

## 五、Critic Update 和 Actor Update 顺序

推荐：

先 Critic。

原因：

Actor 依赖：

$$ Q_\psi(x,a) $$

如果 Critic 没学会：

Actor 没有优化方向。

因此：

    Replay sample

            |

            v

    Critic update

            |

            v

    Actor update

------------------------------------------------------------------------

## 六、Update To Data Ratio（UTD）

强化学习中：

环境数据昂贵。

一次 rollout 获得的数据：

可以多次训练。

定义：

$$ UTD = \frac{\text{gradient updates}}{\text{environment steps}} $$

RLT 类方法通常使用较高 UTD。

第一版：

    UTD = 5

例如：

获得一个 chunk transition：

    environment

    ↓

    replay

    ↓

    5次gradient update

------------------------------------------------------------------------

## 七、Actor Update Frequency

不要每次 Critic 更新都 Actor 更新。

例如：

    critic update

    critic update

    actor update


    critic update

    critic update

    actor update

即：

    critic:actor

    2:1

或者：

    5:1

根据实验调整。

------------------------------------------------------------------------

## 八、完整训练状态流

一次 transition：

    Observation_t


          |

          v


    SmolVLA


          |

          +----------------+

          |                |

          v                v


    tokens          reference action


          |

          v


    RL Token Encoder


          |

          v


    z_rl


          |

          v


    Actor


          |

          v


    executed action


          |

          v


    Environment


          |

          v


    reward


          |

          v


    next observation

------------------------------------------------------------------------

## 九、Checkpoint 设计

建议：

    checkpoints/


        smolvla/

            model.pt


        rl_token/

            encoder.pt


        actor/

            actor.pt


        critic/

            critic1.pt

            critic2.pt


        optimizer/

------------------------------------------------------------------------

## 十、推荐仓库结构

    smolvla-rlt/


    ├── configs/


    │   ├── env/


    │   │   ├── peg.yaml


    │   │   └── plug.yaml


    │   │


    │   ├── vla/


    │   │   └── smolvla.yaml


    │   │


    │   ├── rlt/


    │   │   └── rl_token.yaml


    │   │


    │   └── rl/


    │       └── actor_critic.yaml



    ├── src/


    │   └── smolvla_rlt/


    │


    │       ├── vla/


    │       │   ├── adapter.py


    │       │   └── smolvla.py


    │


    │       ├── rlt/


    │       │   ├── encoder.py


    │       │   ├── decoder.py


    │       │   └── loss.py


    │


    │       ├── rl/


    │       │   ├── actor.py


    │       │   ├── critic.py


    │       │   ├── replay.py


    │       │   └── agent.py


    │


    │       ├── envs/


    │       │   ├── maniskill.py


    │       │   └── chunk_env.py


    │


    │       ├── rollout/


    │       │   ├── collector.py


    │       │   └── transition.py


    │


    │       └── trainers/


    │           ├── train_rl_token.py


    │           └── train_rlt.py


    ├── scripts/


    └── tests/

------------------------------------------------------------------------

## 十一、必须添加的单元测试

强化学习项目最危险：

不是代码报错。

而是：

代码运行，但是算法逻辑错误。

------------------------------------------------------------------------

### 1. RL Token gradient test

检查：

    SmolVLA grad == None


    Encoder grad != None


    Decoder grad != None

------------------------------------------------------------------------

### 2. Decoder causal mask test

检查：

预测：

$$ z_i $$

不能看到：

$$ z_i,z_{i+1} $$

------------------------------------------------------------------------

### 3. Replay transition test

检查：

    state A

    action A

    reward

    state B

确实对应：

    A -> B

------------------------------------------------------------------------

### 4. Critic action test

检查：

Critic 当前 Q：

使用：

    executed_action

不是：

    actor(state)

------------------------------------------------------------------------

### 5. TD target test

检查：

next action：

来自：

    current actor

不是：

    buffer old action

------------------------------------------------------------------------

### 6. Actor gradient test

检查：

Actor update:

    actor parameters change


    critic parameters unchanged

但是：

$$ \nabla_a Q $$

存在。

------------------------------------------------------------------------

## 十二、开发路线规划

------------------------------------------------------------------------

## M0：环境打通

目标：

    ManiSkill

    reset

    step

    reward

    success

    RGB

    proprio

    action

全部可用。

------------------------------------------------------------------------

## M1：数据系统

完成：

    ManiSkill demo

            |

            v

    LeRobot dataset

------------------------------------------------------------------------

## M2：SmolVLA baseline

目标：

    SmolVLA

    ↓

    PegInsertion

    ↓

    non-zero success

------------------------------------------------------------------------

## M3：VLA feature extraction

完成：

    observation

    ↓

    token embedding

    ↓

    reference action

------------------------------------------------------------------------

## M4：RL Token

训练：

    Encoder

    +

    Decoder

验证：

    reconstruction loss下降

------------------------------------------------------------------------

## M5：Replay Buffer

先关闭 Actor。

使用：

    SmolVLA reference policy

收集：

    replay

------------------------------------------------------------------------

## M6：Critic Only

训练：

    Critic

观察：

    success chunk

    Q提高

    ↓

    前序chunk

    Q传播

------------------------------------------------------------------------

## M7：Actor

加入：

    Actor update

验证：

    Q(actor action)

    提高

------------------------------------------------------------------------

## M8：完整 Online RL

打开：

    Actor rollout

    ↓

    Replay

    ↓

    Critic

    ↓

    Actor

------------------------------------------------------------------------

## M9：Sparse Reward

从：

    dense reward debug

切换：

    success reward

------------------------------------------------------------------------

## M10：第二任务

迁移：

    PegInsertion

    ↓

    PlugCharger

验证：

框架通用性。

------------------------------------------------------------------------

## 十三、V1 与 V2 规划

### V1：Correctness First

实现：

    ManiSkill

    PegInsertion

    SmolVLA

    RL Token

    C=10

    Twin Critic

    Actor

    uniform replay

目标：

证明：

RLT pipeline 正确。

------------------------------------------------------------------------

### V2：Paper-level Enhancement

加入：

    stride sampling

    reference dropout

    async rollout

    multi-task

    human correction

    more ablation

------------------------------------------------------------------------

## 十四、最终实验设计

核心 baseline：

  方法                                   作用
  -------------------------------------- -----------------------
  SmolVLA                                VLA baseline
  Actor without RL Token                 验证 RL Token
  RLT                                    主模型
  RLT without reference regularization   验证约束
  C=1                                    action chunk ablation
  C=10                                   主设置
  dense reward                           debug
  sparse reward                          正式实验

------------------------------------------------------------------------

## 十五、最终项目目标

最终形成：

    SmolVLA

            +

    RL Token representation learning

            +

    Chunk-level Actor-Critic

            +

    Simulation robotics benchmark

一个完整、模块化、可开源的 RLT reproduction framework。

核心原则：

    论文定义算法

    GitHub提供工程参考

    自己实现核心模块

而不是简单 fork。
