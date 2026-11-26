# 3D 回放查看器接入方案（2026-11-25 录制侧实现；2026-11-26 站点侧接入）

## 当前状态

| 部分 | 状态 |
| --- | --- |
| `src/smolvla_rltoken/viewer3d.py`（GLB 导出、traj 结构、HUD、镜头预设）+ `tests/test_viewer3d.py` | ✅ |
| `scripts/record_viewer3d.py --demo`：演示 `env_states` 回放，`physx_cpu`，0 显存 | ✅ motionplanning ep0 / rl ep5 已录，结束时的成功判定与演示记录一致 |
| `scripts/record_viewer3d.py --checkpoint`：SFT 策略录制 | ⏳ 代码已写（复用 `run_episode` 的 `frame_callback`），等 SFT checkpoint 实测 |
| 站点侧（`?rec=` 取每局 GLB、读 `metadata.camera_presets` / `xray_names` / `hud_fields`） | ✅ 2026-11-26 接入：`#/p/smolvla-rltoken/viewer`，两局演示可回放（无头 Chromium 检查过镜头预设、孔壁透视、HUD、插入进度） |

站点直接读本仓库的 `outputs/viewer3d/`（`projects.yaml` 里 `recordings: outputs/viewer3d`），**不需要拷贝**：录完刷新查看器页面就会出现在右侧录制列表里，不用重启站点。站点不提供「录制此局」按钮，录制只在本仓库做。

用法：

```bash
python scripts/record_viewer3d.py --check                       # CPU 预检
python scripts/record_viewer3d.py                               # 默认 motionplanning 演示 ep0
python scripts/record_viewer3d.py --demo data/maniskill/demos/PegInsertionSide-v1/rl/trajectory.h5 --episode-id 5
python scripts/record_viewer3d.py --checkpoint outputs/sft/peg_insertion/checkpoints/last/pretrained_model --seed 10000
```

实测（演示 ep0，178 步）：GLB 11.8 MB（34 个 mesh part），traj 195 KB，录制 2.5 s。实现与下文方案的差异：

- 手腕 RealSense（`camera_link`）原网格约 127 万顶点，超过 `MAX_PART_VERTICES=200k` 的 part 换成包围盒，否则 GLB 会多出几十 MB。
- 桌面和地面是贴图材质，导出时用贴图均值（sRGB 转线性）乘到 `baseColorFactor` 上；不导出贴图。
- HUD：`depth_mm` 为 peg 头越过孔口端面（孔坐标系 `x = -L`）的距离，`success_depth_mm = L - 15 mm` 是 ManiSkill 开始判成功的位置；另有 `lateral_mm`、`dy_mm`/`dz_mm`、`aligned`、`success`、`source`（`demo` / `vla`，以后加 `actor`）。
- `metadata` 额外给出 `camera_presets`（全景 / 孔口侧视 / 沿孔轴 / 俯视 / 夹爪特写，按每局孔位计算）、`default_preset`、`xray_names`（4 块孔壁）、`xray_label`、`geometry`（peg 半长、半径、孔半径、间隙）。
- 站点 HUD 由 `metadata.hud_fields`（`[{key, label, unit, digits}]`，对应 `viewer3d.HUD_FIELDS`）和 `metadata.hud_progress`（`depth_mm / success_depth_mm` 插入进度条）驱动；改 `peg_hud_fields` 的输出字段时要同步 `HUD_FIELDS`，`test_site_hud_fields_cover_the_recorded_row_keys` 会检查两者一致。2026-11-26 之前录的文件没有这两项，站点会退回显示原始字段名，需要 `--overwrite` 重录。
- 演示回放先用 `reset(seed, reconfigure=True)` 重建几何，再比对第 0 帧 peg 位姿（误差 > 1e-4 直接报错），避免几何和状态对不上。rl 演示是旧版 state-dict 键名（`peg_0`），回放时会做映射，并且没有 `reset_kwargs.seed`，用 `episode_seed` 代替。

目标：在 rl-experiment-site 上自由视角回放 PegInsertion 的单局（SFT、冻结 VLA、确定性 Actor 对照），直接看 peg 和孔口的毫米级对位、
插入深度和失败时卡在哪里。这是 MP4 视频（固定三相机）看不清的部分。

## 结论

**沿用站点已有的 Three.js WebGL 离线回放路线**（`l20-screwdriver` 和 `rm65-l20` 正在用的 `static/viewer3d.js`），
**录制和 GLB 导出放在本仓库里做**，站点只负责读取和播放：

```text
本仓库（smolvla-rlt env，需要 GPU）                                站点（.venv，0 显存）
scripts/record_viewer3d.py  ── 跑一局策略 ──▶  <key>.glb          ──▶  GET scene.glb?rec=<key>
  （ManiSkill 单环境，与 eval.py 同 seed/同 runner）  <key>.traj.json     ──▶  viewer3d.js 60 fps 回放
                                                  <key>.json（结论/数值）──▶  录制列表 / HUD
```

不选的方案：

| 方案 | 不选的原因 |
| --- | --- |
| 站点服务端跑 SAPIEN 实时串流（类似旧 Isaac Gym 查看器） | 共享 4090 上每次查看都要占显存、加载 VLA；站点 `.venv` 是 Python 3.8，装不了本项目的依赖；rm65-l20 已经因为这个原因迁走 |
| 站点侧写 ManiSkill 录制 worker（类似 `workers/isaaclab_screwdrive_record.py`） | 违反「站点定协议、仓库自维护」；策略加载、seed 和 chunk 执行逻辑必须和 `scripts/eval.py` 完全一致，放在本仓库才能共用代码和测试 |
| 只用 MP4 视频 | 现在就能用（见下文「立即可用」），但视角固定，看不清侧插的间隙 |

## 为什么 PegInsertion 适合这条路线

- **几何简单且可完整导出**：peg 是两段 box（头/尾不同颜色），带孔的 box 是 4 块 box 拼成的 kinematic actor，桌面是 ManiSkill 的 `TableSceneBuilder`；
  Panda 各 link 的视觉网格本身就是 `.glb`（`mani_skill/assets/robots/panda/franka_description/meshes/visual/`，约 10 MB）。
  `mani_skill.utils.geometry.trimesh_utils.get_render_body_meshes` 可以把每个 link/actor 的 render shape 转成 trimesh，用 `trimesh` 5.x 直接导出 GLB。
- **位姿格式与查看器一致**：SAPIEN/ManiSkill 的 `pose.raw_pose` 是 `[x, y, z, qw, qx, qy, qz]`，正好是 `viewer3d.js` 的默认 `wxyz` 七元组，不需要转换。
- **数据量小**：每步只需 Panda 的约 11 个 link 加 peg 1 个位姿（box 是 kinematic，放进 `static_poses`），200 步 × 12 个位姿，JSON 约 100–200 KB。

## 唯一的差异：几何每局都不同

PegInsertion 在 reconfigure 时随机化 peg 半长（0.085–0.125 m）、半径（0.015–0.025 m）和孔的间隙，**不能像另外两个项目那样全项目共用一个 `scene.glb`**。
做法是每局导出一个 `<key>.glb`：Panda 网格每次都打包进去约 10 MB，可以接受。如果以后录制变多，再拆成「共享 `robot.glb` + 每局 `objects.glb`」。

## 需要的改动

### 本仓库（主体）

1. `scripts/record_viewer3d.py`：参数和 `scripts/eval.py` 一致（`--checkpoint`、`--seed`、`--action-steps`，以后加 `--online-rl` 加载 Actor），单环境运行；
   每个控制步记录 link/actor 位姿和 HUD 数值（成功标志、peg 头到孔口的距离和插入深度、当前 chunk 来源：VLA / Actor）。
2. GLB 命名约定（与 `viewer3d.js` 的分组规则一致）：mesh 名为 `<body>__<part>`，例如 `panda_link3__visual0`、`peg__head`、`box__wall2`、`table__top`；
   `traj.frames[i]` 是 `{body: p7}`，`traj.rows[i]` 放 HUD 数值，`metadata` 放 `quat_order: "wxyz"`、`step_dt: 0.05`（20 Hz）、seed、checkpoint、结论。
3. 输出到 `outputs/viewer3d/<key>.{glb,traj.json,json}`（gitignore 范围内）。~~再由站点的打包工具拷进站点 `cache/smolvla-rltoken/recordings/`~~ 实际实现：站点通过 `projects.yaml` 的 `recordings:` 直接读取这个目录（每局 GLB 约 12 MB，拷贝没有意义）。
4. CPU 测试：用 Mock env 验证 traj 结构、mesh 命名和 GLB 能被 `trimesh.load` 读回。

### 站点（小改动，需要在站点仓库单独提交，`server.py` 改完要 `tools/serve.sh restart`）

1. `api_scene_glb` 支持 `?rec=<key>`，优先返回 `recordings/<key>.glb`，没有时回退到 `scene.glb`。
2. `app.js` 不再按项目 id 写死镜头预设：优先读 `traj.metadata.camera_presets`，没有时用现有默认值。本项目会提供「全景 / 孔口侧视 / 俯视 / 夹爪特写」。
3. 不支持「录制此局」按钮：录制需要本仓库的 GPU 环境，由本仓库的 agent 离线执行（与站点「只读取项目仓库」的原则一致）。
4. 同步更新 `docs/integration-guide.md`，加一节「标准 3D 回放协议」，让以后的仓库也能零代码接入。

### 显存与时机

单环境录制需要加载 SmolVLA（约 2–3 GB）和 SAPIEN 渲染（不开相机时可以关闭传感器渲染）。必须先用 `nvidia-smi` 确认显存够用，不要和训练、评测同时跑。
等 SFT 重训完成、有 checkpoint 以后再实现；在那之前只能回放演示数据（演示 `.h5` 里有 `env_states`，可以用来做无策略的链路验证，需要先下载原始演示）。

## 立即可用：MP4 视频

`scripts/eval.py` 已经按成败各保留前 N 局三相机拼接 MP4（`outputs/eval/.../videos/{success,failure}/episode_XXXX_seed_N.mp4`）。不需要任何新代码，
在 `progress.yaml` 的阶段里写 `videos` 条目，`src` 指向具体的 `.mp4` 文件（指向目录时站点只取按文件名排序的第一个），然后在站点目录运行：

```bash
.venv/bin/python tools/build_media.py --project smolvla-rltoken --worktree
```
