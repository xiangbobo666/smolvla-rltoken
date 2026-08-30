#!/usr/bin/env python3
"""Serve a live browser UI for tuning PegInsertionSide camera poses.

The app keeps one recorded demonstration state fixed.  Moving a slider updates
the corresponding ManiSkill sensor pose in the already-created scene and
returns a freshly rendered RGB frame.  It never alters data until the user
explicitly presses "Save configuration" in the browser.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import threading
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import sapien
from PIL import Image

import mani_skill.envs  # Registers the parent environment.
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import sapien_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Importing this module registers PegInsertionSideThreeCamera-v1.
import smolvla_rltoken.envs.maniskill_env  # noqa: F401, E402
from smolvla_rltoken.envs.maniskill_env import hand_camera_pose, peg_end_camera_pose


DEFAULT_TRAJECTORY = (
    REPO_ROOT / "data/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
)
DEFAULT_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
WORLD_CAMERA_UIDS = ("environment_camera",)
CAMERA_UIDS = (*WORLD_CAMERA_UIDS, "insertion_camera", "hand_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def to_uint8_rgb(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def encode_jpeg(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def camera_pose(spec: dict[str, Any]):
    return sapien_utils.look_at(spec["position"], spec["target"])


class CameraTuner:
    """Owns one environment and serializes access to the GPU renderer."""

    def __init__(self, trajectory_path: Path, config_path: Path):
        self.trajectory_path = trajectory_path
        self.config_path = config_path
        self.lock = threading.Lock()
        self.specs = self._read_config()
        self.state_cache: dict[int, list[dict[str, Any]]] = {}
        self.episode_seeds = self._read_episode_seeds()
        self.loaded_episode: int | None = None
        self.current_episode = 0
        self.current_frame = 0
        self.env = gym.make(
            "PegInsertionSideThreeCamera-v1",
            obs_mode="rgb",
            control_mode="pd_joint_pos",
            sim_backend="physx_cpu",
            camera_specs=self.specs,
        )
        self.env.reset(seed=0)

    def _read_episode_seeds(self) -> dict[int, int]:
        with self.trajectory_path.with_suffix(".json").open() as metadata_file:
            episodes = json.load(metadata_file)["episodes"]
        return {int(item["episode_id"]): int(item["episode_seed"]) for item in episodes}

    def _read_config(self) -> dict[str, Any]:
        with self.config_path.open() as config_file:
            specs = json.load(config_file)
        required = set(CAMERA_UIDS)
        missing = required - set(specs)
        if missing:
            raise ValueError(f"Missing camera configuration(s): {sorted(missing)}")
        return specs

    def _states_for_episode(self, episode: int) -> list[dict[str, Any]]:
        if episode not in self.state_cache:
            with h5py.File(self.trajectory_path, "r") as h5_file:
                key = f"traj_{episode}"
                if key not in h5_file:
                    raise KeyError(f"{key} is not present in {self.trajectory_path}")
                self.state_cache[episode] = trajectory_utils.dict_to_list_of_dicts(
                    h5_file[key]["env_states"]
                )
        return self.state_cache[episode]

    @staticmethod
    def _validate_specs(specs: dict[str, Any]) -> None:
        for uid in WORLD_CAMERA_UIDS:
            spec = specs[uid]
            for key in ("position", "target"):
                values = spec[key]
                if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"{uid}.{key} must contain three finite numbers")
            fov = float(spec["fov_deg"])
            if not 5 < fov < 175:
                raise ValueError(f"{uid}.fov_deg must be between 5 and 175")

        insertion_spec = specs["insertion_camera"]
        for key in ("relative_position", "relative_target"):
            values = insertion_spec[key]
            if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"insertion_camera.{key} must contain three finite numbers")
        insertion_fov = float(insertion_spec["fov_deg"])
        if not 5 < insertion_fov < 175:
            raise ValueError("insertion_camera.fov_deg must be between 5 and 175")

        hand_spec = specs["hand_camera"]
        for key in ("local_position", "local_rpy_deg"):
            values = hand_spec[key]
            if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"hand_camera.{key} must contain three finite numbers")
        hand_fov = float(hand_spec["fov_deg"])
        if not 5 < hand_fov < 175:
            raise ValueError("hand_camera.fov_deg must be between 5 and 175")

    def _apply_specs(self) -> None:
        self._validate_specs(self.specs)
        for uid in WORLD_CAMERA_UIDS:
            spec = self.specs[uid]
            sensor = self.env.unwrapped._sensors[uid]
            pose = camera_pose(spec)
            sensor.camera.set_local_pose(pose.sp)
            sensor.camera.set_fovy(np.deg2rad(float(spec["fov_deg"])))
            sensor.config.pose = pose
            sensor.config.fov = np.deg2rad(float(spec["fov_deg"]))

        insertion_spec = self.specs["insertion_camera"]
        insertion_sensor = self.env.unwrapped._sensors["insertion_camera"]
        peg_end = self.env.unwrapped.peg_head_offsets.p[0].detach().cpu().numpy()
        insertion_pose = peg_end_camera_pose(insertion_spec, peg_end)
        insertion_sensor.camera.set_local_pose(insertion_pose.sp)
        insertion_sensor.camera.set_fovy(np.deg2rad(float(insertion_spec["fov_deg"])))
        insertion_sensor.config.pose = insertion_pose
        insertion_sensor.config.fov = np.deg2rad(float(insertion_spec["fov_deg"]))

        hand_spec = self.specs["hand_camera"]
        hand_sensor = self.env.unwrapped._sensors["hand_camera"]
        hand_pose = hand_camera_pose(hand_spec)
        hand_sensor.camera.set_local_pose(hand_pose)
        hand_sensor.camera.set_fovy(np.deg2rad(float(hand_spec["fov_deg"])))
        hand_sensor.config.pose = hand_pose
        hand_sensor.config.fov = np.deg2rad(float(hand_spec["fov_deg"]))

    def render(
        self, specs: dict[str, Any] | None = None, episode: int | None = None, frame: int | None = None
    ) -> dict[str, Any]:
        with self.lock:
            if specs is not None:
                candidate = deepcopy(self.specs)
                for uid in CAMERA_UIDS:
                    candidate[uid].update(specs[uid])
                self._validate_specs(candidate)
                self.specs = candidate

            if episode is not None:
                self.current_episode = int(episode)
            if frame is not None:
                self.current_frame = int(frame)
            states = self._states_for_episode(self.current_episode)
            if self.current_episode not in self.episode_seeds:
                raise KeyError(f"No metadata seed is available for traj_{self.current_episode}")
            if not -len(states) <= self.current_frame < len(states):
                raise IndexError(
                    f"frame {self.current_frame} is outside [{-len(states)}, {len(states) - 1}]"
                )
            self.current_frame %= len(states)
            if self.loaded_episode != self.current_episode:
                # HDF5 states restore poses and joints but not the randomized
                # box dimensions/hole offset. Rebuild them from the saved seed.
                self.env.reset(seed=self.episode_seeds[self.current_episode])
                self.loaded_episode = self.current_episode
            self._apply_specs()
            self.env.unwrapped.set_state_dict(states[self.current_frame])

            sensor_images = self.env.unwrapped.get_sensor_images()
            images = {
                uid: encode_jpeg(to_uint8_rgb(sensor_images[uid]["rgb"]))
                for uid in ("environment_camera", "hand_camera", "insertion_camera")
            }
            return {
                "images": images,
                "specs": self.specs,
                "episode": self.current_episode,
                "frame": self.current_frame,
                "frame_count": len(states),
            }

    def save(self) -> dict[str, Any]:
        with self.lock:
            temporary_path = self.config_path.with_suffix(".json.tmp")
            with temporary_path.open("w") as config_file:
                json.dump(self.specs, config_file, indent=2)
                config_file.write("\n")
            temporary_path.replace(self.config_path)
            return {"specs": self.specs}

    def load_saved_config(self) -> dict[str, Any]:
        with self.lock:
            self.specs = self._read_config()
            self._apply_specs()
            return {"specs": self.specs}

    def close(self) -> None:
        self.env.close()


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Peg Insertion Camera Tuner</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #16191f; color: #eef1f5; }
    header { padding: 14px 20px; background: #202630; display: flex; gap: 14px; align-items: center; }
    h1 { font-size: 18px; margin: 0; }
    #status { color: #aab7c7; font-size: 13px; }
    main { display: grid; grid-template-columns: 330px minmax(0, 1fr); gap: 16px; padding: 16px; }
    .controls, .frame-control { background: #202630; border: 1px solid #303a48; border-radius: 8px; padding: 14px; }
    .controls h2 { font-size: 16px; margin: 0 0 10px; }
    .group { border-top: 1px solid #303a48; margin-top: 12px; padding-top: 12px; }
    .group:first-of-type { border-top: 0; margin-top: 0; padding-top: 0; }
    .label { display: flex; justify-content: space-between; align-items: center; font-size: 13px; margin: 9px 0 3px; }
    input[type=range] { width: 100%; accent-color: #69b3ff; }
    input[type=number] { width: 74px; background: #12161d; border: 1px solid #475568; color: #eef1f5; border-radius: 4px; padding: 3px 5px; }
    button { background: #2f81f7; border: 0; color: white; border-radius: 5px; padding: 8px 10px; cursor: pointer; margin: 10px 6px 0 0; }
    button.secondary { background: #3b4654; }
    .panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-content: start; }
    figure { margin: 0; background: #202630; border: 1px solid #303a48; border-radius: 8px; overflow: hidden; }
    figcaption { padding: 8px 10px; font-size: 13px; }
    img { width: 100%; display: block; aspect-ratio: 1; object-fit: contain; background: black; }
    .frame-control { grid-column: 1 / -1; display: flex; align-items: end; gap: 9px; padding: 10px; }
    .frame-control label { font-size: 13px; display: grid; gap: 4px; }
    @media (max-width: 950px) { main { grid-template-columns: 1fr; } .panels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>PegInsertionSide 三相机实时调参</h1><span id="status">正在连接渲染器…</span></header>
  <main>
    <section class="controls">
      <div class="group" data-camera="environment_camera"><h2>环境相机（世界坐标）</h2></div>
      <div class="group" data-camera="insertion_camera"><h2>插入位相机（随红色末端移动）</h2></div>
      <div class="group" data-camera="hand_camera"><h2>腕部相机（相对 camera_link）</h2></div>
      <button id="save">保存配置</button><button id="reload" class="secondary">恢复已保存配置</button>
    </section>
    <section class="panels">
      <div class="frame-control">
        <label>轨迹编号<input id="episode" type="number" value="0" min="0" step="1"></label>
        <label>状态帧<input id="frame" type="number" value="120" step="1"></label>
        <button id="load-frame" class="secondary">载入状态帧</button>
        <span id="frame-info"></span>
      </div>
      <figure><figcaption>environment_camera</figcaption><img id="environment_camera" alt="环境相机"></figure>
      <figure><figcaption>hand_camera（随腕部运动）</figcaption><img id="hand_camera" alt="腕部相机"></figure>
      <figure><figcaption>insertion_camera</figcaption><img id="insertion_camera" alt="插入位相机"></figure>
    </section>
  </main>
  <script>
    const worldRanges = {
      'position.0': [-1.0, 1.0, 0.01, '位置 x'], 'position.1': [-1.0, 1.0, 0.01, '位置 y'], 'position.2': [0.02, 1.2, 0.01, '位置 z'],
      'target.0': [-0.6, 0.6, 0.01, '注视 x'], 'target.1': [-0.6, 0.9, 0.01, '注视 y'], 'target.2': [0.02, 0.7, 0.01, '注视 z'],
      'fov_deg': [20, 130, 1, 'FOV（度）']
    };
    const wristRanges = {
      'local_position.0': [-0.20, 0.20, 0.005, '局部位置 x'], 'local_position.1': [-0.20, 0.20, 0.005, '局部位置 y'], 'local_position.2': [-0.20, 0.20, 0.005, '局部位置 z'],
      'local_rpy_deg.0': [-180, 180, 1, '局部 Roll'], 'local_rpy_deg.1': [-180, 180, 1, '局部 Pitch'], 'local_rpy_deg.2': [-180, 180, 1, '局部 Yaw'],
      'fov_deg': [20, 130, 1, 'FOV（度）']
    };
    const insertionRanges = {
      'relative_position.0': [-0.60, 0.60, 0.01, '相对红端位置 x'], 'relative_position.1': [-0.60, 0.60, 0.01, '相对红端位置 y'], 'relative_position.2': [-0.60, 0.60, 0.01, '相对红端位置 z'],
      'relative_target.0': [-0.20, 0.20, 0.01, '相对红端注视 x'], 'relative_target.1': [-0.20, 0.20, 0.01, '相对红端注视 y'], 'relative_target.2': [-0.20, 0.20, 0.01, '相对红端注视 z'],
      'fov_deg': [20, 130, 1, 'FOV（度）']
    };
    const cameraRanges = {environment_camera: worldRanges, insertion_camera: insertionRanges, hand_camera: wristRanges};
    let specs = null;
    let queued = null;
    function nestedValue(camera, path) { const [root, index] = path.split('.'); return index === undefined ? specs[camera][root] : specs[camera][root][Number(index)]; }
    function setNested(camera, path, value) { const [root, index] = path.split('.'); if (index === undefined) specs[camera][root] = value; else specs[camera][root][Number(index)] = value; }
    function buildControls() {
      for (const [camera, ranges] of Object.entries(cameraRanges)) {
        const root = document.querySelector(`[data-camera="${camera}"]`);
        for (const [path, [min, max, step, label]] of Object.entries(ranges)) {
          const key = `${camera}:${path}`;
          const row = document.createElement('div');
          row.innerHTML = `<div class="label"><span>${label}</span><input id="number-${key}" type="number" min="${min}" max="${max}" step="${step}"></div><input id="range-${key}" type="range" min="${min}" max="${max}" step="${step}">`;
          root.appendChild(row);
          const range = row.querySelector(`#range-${CSS.escape(key)}`);
          const number = row.querySelector(`#number-${CSS.escape(key)}`);
          const sync = value => { setNested(camera, path, Number(value)); range.value = value; number.value = Number(value).toFixed(step < 1 ? 2 : 0); queueRender(); };
          range.addEventListener('input', event => sync(event.target.value));
          number.addEventListener('change', event => sync(event.target.value));
        }
      }
      syncControls();
    }
    function syncControls() {
      for (const [camera, ranges] of Object.entries(cameraRanges)) for (const [path, [, , step]] of Object.entries(ranges)) {
        const key = `${camera}:${path}`; const value = nestedValue(camera, path);
        document.querySelector(`#range-${CSS.escape(key)}`).value = value;
        document.querySelector(`#number-${CSS.escape(key)}`).value = Number(value).toFixed(step < 1 ? 2 : 0);
      }
    }
    async function request(url, body) {
      const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
      const payload = await response.json(); if (!response.ok) throw new Error(payload.error || response.statusText); return payload;
    }
    function showResult(result) {
      specs = result.specs;
      for (const [uid, image] of Object.entries(result.images || {})) document.getElementById(uid).src = `data:image/jpeg;base64,${image}`;
      if (result.frame_count !== undefined) document.getElementById('frame-info').textContent = `第 ${result.frame} 帧 / 共 ${result.frame_count} 帧`;
      syncControls(); document.getElementById('status').textContent = '实时预览中；未保存的修改仅存在于当前进程。';
    }
    async function render() {
      document.getElementById('status').textContent = 'GPU 正在渲染…';
      try { showResult(await request('/api/render', {specs, episode: Number(document.getElementById('episode').value), frame: Number(document.getElementById('frame').value)})); }
      catch (error) { document.getElementById('status').textContent = `渲染失败：${error.message}`; }
    }
    function queueRender() { clearTimeout(queued); queued = setTimeout(render, 120); }
    document.getElementById('load-frame').addEventListener('click', render);
    document.getElementById('save').addEventListener('click', async () => { try { await request('/api/save'); document.getElementById('status').textContent = '已保存到相机配置 JSON。'; } catch (error) { document.getElementById('status').textContent = `保存失败：${error.message}`; } });
    document.getElementById('reload').addEventListener('click', async () => { try { const result = await request('/api/reload'); specs = result.specs; syncControls(); render(); } catch (error) { document.getElementById('status').textContent = `恢复失败：${error.message}`; } });
    (async () => { try { const result = await request('/api/render', {episode: 0, frame: 120}); specs = result.specs; buildControls(); showResult(result); } catch (error) { document.getElementById('status').textContent = `启动失败：${error.message}`; } })();
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    tuner: CameraTuner

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            if self.path == "/api/render":
                result = self.tuner.render(
                    specs=payload.get("specs"),
                    episode=payload.get("episode"),
                    frame=payload.get("frame"),
                )
            elif self.path == "/api/save":
                result = self.tuner.save()
            elif self.path == "/api/reload":
                result = self.tuner.load_saved_config()
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
                return
        except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # Keep the server usable when a renderer request fails.
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            return
        self._write_json(HTTPStatus.OK, result)


def main() -> None:
    args = parse_args()
    Handler.tuner = CameraTuner(args.trajectory, args.config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Camera tuner ready at http://{args.host}:{args.port}")
    print("Stop with Ctrl-C. Changes are written only after clicking '保存配置'.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        Handler.tuner.close()


if __name__ == "__main__":
    main()
