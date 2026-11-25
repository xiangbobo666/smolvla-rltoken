"""CPU tests for the 3D replay export (no simulator, no GPU)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from smolvla_rltoken import viewer3d

trimesh = pytest.importorskip("trimesh")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _box_part(body: str, part: str, rgba=(0.8, 0.2, 0.1, 1.0)) -> viewer3d.MeshPart:
    box = trimesh.creation.box(extents=[0.2, 0.04, 0.04])
    return viewer3d.MeshPart(body=body, part=part, vertices=box.vertices, faces=box.faces, rgba=rgba)


def _mesh_part_with_normals() -> viewer3d.MeshPart:
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    return viewer3d.MeshPart(
        body="panda_hand",
        part="visual0",
        vertices=sphere.vertices,
        faces=sphere.faces,
        normals=sphere.vertex_normals,
        rgba=(0.9, 0.92, 0.93, 1.0),
    )


def test_glb_round_trips_with_body_part_names(tmp_path: Path):
    parts = [_box_part("peg", "head"), _box_part("peg", "tail", (0.85, 0.92, 0.95, 1.0)),
             _box_part("box", "wall0"), _mesh_part_with_normals()]
    path = tmp_path / "scene.glb"
    path.write_bytes(viewer3d.build_glb(parts))

    scene = trimesh.load(path)
    assert sorted(scene.geometry) == sorted(p.name for p in parts)
    assert sorted(scene.graph.nodes_geometry) == sorted(p.name for p in parts)
    # Every mesh name splits into a viewer body group on the first "__".
    assert {name.split("__", 1)[0] for name in scene.geometry} == {"peg", "box", "panda_hand"}
    head = scene.geometry["peg__head"]
    np.testing.assert_allclose(head.bounds, [[-0.1, -0.02, -0.02], [0.1, 0.02, 0.02]], atol=1e-6)
    np.testing.assert_allclose(head.visual.material.baseColorFactor[:3], [204, 51, 25], atol=1)


def test_glb_rejects_duplicate_names_and_empty_scene():
    with pytest.raises(ValueError, match="duplicate"):
        viewer3d.build_glb([_box_part("peg", "head"), _box_part("peg", "head")])
    with pytest.raises(ValueError, match="no mesh"):
        viewer3d.build_glb([])


def test_trajectory_payload_matches_viewer_protocol():
    recorder = viewer3d.TrajectoryRecorder(["panda_hand", "peg"], {"box": [0, 0.3, 0.1, 1, 0, 0, 0]})
    for step in range(3):
        recorder.capture(
            {"panda_hand": [0.1 * step, 0, 0.2, 1, 0, 0, 0], "peg": [0, 0, 0.02, 0.7071, 0, 0, 0.7071],
             "box": [9, 9, 9, 1, 0, 0, 0]},
            {"step": step, "depth_mm": np.float32(1.5) * step, "success": np.bool_(step == 2), "source": "demo"},
        )
    payload = recorder.payload({"step_dt": 0.05, "key": "k"})

    assert set(payload) == {"metadata", "static_poses", "rows", "frames"}
    assert payload["metadata"]["quat_order"] == "wxyz"
    assert payload["metadata"]["body_names"] == ["panda_hand", "peg"]
    assert payload["metadata"]["steps"] == 2
    assert len(payload["rows"]) == len(payload["frames"]) == 3
    assert set(payload["frames"][1]) == {"panda_hand", "peg"}  # static bodies are not repeated per frame
    assert payload["rows"][2] == {"step": 2, "depth_mm": 3.0, "success": True, "source": "demo"}
    json.dumps(payload)  # numpy scalars were unwrapped


def test_trajectory_recorder_validates_input():
    with pytest.raises(ValueError, match="both dynamic and static"):
        viewer3d.TrajectoryRecorder(["peg"], {"peg": [0, 0, 0, 1, 0, 0, 0]})
    recorder = viewer3d.TrajectoryRecorder(["peg"], {})
    with pytest.raises(KeyError, match="peg"):
        recorder.capture({}, {"step": 0})
    with pytest.raises(ValueError, match="7 finite"):
        recorder.capture({"peg": [0, 0, 0, 1, 0, 0]}, {"step": 0})
    with pytest.raises(TypeError, match="scalar"):
        recorder.capture({"peg": [0, 0, 0, 1, 0, 0, 0]}, {"xyz": [1, 2, 3]})
    with pytest.raises(ValueError, match="no frames"):
        viewer3d.TrajectoryRecorder(["peg"], {}).payload({})


def test_peg_hud_fields_follow_maniskill_success_geometry():
    half_length, hole_radius = 0.1, 0.018
    at_threshold = viewer3d.peg_hud_fields([-0.015, 0.003, -0.004], peg_half_length=half_length,
                                           hole_radius=hole_radius)
    assert at_threshold["depth_mm"] == pytest.approx(85.0)
    assert at_threshold["success_depth_mm"] == pytest.approx(85.0)
    assert at_threshold["lateral_mm"] == pytest.approx(5.0)
    assert at_threshold["aligned"] is True

    outside = viewer3d.peg_hud_fields([-0.2, 0.03, 0.0], peg_half_length=half_length, hole_radius=hole_radius)
    assert outside["depth_mm"] == pytest.approx(-100.0)
    assert outside["aligned"] is False


def test_camera_presets_look_at_the_hole_entrance_from_away_from_the_robot():
    yaw = math.pi / 2  # hole axis along +y
    hole_pose = [0.0, 0.3, 0.1, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    presets = viewer3d.camera_presets(hole_pose, box_half_depth=0.1, tcp_position=[0.0, 0.0, 0.2])

    assert {"wide", "hole_side", "hole_axis", "top", "gripper"} <= set(presets)
    np.testing.assert_allclose(presets["hole_side"]["target"], [0.0, 0.2, 0.1], atol=1e-4)
    assert presets["hole_side"]["position"][0] > 0.2  # +x side, not behind the Panda base
    np.testing.assert_allclose(presets["gripper"]["target"], [0.0, 0.0, 0.2], atol=1e-4)
    for view in presets.values():
        assert view["label"] and len(view["position"]) == len(view["target"]) == 3


def test_quat_wxyz_to_matrix_matches_known_rotation():
    rot = viewer3d.quat_wxyz_to_matrix([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    np.testing.assert_allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_write_recording_writes_three_files_with_summary_last(tmp_path: Path):
    recorder = viewer3d.TrajectoryRecorder(["peg"], {})
    recorder.capture({"peg": [0, 0, 0, 1, 0, 0, 0]}, {"step": 0})
    glb = viewer3d.build_glb([_box_part("peg", "head")])
    paths = viewer3d.write_recording(tmp_path, "demo-mp-ep0000-seed2", glb, recorder.payload({}), {"ok": True})

    assert paths["glb"].read_bytes()[:4] == b"glTF"
    assert json.loads(paths["traj"].read_text())["frames"] == [{"peg": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}]
    summary = json.loads(paths["summary"].read_text())
    assert summary["key"] == "demo-mp-ep0000-seed2" and summary["ok"] is True and "created" in summary
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="recording key"):
        viewer3d.write_recording(tmp_path, "../escape", glb, recorder.payload({}), {})


def _load_script():
    spec = importlib.util.spec_from_file_location("record_viewer3d", REPO_ROOT / "scripts/record_viewer3d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_demo(tmp_path: Path) -> Path:
    demo_dir = tmp_path / "motionplanning"
    demo_dir.mkdir()
    (demo_dir / "trajectory.h5").write_bytes(b"")
    (demo_dir / "trajectory.json").write_text(json.dumps({
        "env_info": {"env_id": "PegInsertionSide-v1"},
        "episodes": [{"episode_id": 3, "control_mode": "pd_joint_pos", "elapsed_steps": 10,
                      "reset_kwargs": {"seed": 42}, "success": True}],
    }))
    return demo_dir / "trajectory.h5"


def test_script_check_resolves_demo_seed_and_key(tmp_path: Path):
    script = _load_script()
    demo = _write_demo(tmp_path)
    args = script.parse_args(["--demo", str(demo), "--episode-id", "3", "--output-dir", str(tmp_path / "out")])
    check = script.check_inputs(args)
    assert check["ok"], check["errors"]
    assert check["seed"] == 42 and check["key"] == "demo-mp-ep0003-seed42"
    assert args.sim_backend == "physx_cpu"

    missing = script.check_inputs(script.parse_args(["--demo", str(demo), "--episode-id", "7"]))
    assert not missing["ok"] and "episode_id 7" in missing["errors"][0]

    # rl demos carry no reset seed, only episode_seed.
    assert script.episode_seed({"episode_seed": 9, "reset_kwargs": {"options": {}}}) == 9


def test_script_check_refuses_to_overwrite_and_reports_missing_checkpoint(tmp_path: Path):
    script = _load_script()
    demo = _write_demo(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "demo-mp-ep0003-seed42.json").write_text("{}")
    args = script.parse_args(["--demo", str(demo), "--episode-id", "3", "--output-dir", str(out)])
    assert any("already exists" in e for e in script.check_inputs(args)["errors"])
    args.overwrite = True
    assert script.check_inputs(args)["ok"]

    policy = script.parse_args(["--checkpoint", str(tmp_path / "missing"), "--output-dir", str(out)])
    check = script.check_inputs(policy)
    assert policy.sim_backend == "physx_cuda" and check["key"] == "sft-seed10000"
    assert not check["ok"] and any("missing checkpoint file" in e for e in check["errors"])
