"""Offline 3D replay files for rl-experiment-site's Three.js viewer.

One recording is three files sharing a key (see ``docs/site/viewer3d.md``)::

  <key>.glb        per-episode scene: Panda links, peg, box with hole, table, ground
  <key>.traj.json  {metadata, static_poses, rows, frames}; frames[i] = {body: p7}
  <key>.json       one-line summary for the site's recording list

Mesh names are ``<body>__<part>`` and are expressed in the body frame, so the
viewer groups them by the prefix and applies ``frames[i][body]``.  Poses are
SAPIEN ``raw_pose`` seven-tuples ``[x, y, z, qw, qx, qy, qz]`` (``wxyz``).

The pure parts (GLB building, trajectory payload, HUD geometry, camera presets)
are CPU-testable; the ``collect_*`` / ``read_*`` helpers need a live ManiSkill
``PegInsertionSide`` env and import SAPIEN lazily.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

QUAT_ORDER = "wxyz"
# Bodies whose pose is constant within an episode go to ``static_poses``.
STATIC_BODIES = ("table", "ground", "box")
# Box walls are made semi-transparent by the viewer's X-ray toggle so the peg
# head is visible inside the hole.
XRAY_NAMES = tuple(f"box__wall{i}" for i in range(4))
XRAY_LABEL = "孔壁半透明透视"
# Site HUD (metadata.hud_fields / hud_progress): labels and units for the
# ``peg_hud_fields`` row keys, and insertion progress towards the success depth.
HUD_FIELDS = (
    {"key": "depth_mm", "label": "插入深度", "unit": "mm", "digits": 1},
    {"key": "success_depth_mm", "label": "成功深度", "unit": "mm", "digits": 1},
    {"key": "lateral_mm", "label": "径向偏差", "unit": "mm", "digits": 2},
    {"key": "dy_mm", "label": "偏差 y", "unit": "mm", "digits": 2},
    {"key": "dz_mm", "label": "偏差 z", "unit": "mm", "digits": 2},
    {"key": "aligned", "label": "对准孔口"},
    {"key": "success", "label": "已成功"},
)
HUD_PROGRESS = {"key": "depth_mm", "target_key": "success_depth_mm", "label": "插入进度"}
# PegInsertionSide-v1: success needs the peg head within 15 mm of the hole
# centre along the hole axis (``has_peg_inserted``).
SUCCESS_X_THRESHOLD_M = -0.015
# The Panda wrist RealSense mesh alone has ~1.3M vertices; parts above this are
# replaced by their bounding box to keep the per-episode GLB small.
MAX_PART_VERTICES = 200_000
_KEY_RE = re.compile(r"^[\w.-]+$")


@dataclass
class MeshPart:
    """One visual mesh in its body frame, with a linear-RGBA base colour."""

    body: str
    part: str
    vertices: np.ndarray
    faces: np.ndarray
    rgba: tuple[float, float, float, float]
    normals: np.ndarray | None = None
    metallic: float = 0.0
    roughness: float = 0.6

    @property
    def name(self) -> str:
        return f"{self.body}__{self.part}"


def validate_key(key: str) -> str:
    """Recording keys become file names and site URL path segments."""
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"recording key must match [\\w.-]+, got {key!r}")
    return key


def build_glb(parts: Sequence[MeshPart]) -> bytes:
    """Pack mesh parts into one GLB with node and mesh both named ``<body>__<part>``."""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    if not parts:
        raise ValueError("no mesh parts to export")
    names = [part.name for part in parts]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate mesh names: {duplicates}")

    scene = trimesh.Scene()
    for part in parts:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(part.vertices, dtype=np.float64),
            faces=np.asarray(part.faces, dtype=np.int64),
            vertex_normals=None if part.normals is None else np.asarray(part.normals, dtype=np.float64),
            process=False,
        )
        if part.normals is None:
            # Primitives (boxes) share corner vertices; split them so the
            # computed normals are flat instead of smoothed across edges.
            mesh.unmerge_vertices()
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                name=part.name,
                baseColorFactor=[float(c) for c in part.rgba],
                metallicFactor=float(part.metallic),
                roughnessFactor=float(part.roughness),
                doubleSided=True,
            )
        )
        scene.add_geometry(mesh, node_name=part.name, geom_name=part.name)
    return trimesh.exchange.gltf.export_glb(scene, include_normals=True)


def _rounded(p7: Sequence[float], ndigits: int = 5) -> list[float]:
    values = [round(float(v), ndigits) for v in p7]
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"pose must be 7 finite numbers, got {p7!r}")
    return values


class TrajectoryRecorder:
    """Accumulate per-step poses and HUD rows into the viewer's traj payload."""

    def __init__(self, dynamic_bodies: Sequence[str], static_poses: Mapping[str, Sequence[float]]):
        overlap = set(dynamic_bodies) & set(static_poses)
        if overlap:
            raise ValueError(f"bodies cannot be both dynamic and static: {sorted(overlap)}")
        self.dynamic_bodies = list(dynamic_bodies)
        self.static_poses = {body: _rounded(p7) for body, p7 in static_poses.items()}
        self.frames: list[dict[str, list[float]]] = []
        self.rows: list[dict[str, Any]] = []

    def capture(self, poses: Mapping[str, Sequence[float]], row: Mapping[str, Any]) -> None:
        missing = [body for body in self.dynamic_bodies if body not in poses]
        if missing:
            raise KeyError(f"missing poses for dynamic bodies: {missing}")
        self.frames.append({body: _rounded(poses[body]) for body in self.dynamic_bodies})
        self.rows.append(_clean_row(row))

    def payload(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("no frames captured")
        meta = dict(metadata)
        meta.setdefault("quat_order", QUAT_ORDER)
        meta["body_names"] = self.dynamic_bodies
        meta["steps"] = len(self.frames) - 1
        return {
            "metadata": meta,
            "static_poses": self.static_poses,
            "rows": self.rows,
            "frames": self.frames,
        }


def _clean_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        if isinstance(value, (bool, str)) or value is None:
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = round(float(value), 4) if isinstance(value, float) else value
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        else:
            raise TypeError(f"HUD field {key!r} must be a scalar, got {type(value).__name__}")
    return out


def peg_hud_fields(
    peg_head_at_hole: Sequence[float], *, peg_half_length: float, hole_radius: float
) -> dict[str, Any]:
    """HUD numbers from the peg head position in the hole frame (metres).

    The hole frame's x-axis is the insertion direction; the box spans
    ``x in [-L, L]`` with ``L`` equal to the peg half length, so the entrance
    face is at ``x = -L``.  ``depth_mm`` is how far the head is past that face
    and ``success_depth_mm`` is where ManiSkill starts counting success.
    """
    x, y, z = (float(v) for v in peg_head_at_hole)
    inside = abs(y) <= hole_radius and abs(z) <= hole_radius
    return {
        "depth_mm": (x + peg_half_length) * 1000.0,
        "success_depth_mm": (peg_half_length + SUCCESS_X_THRESHOLD_M) * 1000.0,
        "lateral_mm": math.hypot(y, z) * 1000.0,
        "dy_mm": y * 1000.0,
        "dz_mm": z * 1000.0,
        "aligned": inside,
    }


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_presets(
    hole_pose: Sequence[float], *, box_half_depth: float, tcp_position: Sequence[float]
) -> dict[str, dict[str, Any]]:
    """Per-episode camera presets; the hole is randomized every reconfigure."""
    hole_p = np.asarray(hole_pose[:3], dtype=np.float64)
    rot = quat_wxyz_to_matrix(hole_pose[3:7])
    axis = rot[:, 0]
    entrance = hole_p - axis * box_half_depth
    # Horizontal direction perpendicular to the hole axis (box only yaws).
    side = np.cross([0.0, 0.0, 1.0], axis)
    side = side / (np.linalg.norm(side) or 1.0)
    if side[0] < 0.0:
        side = -side  # look from the +x side; the Panda base sits at x = -0.615
    up = np.array([0.0, 0.0, 1.0])
    tcp = np.asarray(tcp_position, dtype=np.float64)

    def view(label: str, position: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        return {"label": label, "position": _vec(position), "target": _vec(target)}

    middle = 0.5 * (entrance + np.array([tcp[0], tcp[1], entrance[2]]))
    return {
        "wide": view("全景", np.array([0.45, -0.75, 0.55]), np.array([-0.1, 0.05, 0.1])),
        "hole_side": view("孔口侧视", entrance + side * 0.28 + up * 0.06, entrance),
        "hole_axis": view("沿孔轴", entrance - axis * 0.30 + side * 0.04 + up * 0.08, entrance),
        "top": view("俯视", middle + np.array([0.0, -0.05, 0.75]), middle),
        "gripper": view("夹爪特写", tcp + np.array([0.18, -0.18, 0.10]), tcp),
    }


def _vec(v: np.ndarray) -> list[float]:
    return [round(float(x), 4) for x in v]


def write_recording(
    output_dir: str | Path,
    key: str,
    glb: bytes,
    traj: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    """Write ``<key>.{glb,traj.json,json}``; the summary is written last as the ready marker."""
    validate_key(key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "glb": output_dir / f"{key}.glb",
        "traj": output_dir / f"{key}.traj.json",
        "summary": output_dir / f"{key}.json",
    }
    _atomic_write(paths["glb"], glb)
    _atomic_write(paths["traj"], json.dumps(traj, ensure_ascii=False, separators=(",", ":")).encode())
    full_summary = {"key": key, "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"), **summary}
    _atomic_write(paths["summary"], (json.dumps(full_summary, ensure_ascii=False, indent=2) + "\n").encode())
    return paths


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------- live env helpers


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _material_rgba(material: Any) -> tuple[float, float, float, float]:
    """Base colour, multiplied by the mean texel when the material is textured."""
    rgba = np.asarray(material.base_color, dtype=np.float64)
    texture = material.base_color_texture
    if texture is not None:
        try:
            texels = np.asarray(texture.download(), dtype=np.float64)
            mean = texels.reshape(-1, texels.shape[-1]).mean(axis=0)[:3] / 255.0
            if getattr(texture, "is_srgb", True):
                mean = _srgb_to_linear(mean)
            rgba[:3] = rgba[:3] * mean
        except Exception:  # noqa: BLE001 - colour is cosmetic; keep the base colour
            pass
    return tuple(float(v) for v in rgba)  # type: ignore[return-value]


def _bbox_part(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    return np.asarray(box.vertices), np.asarray(box.faces)


def _render_body_parts(entity: Any, body: str, part_prefix: str = "visual") -> list[MeshPart]:
    """Visual meshes of one SAPIEN entity, in the entity frame."""
    import sapien
    import trimesh

    render_body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if render_body is None:
        return []
    parts: list[MeshPart] = []
    for shape in render_body.render_shapes:
        local = shape.local_pose.to_transformation_matrix()
        kind = type(shape).__name__
        pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]] = []
        if kind == "RenderShapeTriangleMesh":
            scale = np.asarray(shape.scale, dtype=np.float64)
            for mesh_part in shape.parts:
                vertices = np.asarray(mesh_part.vertices, dtype=np.float64) * scale
                faces = np.asarray(mesh_part.triangles, dtype=np.int64)
                normals = np.asarray(mesh_part.get_vertex_normal(), dtype=np.float64)
                if normals.shape != vertices.shape or not np.isfinite(normals).all():
                    normals = None
                if len(vertices) > MAX_PART_VERTICES:
                    vertices, faces = _bbox_part(vertices)
                    normals = None
                pieces.append((vertices, faces, normals, mesh_part.material))
        elif kind == "RenderShapeBox":
            box = trimesh.creation.box(extents=2 * np.asarray(shape.half_size))
            pieces.append((np.asarray(box.vertices), np.asarray(box.faces), None, shape.material))
        elif kind in ("RenderShapeCapsule", "RenderShapeCylinder", "RenderShapeSphere"):
            from mani_skill.utils.geometry.trimesh_utils import get_render_shape_meshes

            for mesh in get_render_shape_meshes(shape):
                mesh.apply_transform(np.linalg.inv(local))  # helper already applied local_pose
                pieces.append((np.asarray(mesh.vertices), np.asarray(mesh.faces), None, shape.material))
        else:
            continue  # planes: the viewer has its own background
        for vertices, faces, normals, material in pieces:
            homogeneous = np.c_[vertices, np.ones(len(vertices))]
            vertices = (homogeneous @ local.T)[:, :3]
            if normals is not None:
                normals = normals @ local[:3, :3].T
            parts.append(
                MeshPart(
                    body=body,
                    part=f"{part_prefix}{len(parts)}",
                    vertices=vertices,
                    faces=faces,
                    normals=normals,
                    rgba=_material_rgba(material),
                    metallic=float(getattr(material, "metallic", 0.0) or 0.0),
                    roughness=float(getattr(material, "roughness", 0.6) or 0.6),
                )
            )
    return parts


def _entity(handle: Any) -> Any:
    """ManiSkill Link/Actor wrappers hold per-sub-scene objects; we record env 0."""
    obj = handle._objs[0]
    return getattr(obj, "entity", obj)


def body_handles(unwrapped: Any) -> dict[str, Any]:
    """Body name -> ManiSkill Link/Actor (anything with ``.pose.raw_pose``)."""
    handles: dict[str, Any] = {}
    for link in unwrapped.agent.robot.get_links():
        handles[link.name] = link
    handles["peg"] = unwrapped.peg
    handles["box"] = unwrapped.box
    handles["table"] = unwrapped.table_scene.table
    ground = getattr(unwrapped.table_scene, "ground", None)
    if ground is not None:
        handles["ground"] = ground
    return handles


def collect_mesh_parts(unwrapped: Any) -> tuple[list[MeshPart], list[str]]:
    """All visual meshes plus the names of bodies that move during the episode."""
    parts: list[MeshPart] = []
    dynamic: list[str] = []
    for body, handle in body_handles(unwrapped).items():
        prefix = {"peg": "part", "box": "wall", "table": "top", "ground": "plane"}.get(body, "visual")
        body_parts = _render_body_parts(_entity(handle), body, prefix)
        if not body_parts:
            continue
        parts.extend(body_parts)
        if body not in STATIC_BODIES:
            dynamic.append(body)
    if "peg" in dynamic:
        # Peg visuals are head (red) then tail (white) in ManiSkill's builder.
        for part, name in zip([p for p in parts if p.body == "peg"], ("head", "tail")):
            part.part = name
    return parts, dynamic


def _first(raw: Any) -> list[float]:
    array = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
    return array.reshape(-1, array.shape[-1])[0].tolist()


def read_poses(unwrapped: Any, bodies: Sequence[str]) -> dict[str, list[float]]:
    handles = body_handles(unwrapped)
    return {body: _first(handles[body].pose.raw_pose) for body in bodies}


def read_peg_hud(unwrapped: Any) -> dict[str, Any]:
    success, head_at_hole = unwrapped.has_peg_inserted()
    fields = peg_hud_fields(
        _first(head_at_hole),
        peg_half_length=float(_first(unwrapped.peg_half_sizes)[0]),
        hole_radius=float(_first(unwrapped.box_hole_radii[:, None])[0]),
    )
    fields["success"] = bool(_first(success[:, None])[0])
    return fields


def episode_geometry(unwrapped: Any) -> dict[str, Any]:
    half = _first(unwrapped.peg_half_sizes)
    return {
        "peg_half_length_m": round(float(half[0]), 5),
        "peg_radius_m": round(float(half[1]), 5),
        "hole_radius_m": round(float(_first(unwrapped.box_hole_radii[:, None])[0]), 5),
        "clearance_mm": round(float(unwrapped._clearance) * 1000.0, 3),
    }


def episode_camera_presets(unwrapped: Any) -> dict[str, dict[str, Any]]:
    return camera_presets(
        _first(unwrapped.box_hole_pose.raw_pose),
        box_half_depth=float(_first(unwrapped.peg_half_sizes)[0]),
        tcp_position=_first(unwrapped.agent.tcp.pose.raw_pose)[:3],
    )
