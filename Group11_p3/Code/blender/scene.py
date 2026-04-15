"""
blender/scene.py
================
Master Blender script. Invoked headlessly by run_blender.sh:

  blender --background --python blender/scene.py -- --seq Seq1 --config config.yaml

For each frame in the sequence:
  1. Load the per-frame detection JSON
  2. Set up / update the Blender scene
  3. Place 3D assets (cars, pedestrians, stop signs)
  4. Draw lane geometry
  5. Set traffic light material colors
  6. Render to PNG
After all frames: stitch PNGs → MP4.
"""

import sys
import argparse
from pathlib import Path
import re
import math
from statistics import median

# sys.path must include Code/ so our utils and blender packages are importable
code_dir = Path(__file__).resolve().parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from utils.io_utils import load_detection_json, list_frame_jsons, load_config
from blender.render  import render_frame, frames_to_video
from blender.camera import setup_camera, update_chase_camera
from blender.assets import AssetLibrary
from blender.lanes import LaneRenderer

# bpy is only available inside Blender — guarded below
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def parse_args():
    """
    Parse arguments passed after the '--' separator in the Blender command.
    sys.argv looks like: ['blender', ..., '--', '--scene', 'scene1', ...]
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True, help="Sequence name, e.g. scene1")
    parser.add_argument("--cam",    required=True, help="Camera: front | back | left_repeater | right_repeater")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug",  action="store_true",   help="Print extra diagnostics per frame")
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Skip detection JSONs with frame index less than this value (must be >= 0)",
    )
    return parser.parse_args(argv)


def _frame_idx_from_json_path(json_path: Path):
    """Extract frame index from file names like frame_000123.json."""
    match = re.search(r"frame_(\d+)\.json$", json_path.name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_cfg_path(value, base_dir: Path) -> str:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _normalize_blender_cfg_paths(cfg: dict, config_path: Path):
    """
    Resolve relative paths in the loaded config against config file location.
    Prevents cwd-dependent behavior when invoking run_blender.sh from anywhere.
    """
    base_dir = config_path.parent.resolve()
    cfg.setdefault("_meta", {})["config_dir"] = str(base_dir)

    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_cfg_path(value, base_dir)

    ped_cfg = cfg.get("blender", {}).get("pedestrian", {})
    smpl_pkl = ped_cfg.get("smpl_pkl")
    if isinstance(smpl_pkl, str) and smpl_pkl.strip():
        ped_cfg["smpl_pkl"] = _resolve_cfg_path(smpl_pkl, base_dir)


def setup_scene(cfg: dict):
    """
    One-time Blender scene setup — call this once before the frame loop.
 
    - Delete everything Blender puts in a default scene (cube, camera, light)
    - Set render resolution and engine from config
    - Set background to solid black (Tesla-style dark environment)
    - film_transparent = True so we can later composite onto the real frame
    - EEVEE samples not applicable; CYCLES sample count set from config
    """
    # Delete all default objects (cube, camera, light)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
 
    scene = bpy.context.scene
    scene.render.engine       = cfg["blender"]["render_engine"]        # "CYCLES" or "EEVEE"
    scene.render.resolution_x = cfg["blender"]["resolution"][0]        # 1280
    scene.render.resolution_y = cfg["blender"]["resolution"][1]        # 960

    # make a light
    lamp_data = bpy.data.lights.new(name="TestSun", type="SUN")
    lamp_obj  = bpy.data.objects.new("TestSun", lamp_data)
    bpy.context.collection.objects.link(lamp_obj)
    lamp_obj.location = (5, 5, 10)

    # make ground plane
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.location.z = 0       # slightly below 0 so objects sit on z=0
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)  # dark gray
    ground.data.materials.append(mat)

    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True                               # RGBA output, background = alpha 0
 
    if cfg["blender"]["render_engine"] == "CYCLES":
        scene.cycles.samples = cfg["blender"]["samples"]
 
    # Black world background (visible where film_transparent is off, and for ambient)
    world = bpy.data.worlds.new("EV_World")
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value    = (0.0, 0.0, 0.0, 1.0)   # black
    bg_node.inputs["Strength"].default_value = 0.0                     # no ambient light from world
    scene.world = world
 
    print("[scene] setup_scene complete — "
          f"engine={scene.render.engine}  "
          f"res={scene.render.resolution_x}x{scene.render.resolution_y}")


def _lane_overlaps_arrow_bbox(lane: dict, arrow_bbox: list, pad_px: float = 6.0) -> bool:
    """Return True when lane polyline intersects an expanded bbox."""
    points = lane.get("points") or []
    if len(points) < 2 or not isinstance(arrow_bbox, (list, tuple)) or len(arrow_bbox) != 4:
        return False

    x1, y1, x2, y2 = [float(v) for v in arrow_bbox]
    if x2 <= x1 or y2 <= y1:
        return False

    x1 -= pad_px
    y1 -= pad_px
    x2 += pad_px
    y2 += pad_px

    def _point_in_rect(u: float, v: float) -> bool:
        return x1 <= u <= x2 and y1 <= v <= y2

    def _orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _on_segment(ax, ay, bx, by, cx, cy):
        return min(ax, bx) - 1e-6 <= cx <= max(ax, bx) + 1e-6 and min(ay, by) - 1e-6 <= cy <= max(ay, by) + 1e-6

    def _segments_intersect(a, b, c, d):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d

        o1 = _orient(ax, ay, bx, by, cx, cy)
        o2 = _orient(ax, ay, bx, by, dx, dy)
        o3 = _orient(cx, cy, dx, dy, ax, ay)
        o4 = _orient(cx, cy, dx, dy, bx, by)

        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True

        if abs(o1) <= 1e-6 and _on_segment(ax, ay, bx, by, cx, cy):
            return True
        if abs(o2) <= 1e-6 and _on_segment(ax, ay, bx, by, dx, dy):
            return True
        if abs(o3) <= 1e-6 and _on_segment(cx, cy, dx, dy, ax, ay):
            return True
        if abs(o4) <= 1e-6 and _on_segment(cx, cy, dx, dy, bx, by):
            return True

        return False

    rect_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]

    valid_pts = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        valid_pts.append((float(pt[0]), float(pt[1])))

    if len(valid_pts) < 2:
        return False

    for u, v in valid_pts:
        if _point_in_rect(u, v):
            return True

    for i in range(len(valid_pts) - 1):
        p0 = valid_pts[i]
        p1 = valid_pts[i + 1]

        sx1 = min(p0[0], p1[0])
        sy1 = min(p0[1], p1[1])
        sx2 = max(p0[0], p1[0])
        sy2 = max(p0[1], p1[1])
        if sx2 < x1 or sx1 > x2 or sy2 < y1 or sy1 > y2:
            continue

        if _point_in_rect(p0[0], p0[1]) or _point_in_rect(p1[0], p1[1]):
            return True

        for e0, e1 in rect_edges:
            if _segments_intersect(p0, p1, e0, e1):
                return True

    return False


def _lane_overlaps_ground_arrows(lane: dict, ground_arrows: list) -> bool:
    """Return True when a lane should be suppressed due to ground-arrow overlap."""
    if not ground_arrows:
        return False

    for arrow in ground_arrows:
        bbox = arrow.get("bbox") if isinstance(arrow, dict) else None
        if _lane_overlaps_arrow_bbox(lane, bbox):
            return True

    return False


def _lane_overlaps_ground_text(lane: dict, ground_text_markings: list) -> bool:
    """Return True when a lane should be suppressed due to ONLY text overlap."""
    if not ground_text_markings:
        return False

    for marking in ground_text_markings:
        if not isinstance(marking, dict):
            continue
        if not bool(marking.get("has_only_letters", False)):
            continue
        if float(marking.get("ocr_confidence", 0.0) or 0.0) < 0.6:
            continue

        bbox = _only_render_bbox(marking)
        if _lane_overlaps_arrow_bbox(lane, bbox, pad_px=0.0):
            return True

    return False


def _point_in_bbox(pt, bbox, pad_px: float = 0.0) -> bool:
    """Return True when a 2D point lies within a padded bbox."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return False
    x1 -= pad_px
    y1 -= pad_px
    x2 += pad_px
    y2 += pad_px
    u, v = float(pt[0]), float(pt[1])
    return x1 <= u <= x2 and y1 <= v <= y2


def _segment_intersects_bbox(p0, p1, bbox, pad_px: float = 3.0) -> bool:
    """Return True when a line segment intersects a padded bbox."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return False

    x1 -= pad_px
    y1 -= pad_px
    x2 += pad_px
    y2 += pad_px

    def _orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _on_segment(ax, ay, bx, by, cx, cy):
        return min(ax, bx) - 1e-6 <= cx <= max(ax, bx) + 1e-6 and min(ay, by) - 1e-6 <= cy <= max(ay, by) + 1e-6

    def _segments_intersect(a, b, c, d):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d

        o1 = _orient(ax, ay, bx, by, cx, cy)
        o2 = _orient(ax, ay, bx, by, dx, dy)
        o3 = _orient(cx, cy, dx, dy, ax, ay)
        o4 = _orient(cx, cy, dx, dy, bx, by)

        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True

        if abs(o1) <= 1e-6 and _on_segment(ax, ay, bx, by, cx, cy):
            return True
        if abs(o2) <= 1e-6 and _on_segment(ax, ay, bx, by, dx, dy):
            return True
        if abs(o3) <= 1e-6 and _on_segment(cx, cy, dx, dy, ax, ay):
            return True
        if abs(o4) <= 1e-6 and _on_segment(cx, cy, dx, dy, bx, by):
            return True

        return False

    if _point_in_bbox(p0, (x1, y1, x2, y2), 0.0) or _point_in_bbox(p1, (x1, y1, x2, y2), 0.0):
        return True

    sx1 = min(float(p0[0]), float(p1[0]))
    sy1 = min(float(p0[1]), float(p1[1]))
    sx2 = max(float(p0[0]), float(p1[0]))
    sy2 = max(float(p0[1]), float(p1[1]))
    if sx2 < x1 or sx1 > x2 or sy2 < y1 or sy1 > y2:
        return False

    rect_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for e0, e1 in rect_edges:
        if _segments_intersect(p0, p1, e0, e1):
            return True

    return False


def _collect_lane_block_bboxes(frame_data: dict) -> list:
    """Collect bboxes that should mask lane rendering for this frame."""
    bboxes = []

    for arrow in frame_data.get("ground_arrows", []):
        if not isinstance(arrow, dict):
            continue
        bbox = arrow.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bboxes.append([float(v) for v in bbox])

    for marking in frame_data.get("ground_text_markings", []):
        if not isinstance(marking, dict):
            continue
        if not bool(marking.get("has_only_letters", False)):
            continue
        if float(marking.get("ocr_confidence", 0.0) or 0.0) < 0.6:
            continue
        bbox = _only_render_bbox(marking)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bboxes.append([float(v) for v in bbox])

    return bboxes


def _split_lane_points_by_bboxes(points: list, bboxes: list, pad_px: float = 0.0) -> list:
    """Split lane polyline into non-overlapping fragments outside blocked bboxes."""
    valid_pts = []
    for pt in points or []:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            valid_pts.append([float(pt[0]), float(pt[1])])

    if len(valid_pts) < 2:
        return []

    if not bboxes:
        return [valid_pts]

    fragments = []
    current = []

    for i in range(len(valid_pts) - 1):
        p0 = valid_pts[i]
        p1 = valid_pts[i + 1]

        p0_blocked = any(_point_in_bbox(p0, b, pad_px) for b in bboxes)
        p1_blocked = any(_point_in_bbox(p1, b, pad_px) for b in bboxes)
        seg_blocked = any(_segment_intersects_bbox(p0, p1, b, pad_px) for b in bboxes)

        if not p0_blocked and not seg_blocked and not current:
            current = [p0]

        if p0_blocked or p1_blocked or seg_blocked:
            if len(current) >= 2:
                fragments.append(current)
            current = []
            continue

        if not current:
            current = [p0]
        if current[-1] != p1:
            current.append(p1)

    if len(current) >= 2:
        fragments.append(current)

    return fragments


def _only_render_bbox(marking: dict):
    """Approximate rendered ONLY span from bbox + only_letter_hits."""
    bbox = marking.get("bbox") if isinstance(marking, dict) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2]

    hits_raw = str(marking.get("only_letter_hits", "") or "").upper()
    hits = {ch for ch in hits_raw if ch in {"O", "N", "L", "Y"}}
    if not hits:
        return [x1, y1, x2, y2]

    target = "ONLY"
    present_indices = [i for i, ch in enumerate(target) if ch in hits]
    if not present_indices:
        return [x1, y1, x2, y2]

    first_idx = min(present_indices)
    last_idx = max(present_indices)
    observed_span = max(last_idx - first_idx + 1, 1)

    bbox_w = max(x2 - x1, 1.0)
    est_char_w = bbox_w / float(observed_span)

    render_left = x1 - first_idx * est_char_w
    render_right = render_left + 4.0 * est_char_w

    # Keep the adjusted box in a sane range near the original detection.
    max_extra = 2.0 * bbox_w
    render_left = max(render_left, x1 - max_extra)
    render_right = min(render_right, x2 + max_extra)

    return [render_left, y1, render_right, y2]


def _blender_motion_cfg(cfg: dict) -> dict:
    motion_cfg = cfg.get("blender", {}).get("motion_state", {})
    return {
        "enabled": bool(motion_cfg.get("enabled", True)),
        "history_frames": max(2, int(motion_cfg.get("history_frames", 5))),
        "min_history_frames": max(2, int(motion_cfg.get("min_history_frames", 3))),
        "ema_alpha": max(0.0, min(1.0, float(motion_cfg.get("ema_alpha", 0.35)))),
        "jitter_speed_mps": max(0.0, float(motion_cfg.get("jitter_speed_mps", 0.20))),
        "moving_speed_mps": max(0.0, float(motion_cfg.get("moving_speed_mps", 0.70))),
        "parked_speed_mps": max(0.0, float(motion_cfg.get("parked_speed_mps", 0.35))),
        "move_confirm_frames": max(1, int(motion_cfg.get("move_confirm_frames", 3))),
        "park_confirm_frames": max(1, int(motion_cfg.get("park_confirm_frames", 4))),
        "min_forward_mps": max(0.0, float(motion_cfg.get("min_forward_mps", 0.15))),
        "lateral_min_mps": max(0.0, float(motion_cfg.get("lateral_min_mps", 0.25))),
        "max_lateral_to_forward_ratio": max(
            0.0, float(motion_cfg.get("max_lateral_to_forward_ratio", 1.2))
        ),
        "max_unseen_frames": max(1, int(motion_cfg.get("max_unseen_frames", 20))),
        "anon_assoc_max_dist_m": max(0.0, float(motion_cfg.get("anon_assoc_max_dist_m", 1.6))),
        "debug": bool(motion_cfg.get("debug", False)),
    }


def _default_motion_state() -> dict:
    return {
        "last_xy": None,
        "last_frame": None,
        "age": 0,
        "last_heading": None,
        "raw_vx": 0.0,
        "raw_vy": 0.0,
        "ema_vx": 0.0,
        "ema_vy": 0.0,
        "is_moving": False,
        "move_count": 0,
        "park_count": 0,
        "was_observed": False,
    }


def _json_to_blender_xy(position_3d):
    if not isinstance(position_3d, (list, tuple)) or len(position_3d) < 3:
        return None
    try:
        px = float(position_3d[0])
        py = float(position_3d[1])
        pz = float(position_3d[2])
    except Exception:
        return None
    return (px, pz), -py


def _heading_to_blender_forward(heading_rad):
    if heading_rad is None:
        return None
    try:
        heading = float(heading_rad)
    except Exception:
        return None
    yaw = -(heading - math.pi / 2.0)
    return (math.sin(yaw), math.cos(yaw))


def _vehicle_dedupe_cfg(cfg: dict) -> dict:
    dedupe_cfg = cfg.get("blender", {}).get("vehicle_dedupe", {})
    return {
        "enabled": bool(dedupe_cfg.get("enabled", True)),
        "max_xy_dist_m": max(0.0, float(dedupe_cfg.get("max_xy_dist_m", 0.85))),
        "max_depth_delta_m": max(0.0, float(dedupe_cfg.get("max_depth_delta_m", 2.5))),
        "bbox_iou_thresh": max(0.0, min(1.0, float(dedupe_cfg.get("bbox_iou_thresh", 0.92)))),
    }


def _bbox_area(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return 0.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _bbox_iou(b0, b1) -> float:
    if not isinstance(b0, (list, tuple)) or not isinstance(b1, (list, tuple)):
        return 0.0
    if len(b0) != 4 or len(b1) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in b0]
        bx1, by1, bx2, by2 = [float(v) for v in b1]
    except Exception:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _vehicle_keep_score(vehicle: dict) -> float:
    score = 0.0
    if vehicle.get("track_id") is not None:
        score += 100.0
    if not bool(vehicle.get("is_held", False)):
        score += 20.0
    if vehicle.get("is_class_smoothed", False):
        score += 8.0

    cls = str(vehicle.get("class", "")).lower()
    if cls and cls not in {"car", "truck", "bus"}:
        score += 5.0

    score += min(4.0, _bbox_area(vehicle.get("bbox")) / 12000.0)
    return score


def _vehicles_are_duplicate(v0: dict, v1: dict, dedupe_cfg: dict) -> bool:
    xy_z0 = _json_to_blender_xy(v0.get("position_3d"))
    xy_z1 = _json_to_blender_xy(v1.get("position_3d"))
    close_in_3d = False
    if xy_z0 is not None and xy_z1 is not None:
        (x0, y0), z0 = xy_z0
        (x1, y1), z1 = xy_z1
        d_xy = math.hypot(x0 - x1, y0 - y1)
        d_z = abs(z0 - z1)
        close_in_3d = (
            d_xy <= dedupe_cfg["max_xy_dist_m"]
            and d_z <= dedupe_cfg["max_depth_delta_m"]
        )

    bbox_iou = _bbox_iou(v0.get("bbox"), v1.get("bbox"))
    return close_in_3d or (bbox_iou >= dedupe_cfg["bbox_iou_thresh"])


def _dedupe_vehicles_for_render(vehicles: list, cfg: dict):
    dedupe_cfg = _vehicle_dedupe_cfg(cfg)
    if not dedupe_cfg["enabled"] or len(vehicles) <= 1:
        return vehicles, 0

    ranked = sorted(vehicles, key=_vehicle_keep_score, reverse=True)
    kept = []
    dropped = 0

    for v in ranked:
        is_dup = False
        for prev in kept:
            if _vehicles_are_duplicate(v, prev, dedupe_cfg):
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(v)

    return kept, dropped


def _associate_anonymous_track_ids(vehicles: list, frame_idx: int, motion_ctx: dict) -> None:
    cfg = motion_ctx["cfg"]
    anon_max_dist = cfg["anon_assoc_max_dist_m"]
    anon_tracks = motion_ctx["anon_tracks"]
    used_anon_ids = set()

    for vehicle in vehicles:
        if vehicle.get("track_id") is not None:
            continue
        xy_z = _json_to_blender_xy(vehicle.get("position_3d"))
        if xy_z is None:
            continue
        xy, _ = xy_z

        best_id = None
        best_dist = float("inf")
        for anon_id, state in anon_tracks.items():
            if anon_id in used_anon_ids:
                continue
            if (frame_idx - int(state.get("last_frame", -999))) > 1:
                continue
            prev_xy = state.get("last_xy")
            if prev_xy is None:
                continue
            dx = xy[0] - prev_xy[0]
            dy = xy[1] - prev_xy[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_id = anon_id

        if best_id is None or best_dist > anon_max_dist:
            best_id = motion_ctx["next_anon_id"]
            motion_ctx["next_anon_id"] += 1

        used_anon_ids.add(best_id)
        vehicle["__motion_track_id"] = f"anon-{best_id}"
        vehicle["__motion_low_confidence"] = True


def _estimate_dominant_flow(vehicles: list, frame_idx: int, motion_ctx: dict):
    vx_samples = []
    vy_samples = []

    for vehicle in vehicles:
        track_key = vehicle.get("__motion_track_id")
        if track_key is None:
            continue
        state = motion_ctx["tracks"].get(track_key)
        if state is None:
            continue
        prev_xy = state.get("last_xy")
        prev_frame = state.get("last_frame")
        if prev_xy is None or prev_frame is None:
            continue

        frame_delta = max(1, frame_idx - int(prev_frame))
        if frame_delta > 2:
            continue

        xy_z = _json_to_blender_xy(vehicle.get("position_3d"))
        if xy_z is None:
            continue
        xy, _ = xy_z
        dt = frame_delta / max(1e-6, motion_ctx["fps"])

        raw_vx = (xy[0] - prev_xy[0]) / max(1e-6, dt)
        raw_vy = (xy[1] - prev_xy[1]) / max(1e-6, dt)

        if state.get("age", 0) < 2:
            continue
        if math.hypot(raw_vx, raw_vy) > 20.0:
            continue

        vx_samples.append(raw_vx)
        vy_samples.append(raw_vy)

    if len(vx_samples) < 3:
        return 0.0, 0.0
    return float(median(vx_samples)), float(median(vy_samples))


def _update_single_motion_state(vehicle: dict, frame_idx: int, dominant_flow: tuple, motion_ctx: dict) -> None:
    cfg = motion_ctx["cfg"]
    track_key = vehicle.get("__motion_track_id")
    if track_key is None:
        vehicle["__motion_is_moving"] = False
        vehicle["__motion_forward_dir"] = None
        return

    state = motion_ctx["tracks"].setdefault(track_key, _default_motion_state())
    xy_z = _json_to_blender_xy(vehicle.get("position_3d"))
    if xy_z is None:
        vehicle["__motion_is_moving"] = False
        vehicle["__motion_forward_dir"] = None
        return

    xy, _ = xy_z
    prev_xy = state.get("last_xy")
    prev_frame = state.get("last_frame")
    frame_delta = 1 if prev_frame is None else max(1, frame_idx - int(prev_frame))
    dt = frame_delta / max(1e-6, motion_ctx["fps"])

    raw_vx = 0.0
    raw_vy = 0.0
    if prev_xy is not None and frame_delta <= 2:
        raw_vx = (xy[0] - prev_xy[0]) / max(1e-6, dt)
        raw_vy = (xy[1] - prev_xy[1]) / max(1e-6, dt)

    residual_vx = raw_vx - dominant_flow[0]
    residual_vy = raw_vy - dominant_flow[1]

    alpha = cfg["ema_alpha"]
    state["ema_vx"] = alpha * residual_vx + (1.0 - alpha) * float(state.get("ema_vx", 0.0))
    state["ema_vy"] = alpha * residual_vy + (1.0 - alpha) * float(state.get("ema_vy", 0.0))
    speed = math.hypot(state["ema_vx"], state["ema_vy"])

    heading_forward = _heading_to_blender_forward(vehicle.get("heading_rad"))
    forward_comp = None
    lateral_comp = None
    hard_reject = False

    if heading_forward is not None:
        fx, fy = heading_forward
        lx, ly = -fy, fx
        forward_comp = state["ema_vx"] * fx + state["ema_vy"] * fy
        lateral_comp = state["ema_vx"] * lx + state["ema_vy"] * ly

        if forward_comp <= 0.0:
            hard_reject = True
        if (
            abs(forward_comp) < cfg["min_forward_mps"]
            and abs(lateral_comp) >= cfg["lateral_min_mps"]
        ):
            hard_reject = True
        if abs(forward_comp) > 1e-6:
            if abs(lateral_comp) / abs(forward_comp) > cfg["max_lateral_to_forward_ratio"]:
                hard_reject = True

    if speed <= cfg["jitter_speed_mps"]:
        hard_reject = True

    moving_candidate = (speed >= cfg["moving_speed_mps"]) and (not hard_reject)
    parked_candidate = (speed <= cfg["parked_speed_mps"]) or hard_reject

    low_conf = bool(vehicle.get("__motion_low_confidence", False))
    enough_history = int(state.get("age", 0)) >= cfg["min_history_frames"]
    if low_conf:
        enough_history = enough_history and (int(state.get("age", 0)) >= (cfg["min_history_frames"] + 1))

    if not enough_history:
        moving_candidate = False
        parked_candidate = True

    if moving_candidate:
        state["move_count"] = int(state.get("move_count", 0)) + 1
    else:
        state["move_count"] = 0

    if parked_candidate:
        state["park_count"] = int(state.get("park_count", 0)) + 1
    else:
        state["park_count"] = 0

    is_moving = bool(state.get("is_moving", False))
    if is_moving:
        if state["park_count"] >= cfg["park_confirm_frames"]:
            is_moving = False
    else:
        if state["move_count"] >= cfg["move_confirm_frames"]:
            is_moving = True

    state["is_moving"] = is_moving
    state["raw_vx"] = raw_vx
    state["raw_vy"] = raw_vy
    state["last_xy"] = xy
    state["last_frame"] = frame_idx
    state["last_heading"] = vehicle.get("heading_rad")
    state["age"] = int(state.get("age", 0)) + 1
    state["was_observed"] = True

    vehicle["__motion_is_moving"] = bool(is_moving)
    vehicle["__motion_forward_dir"] = heading_forward


def _cleanup_motion_tracks(frame_idx: int, motion_ctx: dict) -> None:
    max_unseen = motion_ctx["cfg"]["max_unseen_frames"]
    stale_track_keys = []
    for track_key, state in motion_ctx["tracks"].items():
        if frame_idx - int(state.get("last_frame", -999)) > max_unseen:
            stale_track_keys.append(track_key)
    for track_key in stale_track_keys:
        motion_ctx["tracks"].pop(track_key, None)

    stale_anon_ids = []
    for anon_id, state in motion_ctx["anon_tracks"].items():
        if frame_idx - int(state.get("last_frame", -999)) > max_unseen:
            stale_anon_ids.append(anon_id)
    for anon_id in stale_anon_ids:
        motion_ctx["anon_tracks"].pop(anon_id, None)


def _annotate_vehicle_motion_state(vehicles: list, frame_idx: int, cfg: dict, motion_ctx: dict) -> dict:
    if not motion_ctx["cfg"]["enabled"]:
        for vehicle in vehicles:
            vehicle["__motion_is_moving"] = False
            vehicle["__motion_forward_dir"] = None
        return {"moving": 0, "parked": len(vehicles), "dominant_flow": (0.0, 0.0)}

    for vehicle in vehicles:
        track_id = vehicle.get("track_id")
        if track_id is None:
            vehicle["__motion_track_id"] = None
            vehicle["__motion_low_confidence"] = True
        else:
            vehicle["__motion_track_id"] = f"track-{int(track_id)}"
            vehicle["__motion_low_confidence"] = False

    _associate_anonymous_track_ids(vehicles, frame_idx, motion_ctx)

    dominant_flow = _estimate_dominant_flow(vehicles, frame_idx, motion_ctx)

    moving_count = 0
    for vehicle in vehicles:
        _update_single_motion_state(vehicle, frame_idx, dominant_flow, motion_ctx)
        track_key = vehicle.get("__motion_track_id")
        if track_key and str(track_key).startswith("anon-"):
            state = motion_ctx["tracks"].get(track_key)
            if state is not None:
                motion_ctx["anon_tracks"][int(str(track_key).split("-", 1)[1])] = {
                    "last_xy": state.get("last_xy"),
                    "last_frame": state.get("last_frame"),
                }
        if bool(vehicle.get("__motion_is_moving", False)):
            moving_count += 1

    _cleanup_motion_tracks(frame_idx, motion_ctx)

    parked_count = max(0, len(vehicles) - moving_count)
    return {
        "moving": moving_count,
        "parked": parked_count,
        "dominant_flow": dominant_flow,
    }


def render_sequence(
    scene_name: str,
    camera: str,
    cfg: dict,
    debug: bool = False,
    start_frame: int = 0,
    asset_lib=None,
):
    """
    Main loop: iterate over all detection JSONs for one scene+camera pair,
    update the scene per frame, and render each frame to PNG.
    Then stitch all PNGs to an MP4.
    """

    det_dir  = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    frames_dir = Path(cfg["paths"]["frames_dir"])   / scene_name / camera
    video_dir  = Path(cfg["paths"]["videos_dir"])
 
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # stale_frames = list(frames_dir.glob("frame_*.png"))
    # if stale_frames:
    #     for stale in stale_frames:
    #         stale.unlink()
    #     print(f"[scene] Cleared {len(stale_frames)} stale frame PNGs from {frames_dir}")
 
    jsons = list_frame_jsons(det_dir)
    if not jsons:
        print(f"[scene] WARNING: no JSONs found in {det_dir} — nothing to render")
        return

    filtered_jsons = []
    for json_path in jsons:
        parsed_idx = _frame_idx_from_json_path(json_path)
        if parsed_idx is not None:
            if parsed_idx >= start_frame:
                filtered_jsons.append(json_path)
            continue

        frame_data = load_detection_json(json_path)
        frame_idx = int(frame_data.get("frame", -1))
        if frame_idx >= start_frame:
            filtered_jsons.append(json_path)

    jsons = filtered_jsons
    if not jsons:
        print(
            f"[scene] WARNING: no JSONs at or after start_frame={start_frame} "
            f"in {det_dir} — nothing to render"
        )
        return
 
    print(
        f"[scene] Rendering {len(jsons)} frames for {scene_name}/{camera} "
        f"(start_frame={start_frame})"
    )

    motion_cfg = _blender_motion_cfg(cfg)
    motion_ctx = {
        "cfg": motion_cfg,
        "fps": float(cfg.get("blender", {}).get("fps", 30.0)),
        "tracks": {},
        "anon_tracks": {},
        "next_anon_id": 1,
    }

    # Provide current sequence context to asset renderer so it can apply
    # scene-specific rendering rules safely.
    if asset_lib and hasattr(asset_lib, "set_render_context"):
        asset_lib.set_render_context(scene_name=scene_name, camera_name=camera)
    
    # Initialize camera once, then update per-frame in third-person mode.
    cam_obj = setup_camera(cfg)
    camera_cfg = cfg.get("blender", {}).get("camera", {})
    camera_mode = str(camera_cfg.get("mode", "first_person")).strip().lower()
    use_third_person = camera_mode == "third_person"

    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    nonCOCO_objects = non_coco_cfg.get("object_list", [])
    if not nonCOCO_objects:
        classes_cfg = non_coco_cfg.get("classes", [])
        nonCOCO_objects = [c.get("export_bucket") for c in classes_cfg if isinstance(c, dict) and c.get("export_bucket")]

    dispatch = {
        "traffic_cones": "place_traffic_cone",
        "trash_cans": "place_trash_can",
        "traffic_poles": "place_traffic_pole",
        "speed_limit_signs": "place_speed_limit_sign",
        "ground_arrows": "place_ground_arrow",
        "ground_text_markings": "place_ground_text_marking",
    }

    # Lane renderer: persists across frames, geometry cleared per frame
    lane_renderer = LaneRenderer(cfg)

    for i, json_path in enumerate(jsons):
        frame_data = load_detection_json(json_path)
        frame_idx  = frame_data["frame"] 
        # ── Per-frame work goes here as we implement Steps 3-7 ──────────────
        # Step 3: camera.setup_camera(cfg)          (added in Step 3)
        # Step 4: asset_lib.place_vehicle(v) etc.   (added in Step 4)
        # Step 6: materials                          (added in Step 6)
        # Step 7: traffic lights / stop signs        (added in Step 7)
        # ────────────────────────────────────────────────────────────────────
        
        # empty the frame from all the objects
        if asset_lib:
            asset_lib.clear_frame_objects()

        # clear previous frame's lane geometry
        lane_renderer.clear()

        # go through all of the assets detected and place vehicles and pedestrians
        if asset_lib:
            ego_obj = None
            if use_third_person:
                ego_obj = asset_lib.place_ego_vehicle()
                if ego_obj is not None:
                    ego_cfg = cfg.get("blender", {}).get("ego_vehicle", {})
                    ego_heading_rad = float(ego_cfg.get("yaw_rad", 0.0))
                    update_chase_camera(
                        cam_obj,
                        cfg,
                        ego_location=(ego_obj.location.x, ego_obj.location.y, ego_obj.location.z),
                        ego_heading_rad=ego_heading_rad,
                    )
                else:
                    update_chase_camera(cam_obj, cfg)

            vehicles = []
            for v in frame_data.get("vehicles", []):
                cls = str(v.get("class", "")).lower()
                if bool(v.get("is_ego", False)) or cls in {"ego", "tesla"}:
                    continue
                vehicles.append(v)

            vehicles, dropped_dupes = _dedupe_vehicles_for_render(vehicles, cfg)
            if dropped_dupes > 0 and (debug or motion_cfg["debug"]):
                print(f"[scene] frame={frame_idx:06d} removed duplicate vehicles={dropped_dupes}")


            # Use is_moving from JSON, no longer compute motion state here
            for v in vehicles:
                asset_lib.place_vehicle(
                    v,
                    is_moving=bool(v.get("is_moving", False)),
                    motion_forward=None,
                )

            for p in frame_data.get("pedestrians", []):
                asset_lib.place_pedestrian(p)
            # traffic lights
            for t in frame_data.get("traffic_lights", []):
                asset_lib.place_traffic_light(t)
            # stop signs
            for s in frame_data.get("stop_signs", []):
                asset_lib.place_stop_sign(s)
            # lanes
            lane_block_bboxes = _collect_lane_block_bboxes(frame_data)
            for lane in frame_data.get("lanes", []):
                lane_frags = _split_lane_points_by_bboxes(
                    lane.get("points", []),
                    lane_block_bboxes,
                    pad_px=0.0,
                )
                for frag in lane_frags:
                    lane_frag = dict(lane)
                    lane_frag["points"] = frag
                    lane_renderer.draw_lane(lane_frag)
            # non-COCO objects
            for obj_type in nonCOCO_objects:
                place_fn_name = dispatch.get(obj_type)
                place_fn = getattr(asset_lib, place_fn_name, None) if place_fn_name else None
                if place_fn is None:
                    continue
                for obj in frame_data.get(obj_type, []):
                    if obj_type in {"ground_text_markings", "ground_arrows"}:
                        place_fn(obj, frame_data.get("lanes", []))
                    else:
                        place_fn(obj)

            # some white space for debugging
            print()

 
        out_png = frames_dir / f"frame_{frame_idx:06d}.png"

        render_frame(cfg, out_png)
 
        if debug:
            print(f"  [{scene_name}/{camera}] frame {frame_idx:06d} → {out_png.name}")
 
        if (i + 1) % 20 == 0:
            print(f"  [{scene_name}/{camera}] {i + 1}/{len(jsons)} frames rendered")
 
    # Stitch all PNGs to MP4
    out_mp4 = video_dir / f"{scene_name}_{camera}.mp4"
    frames_to_video(frames_dir, out_mp4, fps=cfg["blender"]["fps"])
    print(f"[scene] Done → {out_mp4}")


def main():
    args = parse_args()
    if args.start_frame < 0:
        raise ValueError("--start_frame must be >= 0")
 
    # config path is relative to Code/ unless absolute
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = code_dir / config_path
 
    cfg = load_config(str(config_path))
    _normalize_blender_cfg_paths(cfg, config_path)
 
    setup_scene(cfg)
    asset_lib = AssetLibrary(cfg)
    render_sequence(
        args.scene,
        args.cam,
        cfg,
        debug=args.debug,
        start_frame=args.start_frame,
        asset_lib=asset_lib,
    )
 
 
if __name__ == "__main__":
    if not IN_BLENDER:
        print("ERROR: This script must be run inside Blender:")
        print("  blender --background --python blender/scene.py -- --scene scene1 --cam front")
        sys.exit(1)
    main()
