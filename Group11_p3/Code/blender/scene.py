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

# sys.path must include Code/ so our utils and blender packages are importable
code_dir = Path(__file__).resolve().parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from utils.io_utils import load_detection_json, list_frame_jsons, load_config
from blender.render  import render_frame, frames_to_video
from blender.camera import setup_camera
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
    return parser.parse_args(argv)


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


def render_sequence(scene_name: str, camera: str, cfg: dict, debug: bool = False, asset_lib=None):
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
 
    jsons = list_frame_jsons(det_dir)
    if not jsons:
        print(f"[scene] WARNING: no JSONs found in {det_dir} — nothing to render")
        return
 
    print(f"[scene] Rendering {len(jsons)} frames for {scene_name}/{camera}")

    # Provide current sequence context to asset renderer so it can apply
    # scene-specific rendering rules safely.
    if asset_lib and hasattr(asset_lib, "set_render_context"):
        asset_lib.set_render_context(scene_name=scene_name, camera_name=camera)
    
    # Camera is fixed for the entire sequence, the objects are the ones that move around
    setup_camera(cfg)

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
            for v in frame_data.get("vehicles", []):
                asset_lib.place_vehicle(v)
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
 
    # config path is relative to Code/ unless absolute
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = code_dir / config_path
 
    cfg = load_config(str(config_path))
 
    setup_scene(cfg)
    asset_lib = AssetLibrary(cfg)
    render_sequence(args.scene, args.cam, cfg, debug=args.debug, asset_lib=asset_lib)
 
 
if __name__ == "__main__":
    if not IN_BLENDER:
        print("ERROR: This script must be run inside Blender:")
        print("  blender --background --python blender/scene.py -- --scene scene1 --cam front")
        sys.exit(1)
    main()