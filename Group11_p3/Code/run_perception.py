"""
run_perception.py
=================
Entry point for the perception pipeline. Run this on the SSH cluster.

Usage
-----
  # Single sequence
  python run_perception.py --seq Seq1

  # All sequences defined in config.yaml
  python run_perception.py --all

  # Override device
  python run_perception.py --seq Seq1 --device cpu

Output
------
  Writes per-frame JSON files to:
    Data/outputs/detections/<seq>/frame_XXXXXX.json
"""

import argparse
import torch
from pathlib import Path
import os
import sys
import cv2
import numpy as np
sys.dont_write_bytecode = True

# Make sure we can import our modules regardless of where the script is run from
# sys.path.insert(0, str(Path(__file__).parent))

from utils.io_utils import load_config, frame_generator, get_video_frames, save_detection_json
from utils.viz import draw_detections, show_or_save, draw_traffic_lights, draw_signs, draw_non_coco_objects
from perception.lanes import LaneDetector
from perception.objects import ObjectDetector
from perception.objectsDART import NonCocoDartDetector
from perception.depth import DepthEstimator
from perception.traffic import TrafficLightDetector
from perception.signs import SignDetector
from perception.orientation import OrientationEstimator
from perception.export import build_frame_dict


def parse_args():
    parser = argparse.ArgumentParser()

    # this group makes is to that some arguments can only be called in a mutually exclusive way
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scene", 
        type=str,
        help="Single sequence name, e.g. scene1"
    )
    group.add_argument(
        "--all", 
        action="store_true", 
        help="Run all sequences from config"
    )

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "--cam",
        type=str,
        help="Camera name: [front, back, left_repeater, right_repeater]"
    )
    group2.add_argument(
        "--allcam",
        action="store_true",
        help="Run for all four cameras"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Override device from config (cuda|cpu)"
    )
    
    # this is for overlaying over footage
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Write debug overlay images alongside JSONs"
    )

    return parser.parse_args()


def load_models(cfg, device):
    """Instantiate all detectors once — expensive, do it outside the frame loop."""
    print("[init] Loading models...")
    perception_cfg = cfg.get("perception", {})
    non_coco_cfg = perception_cfg.get("non_coco_dart") or perception_cfg.get("cones", {})
    non_coco_enabled = bool(non_coco_cfg.get("enabled", False))
    models = {
        "objects":     ObjectDetector(cfg, device),
        "orientation": OrientationEstimator(cfg, device, strict=True),
        "lanes":   LaneDetector(cfg, device),
        "depth":   DepthEstimator(cfg, device),
        "traffic": TrafficLightDetector(cfg),
        "signs":   SignDetector(cfg),
        "non_coco": NonCocoDartDetector(cfg, device) if non_coco_enabled else None,
    }
    print("[init] All models loaded in the process of instantializing detectors.")
    return models


def _lane_color_bgr(color_name: str):
    name = str(color_name).lower()
    if name == "yellow":
        return (0, 220, 255)
    return (255, 255, 255)


def _draw_dashed_polyline(img, points, color, thickness=2, dash_len=14, gap_len=8):
    if len(points) < 2:
        return

    draw_segment = True
    phase = 0.0
    on_off = float(dash_len)

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        seg_vec = (p2[0] - p1[0], p2[1] - p1[1])
        seg_len = (seg_vec[0] ** 2 + seg_vec[1] ** 2) ** 0.5
        if seg_len < 1e-6:
            continue

        ux, uy = seg_vec[0] / seg_len, seg_vec[1] / seg_len
        progress = 0.0
        while progress < seg_len:
            step = min(on_off - phase, seg_len - progress)
            if draw_segment and step > 0:
                s = (
                    int(round(p1[0] + ux * progress)),
                    int(round(p1[1] + uy * progress)),
                )
                e = (
                    int(round(p1[0] + ux * (progress + step))),
                    int(round(p1[1] + uy * (progress + step))),
                )
                cv2.line(img, s, e, color, thickness, lineType=cv2.LINE_AA)

            progress += step
            phase += step
            if phase >= on_off - 1e-6:
                phase = 0.0
                draw_segment = not draw_segment
                on_off = float(dash_len if draw_segment else gap_len)


def draw_lane_debug_overlay(frame_bgr, lane_results, lane_raw=None):
    """Draw lane polylines and optional raw detector boxes on a frame."""
    vis = frame_bgr.copy()

    for lane in lane_results:
        points = lane.get("points", []) if isinstance(lane, dict) else []
        if len(points) < 2:
            continue

        poly = [(int(round(x)), int(round(y))) for x, y in points]
        color = _lane_color_bgr(lane.get("color", "white"))
        conf = float(lane.get("confidence", 0.0))

        cv2.polylines(
            vis,
            [np.array(poly, dtype=np.int32)],
            False,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        label = f"{lane.get('color', 'white')} {conf:.2f}"
        cv2.putText(
            vis,
            label,
            (poly[-1][0] + 6, max(poly[-1][1] - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    if isinstance(lane_raw, dict) and "boxes" in lane_raw and "scores" in lane_raw:
        boxes = lane_raw["boxes"]
        scores = lane_raw["scores"]
        names = lane_raw.get("class_names", [])
        for i in range(len(scores)):
            box = boxes[i].detach().cpu().tolist()
            score = float(scores[i].item())
            cls_name = names[i] if i < len(names) else "lane"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 255, 50), 1)
            cv2.putText(
                vis,
                f"{cls_name} {score:.2f}",
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (50, 255, 50),
                1,
                lineType=cv2.LINE_AA,
            )

    return vis


def _camera_projection_matrix(cfg: dict) -> np.ndarray:
    cam = cfg["blender"]["camera"]
    return np.array(
        [
            [float(cam["fx"]), 0.0, float(cam["cx"]), 0.0],
            [0.0, float(cam["fy"]), float(cam["cy"]), 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def process_sequence(
    scene_name: str,
    camera: str,
    cfg: dict,
    models: dict,
    debug: bool = False,
):
    """Run every active detector on every frame of one sequence and write JSONs."""
    vehicle_labels = {"bicycle", "car", "motorcycle", "bus", "truck"}

    # scene_dir is the root of this sequence, e.g. Data/Sequences/scene1/
    scene_dir = Path(cfg["paths"]["sequences_dir"]) / scene_name

    # Output dir per sequence + camera, e.g. Outputs/Detections/scene1/front/
    out_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = out_dir / "../debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    debug_proj_matrix = _camera_projection_matrix(cfg) if debug else None

    print(f"[{scene_name}] Camera: {camera}")
    # Use the generator so we never load all frames into RAM at once
    for i, (frame_idx, frame_bgr) in enumerate(
        frame_generator(scene_dir, camera=camera, frame_skip=cfg["perception"]["frame_skip"])
):
        # --- Run detectors ---
        object_results = models["objects"].detect(frame_bgr)
        lane_results = []
        lane_raw = None
        traffic_results = []
        sign_results = []

        lane_output = models["lanes"].detect(frame_bgr)
        non_coco_results = models["non_coco"].detect(frame_bgr) if models.get("non_coco") is not None else []
        depth_map = models["depth"].estimate(frame_bgr)
        object_results = models["depth"].lift_to_3d(object_results, depth_map)
        non_coco_results = models["depth"].lift_to_3d(non_coco_results, depth_map)
        traffic_results = models["depth"].lift_to_3d(models["traffic"].detect(frame_bgr, object_results), depth_map)
        sign_results = models["signs"].detect(frame_bgr, object_results)
        
        _SPECIALIZED = {"traffic_light", "stop_sign"}
        object_results = [d for d in object_results if d.label not in _SPECIALIZED]

        vehicle_results = [d for d in object_results if d.label in vehicle_labels]
        orientation_estimates = models["orientation"].annotate_detections(frame_bgr, vehicle_results)

        if isinstance(lane_output, dict) and "lanes" in lane_output:
            lane_results = lane_output["lanes"]
            lane_raw = lane_output.get("raw")
        else:
            # Backward compatibility if detect returns a plain list.
            lane_results = lane_output
            lane_raw = None

        # --- Build and save JSON ---
        frame_dict = build_frame_dict(
            frame_idx=frame_idx,
            fps=cfg["blender"]["fps"],
            lanes=lane_results,
            objects=object_results,
            traffic_lights=traffic_results,
            stop_signs=sign_results,
            non_coco_objects=non_coco_results,
        )
        save_detection_json(frame_dict, out_dir / f"frame_{frame_idx:06d}.json")

        if debug:
            annotated = draw_detections(frame_bgr, object_results, proj_matrix=debug_proj_matrix)
            annotated_traffic = draw_traffic_lights(annotated, traffic_results)
            annotated_signs = draw_signs(annotated_traffic, sign_results)
            annotated_non_coco = draw_non_coco_objects(annotated_signs, non_coco_results)

            # FIXME all these paths
            show_or_save(
                annotated_non_coco,
                save_path=str(debug_dir / f"debug_frame_{frame_idx:06d}.png")
            )
            overlay = draw_lane_debug_overlay(frame_bgr, lane_results, lane_raw)
            debug_path = debug_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(debug_path), overlay)

        if (i + 1) % 50 == 0:
            raw_count = 0
            if isinstance(lane_raw, dict) and "scores" in lane_raw:
                raw_count = len(lane_raw["scores"])
            print(
                f"  [{scene_name}/{camera}] {i+1} frames processed | "
                f"lanes={len(lane_results)} raw_dets={raw_count} "
                f"vehicles={len(vehicle_results)} oriented={len(orientation_estimates)}"
            )

    print(f"[{scene_name}] Done. JSONs saved to {out_dir}")
    if debug:
        print(f"[{scene_name}] Debug overlays saved to {debug_dir}")
    return


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[init] Using device: {device}")

    args = parse_args()
    cfg = load_config("config.yaml")

    if args.device:
        device = args.device

    # for the sequences, get all of them unless the argument just specified one
    scenes = cfg["sequences"] if args.all else [args.scene]

    cameras = cfg["cameras"] if args.allcam else [args.cam]

    # instantiate all of the detectors
    models = load_models(cfg, device)

    # process the sequences
    for scene in scenes:
        for camera in cameras:  # this might break at the time to do for all the cameras
            process_sequence(
                scene,
                camera,
                cfg,
                models,
                debug=args.debug,
            )

    print("\n[done] All sequences processed.")
    return

if __name__ == "__main__":
    main()
