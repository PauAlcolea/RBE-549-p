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
from perception.speed_limit_ocr import SpeedLimitOcr
from perception.export import build_frame_dict
from perception.vehicle_subtypes import VehicleSubtypeClassifier


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

    parser.add_argument(
        "--night",
        action="store_true",
        help="Enable night traffic-light mode (pick lowest-saturation ROI)."
    )

    
    
    return parser.parse_args()


def load_models(cfg, device, night_mode: bool = False):
    """Instantiate all detectors once — expensive, do it outside the frame loop."""
    print("[init] Loading models...")
    perception_cfg = cfg.get("perception", {})
    non_coco_cfg = perception_cfg.get("non_coco_dart") or perception_cfg.get("cones", {})
    non_coco_enabled = bool(non_coco_cfg.get("enabled", False))
    vehicle_subtype_cfg = perception_cfg.get("vehicle_subtypes_dart", {})
    vehicle_subtypes_enabled = bool(vehicle_subtype_cfg.get("enabled", False))
    speed_ocr = SpeedLimitOcr(cfg, device=device)
    models = {
        "objects":     ObjectDetector(cfg, device),
        "orientation": OrientationEstimator(cfg, device, strict=True),
        "lanes":   LaneDetector(cfg, device),
        "depth":   DepthEstimator(cfg, device),
        "traffic": TrafficLightDetector(cfg),
        "signs":   SignDetector(cfg),
        "non_coco": NonCocoDartDetector(cfg, device) if non_coco_enabled else None,
        "vehicle_subtypes": VehicleSubtypeClassifier(cfg, device) if vehicle_subtypes_enabled else None,
        "speed_limit_ocr": speed_ocr,
    }
    models["traffic"].set_night_mode(night_mode)
    print("[init] All models loaded in the process of instantializing detectors.")
    return models


def _run_speed_limit_ocr(frame_bgr, non_coco_results, speed_ocr, frame_idx=None, debug_dir=None):
    if speed_ocr is None or not speed_ocr.is_active():
        return

    filtered_results = []
    for det in non_coco_results:
        if getattr(det, "label", "") != "speed_limit_sign":
            filtered_results.append(det)
            continue

        ocr_debug_path = None
        if debug_dir is not None and frame_idx is not None:
            x1, y1, _, _ = [int(round(v)) for v in det.bbox]
            ocr_debug_path = Path(debug_dir) / f"frame_{frame_idx:06d}_x{x1}_y{y1}.png"

        try:
            ocr_result = speed_ocr.infer(frame_bgr, det.bbox, debug_path=ocr_debug_path)
        except Exception as exc:
            print(f"[warn] speed-limit OCR failed for one detection: {exc}")
            speed_ocr.enabled = False
            return

        det.speed_value = ocr_result.speed_value
        det.ocr_confidence = float(ocr_result.ocr_confidence)
        det.ocr_raw_text = str(ocr_result.raw_text)

        # Keep speed-limit detections only when OCR confirms both words.
        if not ocr_result.has_speed_limit_words:
            continue

        filtered_results.append(det)

    non_coco_results[:] = filtered_results


def _lane_color_bgr(color_name: str):
    name = str(color_name).lower()
    if name == "yellow":
        return (0, 220, 255)
    return (255, 255, 255)


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

def draw_traffic_algorithm_overlay(frame_bgr, traffic_debug):
    """Overlay traffic-light region geometry and scores on the full frame."""
    vis = frame_bgr.copy()
    region_colors = [(0, 0, 255), (0, 220, 255), (0, 255, 0)]  # red, yellow, green

    for tl in traffic_debug:
        bbox = tl.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

        focus = tl.get("focus_rect")
        if focus and len(focus) == 4:
            fx1, fy1, fx2, fy2 = [int(v) for v in focus]
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)

            for yline in tl.get("band_lines_y", []):
                y = int(yline)
                cv2.line(vis, (fx1, y), (fx2, y), (200, 200, 200), 1, lineType=cv2.LINE_AA)

        circles = tl.get("roi_circles", [])
        for idx, c in enumerate(circles):
            if len(c) not in (3, 4):
                continue
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            region_idx = int(c[3]) if len(c) == 4 else idx
            color = region_colors[region_idx] if region_idx < len(region_colors) else (255, 255, 255)
            cv2.circle(vis, (cx, cy), r, color, 1, lineType=cv2.LINE_AA)

        scores = tl.get("region_scores", [])
        if len(scores) == 3:
            pred = tl.get("predicted_color", "unknown")
            win = tl.get("winner_idx")
            metric = str(tl.get("selection_metric", "score"))
            if metric == "lowest_sat_high_val":
                sat = tl.get("saturation_means", [0.0, 0.0, 0.0])
                val = tl.get("value_means", [0.0, 0.0, 0.0])
                score_text = (
                    f"M R:{scores[0]:.2f} Y:{scores[1]:.2f} G:{scores[2]:.2f} "
                    f"S/V R:{sat[0]:.1f}/{val[0]:.1f} "
                    f"Y:{sat[1]:.1f}/{val[1]:.1f} "
                    f"G:{sat[2]:.1f}/{val[2]:.1f} -> {pred}"
                )
            else:
                score_text = f"R:{scores[0]:.3f} Y:{scores[1]:.3f} G:{scores[2]:.3f} -> {pred}"
            if win is not None:
                score_text += f" (win={int(win)})"

            cv2.putText(
                vis,
                score_text,
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    return vis


def save_traffic_detail_panels(frame_bgr, traffic_debug, frame_idx, out_dir: Path):
    """Save one crop-level debug image per traffic light with local geometry and scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    region_colors = [(0, 0, 255), (0, 220, 255), (0, 255, 0)]

    for tl_idx, tl in enumerate(traffic_debug):
        crop_rect = tl.get("crop_rect")
        if not crop_rect or len(crop_rect) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in crop_rect]
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        focus = tl.get("focus_rect")
        if focus and len(focus) == 4:
            fx1, fy1, fx2, fy2 = [int(v) for v in focus]
            cv2.rectangle(crop, (fx1 - x1, fy1 - y1), (fx2 - x1, fy2 - y1), (255, 200, 0), 1)

            for yline in tl.get("band_lines_y", []):
                y = int(yline) - y1
                cv2.line(crop, (max(fx1 - x1, 0), y), (min(fx2 - x1, crop.shape[1] - 1), y), (200, 200, 200), 1)

        for idx, c in enumerate(tl.get("roi_circles", [])):
            if len(c) not in (3, 4):
                continue
            cx, cy, r = int(c[0]) - x1, int(c[1]) - y1, int(c[2])
            if 0 <= cx < crop.shape[1] and 0 <= cy < crop.shape[0]:
                region_idx = int(c[3]) if len(c) == 4 else idx
                color = region_colors[region_idx] if region_idx < len(region_colors) else (255, 255, 255)
                cv2.circle(crop, (cx, cy), r, color, 1, lineType=cv2.LINE_AA)

        scores = tl.get("region_scores", [])
        pred = tl.get("predicted_color", "unknown")
        v_thr = tl.get("global_v_thr")
        metric = str(tl.get("selection_metric", "score"))
        text_lines = [f"pred: {pred}", f"metric: {metric}", f"v_thr: {0.0 if v_thr is None else float(v_thr):.1f}"]
        if len(scores) == 3:
            if metric == "lowest_sat_high_val":
                sat = tl.get("saturation_means", [0.0, 0.0, 0.0])
                val = tl.get("value_means", [0.0, 0.0, 0.0])
                text_lines.append(f"M R={scores[0]:.2f} Y={scores[1]:.2f} G={scores[2]:.2f}")
                text_lines.append(f"S R={sat[0]:.1f} Y={sat[1]:.1f} G={sat[2]:.1f}")
                text_lines.append(f"V R={val[0]:.1f} Y={val[1]:.1f} G={val[2]:.1f}")
            else:
                text_lines.append(f"R={scores[0]:.3f} Y={scores[1]:.3f} G={scores[2]:.3f}")

        panel = cv2.copyMakeBorder(crop, 0, 42, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        for i, line in enumerate(text_lines):
            cv2.putText(
                panel,
                line,
                (4, crop.shape[0] + 14 + (i * 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                (240, 240, 240),
                1,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_tl_{tl_idx:02d}.png"), panel)

        
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

def process_sequence(scene_name: str, camera: str, cfg: dict, models: dict, debug: bool = False):
    """Run every active detector on every frame of one sequence and write JSONs."""
    vehicle_labels = {"bicycle", "car", "motorcycle", "bus", "truck", "sedan", "hatchback", "suv", "pickuptruck", "pickup_truck"}

    # scene_dir is the root of this sequence, e.g. Data/Sequences/scene1/
    scene_dir = Path(cfg["paths"]["sequences_dir"]) / scene_name

    # Output dir per sequence + camera, e.g. Outputs/Detections/scene1/front/
    out_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = out_dir / "../debug" 
    lane_debug_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / "lane_debug"
    traffic_algo_dir = out_dir / "../traffic_algo"
    ocr_debug_dir = out_dir / "../ocr_debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        lane_debug_dir.mkdir(parents=True, exist_ok=True)
        traffic_algo_dir.mkdir(parents=True, exist_ok=True)
        ocr_debug_dir.mkdir(parents=True, exist_ok=True)
    debug_proj_matrix = _camera_projection_matrix(cfg) if debug else None

    print(f"[{scene_name}] Camera: {camera}")
    # Use the generator so we never load all frames into RAM at once
    for i, (frame_idx, frame_bgr) in enumerate(
        frame_generator(scene_dir, camera=camera, frame_skip=cfg["perception"]["frame_skip"])
):
        # --- Run detectors ---
        object_results = models["objects"].detect(frame_bgr)
        if models.get("vehicle_subtypes") is not None:
            object_results = models["vehicle_subtypes"].refine_detections(frame_bgr, object_results)
        lane_results = []
        lane_raw = None
        traffic_results = []
        sign_results = []

        lane_output = models["lanes"].detect(frame_bgr)
        non_coco_results = models["non_coco"].detect(frame_bgr) if models.get("non_coco") is not None else []
        _run_speed_limit_ocr(
            frame_bgr,
            non_coco_results,
            models.get("speed_limit_ocr"),
            frame_idx=frame_idx,
            debug_dir=ocr_debug_dir if debug else None,
        )
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
            debug_path = lane_debug_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(debug_path), overlay)

            traffic_debug = models["traffic"].last_debug_info
            if traffic_debug:
                traffic_overlay = draw_traffic_algorithm_overlay(frame_bgr, traffic_debug)
                cv2.imwrite(str(traffic_algo_dir / f"frame_{frame_idx:06d}.png"), traffic_overlay)
                save_traffic_detail_panels(
                    frame_bgr,
                    traffic_debug,
                    frame_idx,
                    traffic_algo_dir / "detail",
                )

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
        print(f"[{scene_name}] Lane overlays saved to {lane_debug_dir}")
        print(f"[{scene_name}] Traffic algorithm figures saved to {traffic_algo_dir}")
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
    models = load_models(cfg, device, night_mode=args.night)
    if args.night:
        print("[init] Night traffic-light mode enabled (--night).")

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
