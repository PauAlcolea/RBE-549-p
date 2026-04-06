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
from copy import deepcopy
import torch
from pathlib import Path
import os
import sys
import cv2
import numpy as np
from types import SimpleNamespace
sys.dont_write_bytecode = True

# Make sure we can import our modules regardless of where the script is run from
# sys.path.insert(0, str(Path(__file__).parent))

from utils.io_utils import (
    load_config,
    frame_generator,
    get_video_frames,
    save_detection_json,
    load_detection_json,
    list_frame_jsons,
)
from utils.viz import (
    draw_detections,
    show_or_save,
    draw_traffic_lights,
    draw_signs,
    draw_non_coco_objects,
    draw_pymaf_matches,
)
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
from perception.pymaf import PymafEstimator


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

    parser.add_argument(
        "--person",
        action="store_true",
        help="Debug mode: only run person detection."
    )

    parser.add_argument(
        "--pymaf-only",
        action="store_true",
        help="Run only PyMAF and update existing detection JSONs in-place."
    )

    
    
    return parser.parse_args()


def load_models(
    cfg,
    device,
    night_mode: bool = False,
    person_only: bool = False,
    pymaf_only: bool = False,
):
    """Instantiate all detectors once — expensive, do it outside the frame loop."""
    print("[init] Loading models...")
    if pymaf_only:
        models = {
            "pymaf": PymafEstimator(cfg, device),
        }
        print("[init] PyMAF-only mode enabled: loading just PyMAF bridge.")
        return models

    if person_only:
        person_cfg = deepcopy(cfg)
        person_cfg["perception"]["yolo"]["classes_phase1"] = [0]
        models = {
            "objects": ObjectDetector(person_cfg, device),
        }
        print("[init] Person-only debug mode enabled: loading just the object detector.")
        return models

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


def _bbox_iou(a, b) -> float:
    """Compute IoU for [x1, y1, x2, y2] boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _bbox_intersection(a, b) -> float:
    """Intersection area for [x1, y1, x2, y2] boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_area(a) -> float:
    return max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))


def _as_traffic_candidate(det, source: str = "unknown"):
    """Normalize a detection-like record to traffic detector candidate shape."""
    return SimpleNamespace(
        label="traffic_light",
        bbox=[float(v) for v in getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])],
        confidence=float(getattr(det, "confidence", 0.0)),
        source=str(source),
    )


def _dedupe_traffic_candidates(candidates: list, iou_thr: float) -> list:
    """NMS-style dedupe for traffic candidates; keep highest-confidence boxes."""
    if len(candidates) <= 1:
        return candidates

    ordered = sorted(candidates, key=lambda d: float(getattr(d, "confidence", 0.0)), reverse=True)
    kept = []
    for cand in ordered:
        cand_bbox = getattr(cand, "bbox", None)
        if cand_bbox is None:
            continue
        if any(_bbox_iou(cand_bbox, getattr(k, "bbox", cand_bbox)) >= iou_thr for k in kept):
            continue
        kept.append(cand)
    return kept


def _suppress_arrow_signal_overlaps(candidates: list, iou_thr: float, ioa_thr: float) -> list:
    """Resolve overlaps between DART arrow-signal boxes and other traffic-light boxes."""
    if len(candidates) <= 1:
        return candidates

    arrow_idxs = [
        i for i, c in enumerate(candidates)
        if str(getattr(c, "source", "")) == "traffic_light_arrow_signal"
    ]
    if not arrow_idxs:
        return candidates

    drop = set()
    for i in arrow_idxs:
        if i in drop:
            continue
        a = candidates[i]
        a_bbox = getattr(a, "bbox", None)
        if a_bbox is None:
            continue

        for j, b in enumerate(candidates):
            if i == j or j in drop:
                continue
            if str(getattr(b, "source", "")) == "traffic_light_arrow_signal":
                continue

            b_bbox = getattr(b, "bbox", None)
            if b_bbox is None:
                continue

            iou = _bbox_iou(a_bbox, b_bbox)
            inter = _bbox_intersection(a_bbox, b_bbox)
            a_area = _bbox_area(a_bbox)
            b_area = _bbox_area(b_bbox)
            min_area = max(min(a_area, b_area), 1e-6)
            ioa_small = inter / min_area

            # Handle both near-equal overlap (IoU) and containment (IoA of smaller box).
            if (iou < iou_thr) and (ioa_small < ioa_thr):
                continue

            a_conf = float(getattr(a, "confidence", 0.0))
            b_conf = float(getattr(b, "confidence", 0.0))
            # Remove one: keep the higher-confidence candidate on overlap.
            if a_conf >= b_conf:
                drop.add(j)
            else:
                drop.add(i)
                break

    return [c for idx, c in enumerate(candidates) if idx not in drop]


def _suppress_contained_candidates(candidates: list, ioa_thr: float) -> list:
    """Drop one candidate when one bbox is largely contained inside another."""
    if len(candidates) <= 1:
        return candidates

    drop = set()
    for i, a in enumerate(candidates):
        if i in drop:
            continue
        a_bbox = getattr(a, "bbox", None)
        if a_bbox is None:
            continue

        for j in range(i + 1, len(candidates)):
            if j in drop:
                continue
            b = candidates[j]
            b_bbox = getattr(b, "bbox", None)
            if b_bbox is None:
                continue

            inter = _bbox_intersection(a_bbox, b_bbox)
            if inter <= 0.0:
                continue

            a_area = max(_bbox_area(a_bbox), 1e-6)
            b_area = max(_bbox_area(b_bbox), 1e-6)
            ioa_small = inter / min(a_area, b_area)
            if ioa_small < ioa_thr:
                continue

            a_conf = float(getattr(a, "confidence", 0.0))
            b_conf = float(getattr(b, "confidence", 0.0))
            if a_conf >= b_conf:
                drop.add(j)
            else:
                drop.add(i)
                break

    return [c for idx, c in enumerate(candidates) if idx not in drop]


def _build_traffic_candidates(object_results: list, non_coco_results: list, cfg: dict, scene_name: str):
    """OR YOLO traffic lights with DART lane-control lights, then dedupe."""
    traffic_cfg = cfg.get("perception", {}).get("traffic_light", {})
    dart_lane_control_enabled = bool(traffic_cfg.get("dart_lane_control_enabled", True))
    dart_lane_control_min_conf = float(traffic_cfg.get("dart_lane_control_min_confidence", 0.30))
    dedupe_iou = float(traffic_cfg.get("dart_lane_control_dedupe_iou", 0.45))
    arrow_overlap_iou = float(traffic_cfg.get("arrow_overlap_iou", 0.35))
    arrow_overlap_ioa = float(traffic_cfg.get("arrow_overlap_ioa", 0.65))
    nested_overlap_ioa = float(traffic_cfg.get("nested_overlap_ioa", 0.80))
    square_aspect_min = float(traffic_cfg.get("square_arrow_aspect_min", 0.75))
    square_aspect_max = float(traffic_cfg.get("square_arrow_aspect_max", 1.35))
    dart_arrow_aspect_min = float(traffic_cfg.get("dart_arrow_signal_aspect_min", 0.70))
    dart_arrow_aspect_max = float(traffic_cfg.get("dart_arrow_signal_aspect_max", 1.35))
    max_lane_control_side_px = 90.0
    scene_name_norm = str(scene_name).strip().lower()
    lane_control_scene_enabled = scene_name_norm == "scene2"
    arrow_signal_scene_enabled = scene_name_norm == "scene6"
    dart_traffic_aux_labels = {"lane_control_light", "traffic_light_arrow_signal"}

    yolo_candidates = [
        _as_traffic_candidate(d, source="yolo_traffic_light")
        for d in object_results
        if getattr(d, "label", "") == "traffic_light"
    ]

    if not dart_lane_control_enabled:
        non_coco_filtered = [d for d in non_coco_results if getattr(d, "label", "") not in dart_traffic_aux_labels]
        return _dedupe_traffic_candidates(yolo_candidates, iou_thr=dedupe_iou), non_coco_filtered

    dart_traffic_candidates = []
    non_coco_filtered = []
    for det in non_coco_results:
        det_label = getattr(det, "label", "")
        if det_label not in dart_traffic_aux_labels:
            non_coco_filtered.append(det)
            continue

        if float(getattr(det, "confidence", 0.0)) < dart_lane_control_min_conf:
            continue

        bbox = getattr(det, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue

        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        if height <= 1e-6:
            continue

        if det_label == "lane_control_light":
            if not lane_control_scene_enabled:
                continue
            if width > max_lane_control_side_px or height > max_lane_control_side_px:
                continue

            aspect = width / height
            if not (square_aspect_min <= aspect <= square_aspect_max):
                continue

            dart_traffic_candidates.append(_as_traffic_candidate(det, source=det_label))
            continue

        if det_label == "traffic_light_arrow_signal":
            if not arrow_signal_scene_enabled:
                continue
            aspect = width / height
            if not (dart_arrow_aspect_min <= aspect <= dart_arrow_aspect_max):
                continue
            dart_traffic_candidates.append(_as_traffic_candidate(det, source=det_label))
            continue

    merged = list(yolo_candidates)
    merged.extend(dart_traffic_candidates)
    merged = _suppress_arrow_signal_overlaps(merged, iou_thr=arrow_overlap_iou, ioa_thr=arrow_overlap_ioa)
    merged = _suppress_contained_candidates(merged, ioa_thr=nested_overlap_ioa)
    deduped = _dedupe_traffic_candidates(merged, iou_thr=dedupe_iou)
    return deduped, non_coco_filtered


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

def process_sequence(
    scene_name: str,
    camera: str,
    cfg: dict,
    models: dict,
    debug: bool = False,
    person_only: bool = False,
):
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
        if not person_only:
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
        lane_results = []
        lane_raw = None
        non_coco_results = []
        traffic_results = []
        sign_results = []
        vehicle_results = []
        orientation_estimates = []

        if person_only:
            object_results = [d for d in object_results if d.label == "person"]
        else:
            if models.get("vehicle_subtypes") is not None:
                object_results = models["vehicle_subtypes"].refine_detections(frame_bgr, object_results)

            lane_output = models["lanes"].detect(frame_bgr)
            non_coco_results = models["non_coco"].detect(frame_bgr) if models.get("non_coco") is not None else []
            traffic_candidates, non_coco_results = _build_traffic_candidates(
                object_results,
                non_coco_results,
                cfg,
                scene_name,
            )
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
            traffic_results = models["depth"].lift_to_3d(
                models["traffic"].detect(frame_bgr, traffic_candidates),
                depth_map,
            )
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

            if not person_only:
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
                f"vehicles={len(vehicle_results)} oriented={len(orientation_estimates)} "
                f"pedestrians={sum(1 for d in object_results if d.label == 'person')}"
            )

    print(f"[{scene_name}] Done. JSONs saved to {out_dir}")
    if debug:
        print(f"[{scene_name}] Debug overlays saved to {debug_dir}")
        if not person_only:
            print(f"[{scene_name}] Lane overlays saved to {lane_debug_dir}")
            print(f"[{scene_name}] Traffic algorithm figures saved to {traffic_algo_dir}")
    return


def _frame_idx_from_json_path(json_path: Path) -> int:
    name = json_path.stem  # frame_000123
    if not name.startswith("frame_"):
        raise ValueError(f"Unexpected frame json name: {json_path.name}")
    return int(name.split("_", 1)[1])


def process_sequence_pymaf_only(
    scene_name: str,
    camera: str,
    cfg: dict,
    models: dict,
    debug: bool = False,
):
    """
    Run only PyMAF and update existing per-frame detection JSON files in place.
    """
    scene_dir = Path(cfg["paths"]["sequences_dir"]) / scene_name
    out_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    pymaf_debug_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / "pymaf_debug"

    json_paths = list_frame_jsons(out_dir)
    if not json_paths:
        print(f"[warn] No existing detection JSONs found in {out_dir}.")
        return

    pymaf_model = models.get("pymaf")
    if pymaf_model is None or not pymaf_model.is_active():
        print(f"[warn] PyMAF is not active for {scene_name}/{camera}; skipping.")
        return

    print(f"[{scene_name}] Camera: {camera} (pymaf-only)")
    pymaf_model.prepare_scene(scene_name, camera, scene_dir)

    per_frame_person_dets = {}
    matched_total = 0
    updated_jsons = 0

    for i, json_path in enumerate(json_paths):
        frame_idx = _frame_idx_from_json_path(json_path)
        frame_dict = load_detection_json(json_path)
        pedestrians = frame_dict.get("pedestrians", [])

        det_pairs = []
        for ped in pedestrians:
            bbox = ped.get("bbox", None)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            # Clear stale PyMAF fields before re-annotating this frame.
            for key in ("pymaf_track_id", "pymaf_match_iou", "smpl_pose", "smpl_betas", "smpl_joints3d"):
                ped.pop(key, None)

            det_obj = SimpleNamespace(
                label="person",
                bbox=[float(v) for v in bbox],
                depth_m=float(ped.get("depth_m", 0.0)),
                confidence=1.0,
                position_3d=[float(v) for v in ped.get("position_3d", [0.0, 0.0, 0.0])],
            )
            det_pairs.append((ped, det_obj))

        dets = [d for _, d in det_pairs]
        matched_frame = pymaf_model.annotate_person_detections(frame_idx, dets)
        matched_total += matched_frame
        per_frame_person_dets[frame_idx] = dets

        for ped, det in det_pairs:
            track_id = getattr(det, "pymaf_track_id", None)
            if track_id is None:
                continue
            ped["pymaf_track_id"] = int(track_id)
            ped["pymaf_match_iou"] = round(float(getattr(det, "pymaf_match_iou", 0.0)), 4)

            smpl_pose = getattr(det, "smpl_pose", None)
            if smpl_pose is not None:
                ped["smpl_pose"] = [round(float(v), 6) for v in smpl_pose]

            smpl_betas = getattr(det, "smpl_betas", None)
            if smpl_betas is not None:
                ped["smpl_betas"] = [round(float(v), 6) for v in smpl_betas]

            smpl_joints3d = getattr(det, "smpl_joints3d", None)
            if smpl_joints3d is not None:
                ped["smpl_joints3d"] = [
                    [round(float(coord), 6) for coord in joint]
                    for joint in smpl_joints3d
                ]

        save_detection_json(frame_dict, json_path)
        updated_jsons += 1

        if (i + 1) % 50 == 0:
            print(
                f"  [{scene_name}/{camera}] {i+1} jsons updated | "
                f"frame_matches={matched_frame} total_matches={matched_total}"
            )

    if debug:
        pymaf_debug_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        target = len(per_frame_person_dets)
        for frame_idx, frame_bgr in frame_generator(scene_dir, camera=camera, frame_skip=1):
            dets = per_frame_person_dets.get(frame_idx, None)
            if dets is None:
                continue
            vis = draw_pymaf_matches(frame_bgr, dets)
            show_or_save(vis, save_path=str(pymaf_debug_dir / f"pymaf_frame_{frame_idx:06d}.png"))
            saved += 1
            if saved >= target:
                break

        print(f"[{scene_name}] PyMAF overlays saved to {pymaf_debug_dir}")

    print(
        f"[{scene_name}] PyMAF-only done for {camera}: "
        f"updated_jsons={updated_jsons}, total_matches={matched_total}"
    )
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
    models = load_models(
        cfg,
        device,
        night_mode=args.night,
        person_only=args.person,
        pymaf_only=args.pymaf_only,
    )
    if args.night:
        print("[init] Night traffic-light mode enabled (--night).")
    if args.person:
        print("[init] Person-only debug mode enabled (--person).")
    if args.pymaf_only:
        print("[init] PyMAF-only mode enabled (--pymaf-only).")

    # process the sequences
    for scene in scenes:
        for camera in cameras:  # this might break at the time to do for all the cameras
            if args.pymaf_only:
                process_sequence_pymaf_only(
                    scene,
                    camera,
                    cfg,
                    models,
                    debug=args.debug,
                )
            else:
                process_sequence(
                    scene,
                    camera,
                    cfg,
                    models,
                    debug=args.debug,
                    person_only=args.person,
                )

    print("\n[done] All sequences processed.")
    return

if __name__ == "__main__":
    main()
