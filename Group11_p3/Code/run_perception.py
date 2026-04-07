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
import re
import sys
import cv2
import numpy as np
from types import SimpleNamespace
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


def _ground_text_rank_key(det):
    hits = str(getattr(det, "only_letter_hits", ""))
    hit_count = len(set(hits))
    text_conf = float(getattr(det, "ocr_confidence", 0.0))
    det_conf = float(getattr(det, "confidence", 0.0))
    area = _bbox_area(getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0]))
    # For overlap dedupe, keep the largest ONLY-positive box first.
    return (area, hit_count, text_conf, det_conf)


def _merge_ground_text_fragments(non_coco_results, cfg):
    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    if not bool(non_coco_cfg.get("ground_text_merge_enabled", True)):
        return

    max_gap_ratio = float(non_coco_cfg.get("ground_text_merge_max_gap_ratio", 1.8))
    y_center_tol_ratio = float(non_coco_cfg.get("ground_text_merge_y_center_tol_ratio", 0.9))
    max_span_ratio = float(non_coco_cfg.get("ground_text_merge_max_span_ratio", 7.0))
    min_cluster_size = int(non_coco_cfg.get("ground_text_merge_min_cluster_size", 2))

    ground_text = [d for d in non_coco_results if getattr(d, "label", "") == "ground_text"]
    if len(ground_text) <= 1:
        return

    others = [d for d in non_coco_results if getattr(d, "label", "") != "ground_text"]
    ground_text_sorted = sorted(ground_text, key=lambda d: float(getattr(d, "bbox", [0.0, 0.0, 0.0, 0.0])[0]))

    clusters = []
    current = [ground_text_sorted[0]]
    c_bbox = [float(v) for v in getattr(current[0], "bbox", [0.0, 0.0, 0.0, 0.0])]
    for det in ground_text_sorted[1:]:
        det_bbox = [float(v) for v in getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])]

        prev_bbox = [float(v) for v in getattr(current[-1], "bbox", [0.0, 0.0, 0.0, 0.0])]
        prev_h = max(1.0, prev_bbox[3] - prev_bbox[1])
        det_h = max(1.0, det_bbox[3] - det_bbox[1])
        cluster_h = max(1.0, c_bbox[3] - c_bbox[1])
        ref_h = max(prev_h, det_h, cluster_h)
        prev_cy = 0.5 * (c_bbox[1] + c_bbox[3])
        det_cy = 0.5 * (det_bbox[1] + det_bbox[3])
        gap = det_bbox[0] - max(prev_bbox[2], c_bbox[2])
        proposed_x1 = min(c_bbox[0], det_bbox[0])
        proposed_x2 = max(c_bbox[2], det_bbox[2])
        proposed_span = proposed_x2 - proposed_x1

        same_line = abs(det_cy - prev_cy) <= (y_center_tol_ratio * ref_h)
        close_enough = gap <= (max_gap_ratio * ref_h)
        span_ok = proposed_span <= (max_span_ratio * ref_h)
        if same_line and close_enough and span_ok:
            current.append(det)
            c_bbox = [
                min(c_bbox[0], det_bbox[0]),
                min(c_bbox[1], det_bbox[1]),
                max(c_bbox[2], det_bbox[2]),
                max(c_bbox[3], det_bbox[3]),
            ]
        else:
            clusters.append(current)
            current = [det]
            c_bbox = [float(v) for v in det_bbox]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            merged.append(cluster[0])
            continue

        best = max(cluster, key=_ground_text_rank_key)
        x1 = min(float(getattr(d, "bbox", [0.0, 0.0, 0.0, 0.0])[0]) for d in cluster)
        y1 = min(float(getattr(d, "bbox", [0.0, 0.0, 0.0, 0.0])[1]) for d in cluster)
        x2 = max(float(getattr(d, "bbox", [0.0, 0.0, 0.0, 0.0])[2]) for d in cluster)
        y2 = max(float(getattr(d, "bbox", [0.0, 0.0, 0.0, 0.0])[3]) for d in cluster)

        hits_union = "".join(sorted({ch for d in cluster for ch in str(getattr(d, "only_letter_hits", ""))}))
        raw_text_parts = [str(getattr(d, "ocr_raw_text", "")).strip() for d in cluster]
        raw_text_parts = [p for p in raw_text_parts if p]

        best.bbox = [x1, y1, x2, y2]
        best.only_letter_hits = hits_union
        best.has_only_letters = bool(hits_union)
        best.ocr_confidence = max(float(getattr(d, "ocr_confidence", 0.0)) for d in cluster)
        best.confidence = max(float(getattr(d, "confidence", 0.0)) for d in cluster)
        if raw_text_parts:
            best.ocr_raw_text = " ".join(raw_text_parts)

        merged.append(best)

    non_coco_results[:] = others + merged


def _dedupe_ground_text_only(non_coco_results, cfg):
    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    if not bool(non_coco_cfg.get("ground_text_dedupe_enabled", True)):
        return

    iou_thr = float(non_coco_cfg.get("ground_text_dedupe_iou", 0.15))
    ioa_thr = float(non_coco_cfg.get("ground_text_dedupe_ioa", 0.55))

    ground_text = [d for d in non_coco_results if getattr(d, "label", "") == "ground_text"]
    if len(ground_text) <= 1:
        return

    others = [d for d in non_coco_results if getattr(d, "label", "") != "ground_text"]
    ordered = sorted(ground_text, key=_ground_text_rank_key, reverse=True)
    kept = []

    for det in ordered:
        det_bbox = getattr(det, "bbox", None)
        if det_bbox is None:
            continue

        suppress = False
        for prev in kept:
            prev_bbox = getattr(prev, "bbox", None)
            if prev_bbox is None:
                continue

            iou = _bbox_iou(det_bbox, prev_bbox)
            inter = _bbox_intersection(det_bbox, prev_bbox)
            small_area = max(min(_bbox_area(det_bbox), _bbox_area(prev_bbox)), 1e-6)
            ioa_small = inter / small_area
            if iou >= iou_thr or ioa_small >= ioa_thr:
                suppress = True
                break

        if not suppress:
            kept.append(det)

    non_coco_results[:] = others + kept


def _run_ground_text_ocr(frame_bgr, non_coco_results, speed_ocr, cfg, frame_idx=None, debug_dir=None):
    filtered_results = []
    if speed_ocr is None or not speed_ocr.is_active():
        for det in non_coco_results:
            if getattr(det, "label", "") != "ground_text":
                filtered_results.append(det)
        non_coco_results[:] = filtered_results
        return

    frame_h = int(frame_bgr.shape[0])
    min_y_center = frame_h * 0.5

    for det in non_coco_results:
        if getattr(det, "label", "") != "ground_text":
            filtered_results.append(det)
            continue

        bbox = [float(v) for v in getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])]
        y_center = (bbox[1] + bbox[3]) * 0.5
        if y_center < min_y_center:
            continue

        ocr_debug_path = None
        if debug_dir is not None and frame_idx is not None:
            x1, y1, _, _ = [int(round(v)) for v in det.bbox]
            ocr_debug_path = Path(debug_dir) / f"ground_text_frame_{frame_idx:06d}_x{x1}_y{y1}.png"

        try:
            ocr_result = speed_ocr.infer(frame_bgr, det.bbox, debug_path=ocr_debug_path)
        except Exception as exc:
            print(f"[warn] ground-text OCR failed for one detection: {exc}")
            speed_ocr.enabled = False
            continue

        det.ocr_confidence = float(ocr_result.text_confidence)
        det.ocr_raw_text = str(ocr_result.raw_text)
        det.has_only_letters = bool(ocr_result.has_only_letters)
        det.only_letter_hits = str(ocr_result.only_letter_hits)
        if not det.has_only_letters:
            continue

        filtered_results.append(det)

    non_coco_results[:] = filtered_results
    _merge_ground_text_fragments(non_coco_results, cfg)
    _dedupe_ground_text_only(non_coco_results, cfg)


def _suppress_ground_arrow_text_overlaps(non_coco_results, cfg):
    """Remove ground-arrow detections when they overlap confirmed ground_text detections."""
    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    iou_thr = float(non_coco_cfg.get("ground_arrow_text_overlap_iou", 0.05))
    ioa_thr = float(non_coco_cfg.get("ground_arrow_text_overlap_ioa", 0.35))

    ground_text = [d for d in non_coco_results if getattr(d, "label", "") == "ground_text"]
    if not ground_text:
        return

    filtered = []
    for det in non_coco_results:
        if getattr(det, "label", "") != "ground_arrow":
            filtered.append(det)
            continue

        det_bbox = getattr(det, "bbox", None)
        if det_bbox is None:
            continue

        suppress = False
        for txt in ground_text:
            txt_bbox = getattr(txt, "bbox", None)
            if txt_bbox is None:
                continue

            iou = _bbox_iou(det_bbox, txt_bbox)
            inter = _bbox_intersection(det_bbox, txt_bbox)
            small_area = max(min(_bbox_area(det_bbox), _bbox_area(txt_bbox)), 1e-6)
            ioa_small = inter / small_area
            if iou >= iou_thr or ioa_small >= ioa_thr:
                suppress = True
                break

        if not suppress:
            filtered.append(det)

    non_coco_results[:] = filtered


def _filter_ground_arrows(frame_bgr, non_coco_results, cfg):
    """Keep only road-like ground-arrow detections in lower image regions."""
    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    min_center_y_ratio = float(non_coco_cfg.get("ground_arrow_min_center_y_ratio", 0.5))
    min_bottom_y_ratio = float(non_coco_cfg.get("ground_arrow_min_bottom_y_ratio", 0.55))

    frame_h = float(frame_bgr.shape[0])
    min_center_y = frame_h * min_center_y_ratio
    min_bottom_y = frame_h * min_bottom_y_ratio

    filtered = []
    for det in non_coco_results:
        if getattr(det, "label", "") != "ground_arrow":
            filtered.append(det)
            continue

        x1, y1, x2, y2 = [float(v) for v in getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])]
        y_center = 0.5 * (y1 + y2)
        if y_center < min_center_y or y2 < min_bottom_y:
            continue
        filtered.append(det)

    non_coco_results[:] = filtered


def _drop_aux_non_coco_labels(non_coco_results, cfg, non_coco_model=None):
    """Remove internal helper labels that should never be exported or visualized."""
    non_coco_cfg = cfg.get("perception", {}).get("non_coco_dart", {})
    aux_labels = set(str(v).strip() for v in non_coco_cfg.get("aux_labels", ["arrow_sign"]))
    if non_coco_model is not None and hasattr(non_coco_model, "get_aux_labels"):
        aux_labels.update(str(v).strip() for v in non_coco_model.get_aux_labels())
    if not aux_labels:
        return

    non_coco_results[:] = [
        det for det in non_coco_results
        if str(getattr(det, "label", "")).strip() not in aux_labels
    ]


def _restrict_ground_markings_to_scenes(non_coco_results, scene_name: str):
    """Keep ground-arrow/ground-text classes only in configured scenes."""
    allowed_scenes = {"scene3", "scene7", "scene11"}
    scene_key = str(scene_name).strip().lower()
    if scene_key in allowed_scenes:
        return

    blocked_labels = {"ground_arrow", "ground_text"}
    non_coco_results[:] = [
        det for det in non_coco_results
        if str(getattr(det, "label", "")).strip() not in blocked_labels
    ]


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


def _lane_points(lane):
    if isinstance(lane, dict):
        return lane.get("points", [])
    return getattr(lane, "points", [])


def _lane_overlaps_boxes(lane, boxes, pad_px: float, min_points: int, min_ratio: float) -> bool:
    points = _lane_points(lane)
    if not points:
        return False

    total = 0
    hits = 0
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        total += 1
        for box in boxes:
            x1 = float(box[0]) - pad_px
            y1 = float(box[1]) - pad_px
            x2 = float(box[2]) + pad_px
            y2 = float(box[3]) + pad_px
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits += 1
                break

    if total <= 0:
        return False
    if hits >= max(1, int(min_points)):
        return True
    return (hits / float(total)) >= float(min_ratio)


def _suppress_lanes_near_ground_text(lanes, non_coco_results, cfg):
    lanes_cfg = cfg.get("perception", {}).get("lanes", {})
    if not bool(lanes_cfg.get("suppress_on_ground_text_only", True)):
        return lanes

    pad_px = float(lanes_cfg.get("ground_text_suppress_pad_px", 4.0))
    min_points = int(lanes_cfg.get("ground_text_suppress_min_points", 2))
    min_ratio = float(lanes_cfg.get("ground_text_suppress_point_ratio", 0.15))

    ground_text_boxes = [
        [float(v) for v in getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])]
        for det in non_coco_results
        if getattr(det, "label", "") == "ground_text"
        and bool(getattr(det, "has_only_letters", False))
        and getattr(det, "bbox", None) is not None
    ]
    if not ground_text_boxes:
        return lanes

    return [
        lane for lane in lanes
        if not _lane_overlaps_boxes(lane, ground_text_boxes, pad_px, min_points, min_ratio)
    ]


def _lane_raw_crosswalk_boxes(lane_raw, min_conf: float):
    if not isinstance(lane_raw, dict):
        return []

    boxes = lane_raw.get("boxes")
    scores = lane_raw.get("scores")
    class_names = lane_raw.get("class_names", [])
    if boxes is None or scores is None:
        return []

    min_conf = float(min_conf)
    crosswalk_boxes = []
    for i in range(len(scores)):
        score = float(scores[i].item())
        if score < min_conf:
            continue

        cls_name = class_names[i] if i < len(class_names) else ""
        cls_name_norm = str(cls_name).strip().lower()
        if "crosswalk" not in cls_name_norm:
            continue

        box_vals = boxes[i].detach().cpu().tolist()
        if len(box_vals) < 4:
            continue
        crosswalk_boxes.append([float(v) for v in box_vals[:4]])

    return crosswalk_boxes


def _suppress_lanes_on_crosswalk_markings(lanes, lane_raw, cfg):
    lanes_cfg = cfg.get("perception", {}).get("lanes", {})
    if not bool(lanes_cfg.get("suppress_on_crosswalk_marking", True)):
        return lanes

    pad_px = float(lanes_cfg.get("crosswalk_suppress_pad_px", 4.0))
    min_points = int(lanes_cfg.get("crosswalk_suppress_min_points", 2))
    min_ratio = float(lanes_cfg.get("crosswalk_suppress_point_ratio", 0.15))
    min_conf = float(lanes_cfg.get("crosswalk_suppress_min_confidence", 0.85))

    crosswalk_boxes = _lane_raw_crosswalk_boxes(lane_raw, min_conf=min_conf)
    if not crosswalk_boxes:
        return lanes

    return [
        lane for lane in lanes
        if not _lane_overlaps_boxes(lane, crosswalk_boxes, pad_px, min_points, min_ratio)
    ]


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


def _slugify_class_name(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return clean.strip("_")


def _lane_confidence(lane) -> float:
    if isinstance(lane, dict):
        return float(lane.get("confidence", 0.0))
    return float(getattr(lane, "confidence", 0.0))


def _lane_raw_index(lane) -> int:
    if isinstance(lane, dict):
        return int(lane.get("raw_index", -1))
    return int(getattr(lane, "raw_index", -1))


def _lane_color_name(lane) -> str:
    if isinstance(lane, dict):
        return str(lane.get("color", "white")).strip().lower()
    return str(getattr(lane, "color", "white")).strip().lower()


def _lane_min_confidence(lane, default_min_conf: float, yellow_min_conf: float, white_min_conf: float) -> float:
    color = _lane_color_name(lane)
    if color == "yellow":
        return float(yellow_min_conf)
    if color == "white":
        return float(white_min_conf)
    return float(default_min_conf)


def _filter_lanes_by_confidence(
    lanes,
    default_min_confidence: float,
    yellow_min_confidence: float,
    white_min_confidence: float,
):
    kept = []
    for lane in lanes:
        score = _lane_confidence(lane)
        min_conf = _lane_min_confidence(
            lane,
            default_min_conf=default_min_confidence,
            yellow_min_conf=yellow_min_confidence,
            white_min_conf=white_min_confidence,
        )
        if score >= min_conf:
            kept.append(lane)
    return kept


def _raw_lane_class_min_confidence(class_name: str, default_min_conf: float, yellow_min_conf: float, white_min_conf: float) -> float:
    cls = str(class_name).strip().lower()
    if "yellow" in cls or "non-white" in cls:
        return float(yellow_min_conf)
    if "white" in cls:
        return float(white_min_conf)
    return float(default_min_conf)


def _raw_box_has_lane_match(box_xyxy, lane_results, pad_px: float, min_points: int, min_ratio: float) -> bool:
    box = [float(v) for v in box_xyxy]
    for lane in lane_results:
        if _lane_overlaps_boxes(lane, [box], pad_px=pad_px, min_points=min_points, min_ratio=min_ratio):
            return True
    return False


def _lane_raw_index_set(lanes) -> set:
    out = set()
    for lane in lanes:
        idx = _lane_raw_index(lane)
        if idx >= 0:
            out.add(idx)
    return out


def _build_lane_drop_reason_map(
    lane_raw,
    lane_results_detected,
    lane_results_after_ground_text,
    lane_results_after_crosswalk,
    lane_results_after_conf,
    detector_min_conf: float,
    default_min_confidence: float,
    yellow_min_confidence: float,
    white_min_confidence: float,
):
    if not isinstance(lane_raw, dict) or "scores" not in lane_raw:
        return {}

    scores = lane_raw.get("scores")
    names = lane_raw.get("class_names", [])
    if scores is None:
        return {}

    detected_set = _lane_raw_index_set(lane_results_detected)
    ground_set = _lane_raw_index_set(lane_results_after_ground_text)
    crosswalk_set = _lane_raw_index_set(lane_results_after_crosswalk)
    conf_set = _lane_raw_index_set(lane_results_after_conf)

    reason_map = {}

    for i in range(len(scores)):
        score = float(scores[i].item())
        cls_name = names[i] if i < len(names) else "lane"
        cls_norm = str(cls_name).strip().lower()

        if "lane" not in cls_norm:
            continue

        if score < float(detector_min_conf):
            reason_map[i] = "below_detector_conf"
            continue

        if i not in detected_set:
            reason_map[i] = "no_polyline_points"
            continue

        if i not in ground_set:
            reason_map[i] = "suppressed_ground_text"
            continue

        if i not in crosswalk_set:
            reason_map[i] = "suppressed_crosswalk"
            continue

        if i not in conf_set:
            min_conf = _raw_lane_class_min_confidence(
                cls_name,
                default_min_conf=default_min_confidence,
                yellow_min_conf=yellow_min_confidence,
                white_min_conf=white_min_confidence,
            )
            reason_map[i] = f"below_export_conf<{min_conf:.2f}"
            continue

    for lane in lane_results_after_conf:
        idx = _lane_raw_index(lane)
        if idx < 0:
            continue
        if len(_lane_points(lane)) < 2:
            reason_map[idx] = "dropped_export_points<2"

    return reason_map


def draw_lane_debug_overlay(
    frame_bgr,
    lane_results,
    lane_raw=None,
    lane_exclude_classes=None,
    default_min_confidence: float = 0.85,
    yellow_min_confidence: float = 0.65,
    white_min_confidence: float = 0.85,
    raw_match_pad_px: float = 6.0,
    raw_match_min_points: int = 2,
    raw_match_point_ratio: float = 0.10,
    draw_unmatched_raw_boxes: bool = False,
    raw_drop_reasons: dict = None,
    show_drop_reasons: bool = True,
):
    """Draw lane polylines and optional raw detector boxes on a frame."""
    vis = frame_bgr.copy()
    exclude_keys = {
        _slugify_class_name(v)
        for v in (lane_exclude_classes or [])
        if str(v).strip()
    }

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
        raw_drop_reasons = raw_drop_reasons or {}
        for i in range(len(scores)):
            box = boxes[i].detach().cpu().tolist()
            score = float(scores[i].item())
            cls_name = names[i] if i < len(names) else "lane"
            if _slugify_class_name(cls_name) in exclude_keys:
                continue
            min_conf = _raw_lane_class_min_confidence(
                cls_name,
                default_min_conf=default_min_confidence,
                yellow_min_conf=yellow_min_confidence,
                white_min_conf=white_min_confidence,
            )
            if score < min_conf:
                continue
            matched_lane = _raw_box_has_lane_match(
                box,
                lane_results,
                pad_px=raw_match_pad_px,
                min_points=raw_match_min_points,
                min_ratio=raw_match_point_ratio,
            )
            has_reason = i in raw_drop_reasons
            if (not matched_lane) and (not draw_unmatched_raw_boxes) and not (show_drop_reasons and has_reason):
                continue

            draw_color = (50, 255, 50) if matched_lane else (0, 140, 255)
            reason_suffix = ""
            if (not matched_lane) and show_drop_reasons and has_reason:
                reason_suffix = f" | {raw_drop_reasons[i]}"

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), draw_color, 1)
            cv2.putText(
                vis,
                f"{cls_name} {score:.2f}{reason_suffix}",
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                draw_color,
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
    lanes_cfg = cfg.get("perception", {}).get("lanes", {})
    lane_export_min_conf = float(lanes_cfg.get("export_min_confidence", 0.85))
    lane_export_min_conf_yellow = float(lanes_cfg.get("export_min_confidence_yellow", 0.65))
    lane_export_min_conf_white = float(lanes_cfg.get("export_min_confidence_white", lane_export_min_conf))
    lane_debug_raw_match_pad_px = float(lanes_cfg.get("debug_raw_match_pad_px", 6.0))
    lane_debug_raw_match_min_points = int(lanes_cfg.get("debug_raw_match_min_points", 2))
    lane_debug_raw_match_point_ratio = float(lanes_cfg.get("debug_raw_match_point_ratio", 0.10))
    lane_debug_draw_unmatched_raw_boxes = bool(lanes_cfg.get("debug_draw_unmatched_raw_boxes", False))
    lane_debug_show_drop_reasons = bool(lanes_cfg.get("debug_show_drop_reasons", True))
    models["traffic"].set_scene_context(scene_name)

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
        _restrict_ground_markings_to_scenes(non_coco_results, scene_name)
        traffic_candidates, non_coco_results = _build_traffic_candidates(
            object_results,
            non_coco_results,
            cfg,
            scene_name,
        )
        _filter_ground_arrows(frame_bgr, non_coco_results, cfg)
        _drop_aux_non_coco_labels(non_coco_results, cfg, models.get("non_coco"))
        _run_speed_limit_ocr(
            frame_bgr,
            non_coco_results,
            models.get("speed_limit_ocr"),
            frame_idx=frame_idx,
            debug_dir=ocr_debug_dir if debug else None,
        )
        _run_ground_text_ocr(
            frame_bgr,
            non_coco_results,
            models.get("speed_limit_ocr"),
            cfg,
            frame_idx=frame_idx,
            debug_dir=ocr_debug_dir if debug else None,
        )
        _suppress_ground_arrow_text_overlaps(non_coco_results, cfg)
        depth_map = models["depth"].estimate(frame_bgr)
        object_results = models["depth"].lift_to_3d(object_results, depth_map)
        non_coco_results = models["depth"].lift_to_3d(non_coco_results, depth_map)
        traffic_results = models["depth"].lift_to_3d(models["traffic"].detect(frame_bgr, traffic_candidates), depth_map)
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

        lane_results_detected = list(lane_results)
        lane_results_after_ground_text = _suppress_lanes_near_ground_text(lane_results_detected, non_coco_results, cfg)
        lane_results_after_crosswalk = _suppress_lanes_on_crosswalk_markings(lane_results_after_ground_text, lane_raw, cfg)
        lane_results = _filter_lanes_by_confidence(
            lane_results_after_crosswalk,
            default_min_confidence=lane_export_min_conf,
            yellow_min_confidence=lane_export_min_conf_yellow,
            white_min_confidence=lane_export_min_conf_white,
        )
        lane_drop_reasons = _build_lane_drop_reason_map(
            lane_raw,
            lane_results_detected,
            lane_results_after_ground_text,
            lane_results_after_crosswalk,
            lane_results,
            detector_min_conf=float(lanes_cfg.get("confidence", 0.30)),
            default_min_confidence=lane_export_min_conf,
            yellow_min_confidence=lane_export_min_conf_yellow,
            white_min_confidence=lane_export_min_conf_white,
        )

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
            lane_exclude_classes = cfg.get("perception", {}).get("lanes", {}).get("exclude_classes", [])
            overlay = draw_lane_debug_overlay(
                frame_bgr,
                lane_results,
                lane_raw,
                lane_exclude_classes=lane_exclude_classes,
                default_min_confidence=lane_export_min_conf,
                yellow_min_confidence=lane_export_min_conf_yellow,
                white_min_confidence=lane_export_min_conf_white,
                raw_match_pad_px=lane_debug_raw_match_pad_px,
                raw_match_min_points=lane_debug_raw_match_min_points,
                raw_match_point_ratio=lane_debug_raw_match_point_ratio,
                draw_unmatched_raw_boxes=lane_debug_draw_unmatched_raw_boxes,
                raw_drop_reasons=lane_drop_reasons,
                show_drop_reasons=lane_debug_show_drop_reasons,
            )
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
