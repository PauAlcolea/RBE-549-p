"""
perception/vehicle_subtypes.py
==============================
Use DART as a second-stage vehicle subtype refiner.

The base object detector still finds generic COCO vehicles. This module runs a
prompt-based DART pass over the full frame, then matches any subtype detections
back onto selected base labels such as "car".
"""

from copy import deepcopy
from typing import List

import numpy as np

from .objects import Detection
from .objectsDART import NonCocoDartDetector


def _bbox_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


class VehicleSubtypeClassifier:
    """Relabel generic vehicle detections with DART subtype prompts."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.subtype_cfg = cfg.get("perception", {}).get("vehicle_subtypes_dart", {})
        self.enabled = bool(self.subtype_cfg.get("enabled", False))
        self.target_labels = {
            str(label).strip().lower()
            for label in self.subtype_cfg.get("target_labels", ["car"])
            if str(label).strip()
        }
        self.match_iou = float(self.subtype_cfg.get("match_iou", 0.25))
        self.allowed_matches = self._parse_allowed_matches(
            self.subtype_cfg.get("allowed_matches", {})
        )
        self.detector = None

        if self.enabled:
            detector_cfg = deepcopy(cfg)
            detector_cfg.setdefault("perception", {})
            detector_cfg["perception"] = dict(detector_cfg["perception"])
            detector_cfg["perception"]["non_coco_dart"] = deepcopy(self.subtype_cfg)
            self.detector = NonCocoDartDetector(
                detector_cfg,
                device,
                log_prefix="vehicle-subtypes-dart",
            )

    @staticmethod
    def _parse_allowed_matches(raw_matches: dict) -> dict:
        parsed = {}
        for base_label, subtype_labels in raw_matches.items():
            base_key = str(base_label).strip().lower()
            if not base_key:
                continue
            parsed[base_key] = {
                str(subtype).strip().lower()
                for subtype in subtype_labels
                if str(subtype).strip()
            }
        return parsed

    def refine_detections(self, frame_bgr: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """
        Match DART subtype detections onto existing generic vehicle detections.

        The highest-confidence subtype detection that overlaps a target vehicle
        with sufficient IoU wins and relabels that object.
        """
        if not self.enabled or self.detector is None or not detections:
            return detections

        subtype_detections = self.detector.detect(frame_bgr)
        if not subtype_detections:
            return detections

        used_subtypes = set()
        for det in detections:
            if det.label not in self.target_labels:
                continue

            best_idx = None
            best_score = -1.0

            for idx, subtype in enumerate(subtype_detections):
                if idx in used_subtypes:
                    continue

                iou = _bbox_iou(det.bbox, subtype.bbox)
                if iou < self.match_iou:
                    continue
                allowed_subtypes = self.allowed_matches.get(det.label)
                if allowed_subtypes is not None and subtype.label not in allowed_subtypes:
                    continue

                score = float(subtype.confidence) + 0.1 * iou
                if score > best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is not None:
                det.label = subtype_detections[best_idx].label
                used_subtypes.add(best_idx)

        return detections
