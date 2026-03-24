"""
perception/objects.py
=====================
Detects vehicles and pedestrians using YOLOv9 (via Ultralytics).

Phase 1 outputs:
  - All vehicles as class "car" (no sub-classification yet)
  - Pedestrians as class "person"

Phase 2 will add sub-classification (sedan/SUV/truck/etc.) in this same file.
"""

from dataclasses import dataclass, field
from typing import List, Optional                     # this is just to have the ability to say types
import numpy as np


# COCO class IDs we care about
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
_VEHICLE_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", }
_PERSON_ID = 0
_TRAFFIC_IDS = {9: "traffic_light", 11: "stop_sign", }


# dataClass makes working with classes easier, no need for __init__ constructor and easier printing for debugging
@dataclass
class Detection:
    """Single detected object, in image coordinates."""
    label: str                      # "car" | "person"
    bbox: List[float]               # [x1, y1, x2, y2] pixels top left and bottom right corners
    confidence: float               # YOLO confidence score from 0 to 1
    depth_m: float = 0.0            # filled in by DepthEstimator.lift_to_3d
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])     # field makes it so that each detection gets its own list

def _cls_id_to_label(cls_id: int) -> Optional[str]:
    """Map a COCO class ID to our label string, or None if we don't care about it."""
    if cls_id == _PERSON_ID:
        return "person"
    if cls_id in _VEHICLE_IDS:
        return _VEHICLE_IDS[cls_id]
    if cls_id in _TRAFFIC_IDS:
        return _TRAFFIC_IDS[cls_id]
    return None


def _compute_iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class ObjectDetector:
    """
    Wraps an Ultralytics YOLO model for vehicle + pedestrian + sign detection.

    Usage
    -----
    detector = ObjectDetector(cfg, device="cuda")
    results  = detector.detect(frame_bgr)   # list[Detection]
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        
        yolo_cfg   = cfg["perception"]["yolo"]
        rtdetr_cfg = cfg["perception"]["rtdetr"]

        self.yolo_conf      = yolo_cfg["confidence"]
        self.yolo_iou       = yolo_cfg["iou_threshold"]
        self.rtdetr_conf    = rtdetr_cfg["confidence"]
        self.rtdetr_iou     = rtdetr_cfg["iou_threshold"]
        self.match_iou      = rtdetr_cfg["match_iou"]        # IoU needed to call two boxes "the same object"
        self.merged_conf    = rtdetr_cfg["merged_confidence"] # final gate after averaging both scores
        self.classes        = yolo_cfg["classes_phase1"]

        self.yolo = self._load_yolo(cfg["weights"]["yolo"])
        self.rtdetr = self._load_rtdetr(cfg["weights"]["rtdetr"])

    def _load_yolo(self, weights_path: str):
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    def _load_rtdetr(self, weights_path: str):
        from ultralytics import RTDETR
        model = RTDETR(weights_path)
        model.to(self.device)
        return model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run both models on one BGR frame and return verified detections.

        :return     list[Detection], (Only objects confirmed by both YOLO and RT-DETR.)
        """
        yolo_dets   = self._run_yolo(frame_bgr)
        rtdetr_dets = self._run_rtdetr(frame_bgr)
        return self._merge(yolo_dets, rtdetr_dets)


    # Run each model and return raw (bbox, label, conf) tuples:

    def _run_yolo(self, frame_bgr: np.ndarray):
        """Run YOLOv9c and return list of (bbox, label, conf)."""
        results = self.yolo.predict(
            frame_bgr,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            classes=self.classes,
            verbose=False,
        )
        out = []
        for box in results[0].boxes:
            label = _cls_id_to_label(int(box.cls))
            if label is None:
                continue
            out.append((box.xyxy[0].tolist(), label, float(box.conf)))
        return out

    def _run_rtdetr(self, frame_bgr: np.ndarray):
        """Run RT-DETR and return list of (bbox, label, conf)."""
        results = self.rtdetr.predict(
            frame_bgr,
            conf=self.rtdetr_conf,
            iou=self.rtdetr_iou,
            classes=self.classes,
            verbose=False,
        )
        out = []
        for box in results[0].boxes:
            label = _cls_id_to_label(int(box.cls))
            if label is None:
                continue
            out.append((box.xyxy[0].tolist(), label, float(box.conf)))
        return out
    
    def _merge(self, yolo_dets: list, rtdetr_dets: list) -> List[Detection]:
        """
        Keep only detections where both models agree.

        For each YOLO detection, look for an RT-DETR detection with:
          - Same label
          - IoU >= self.match_iou

        If found, keep it with confidence = average of both scores,
        provided that merged confidence >= self.merged_conf.

        The bbox used is from YOLO (slightly tighter on average for CNNs).
        """
        matched = []
        used_rtdetr = set()  # prevent one RT-DETR box from matching multiple YOLO boxes

        for y_bbox, y_label, y_conf in yolo_dets:
            best_iou  = 0.0
            best_idx  = -1
            best_conf = 0.0

            for i, (r_bbox, r_label, r_conf) in enumerate(rtdetr_dets):
                if i in used_rtdetr:
                    continue
                if r_label != y_label:
                    continue
                iou = _compute_iou(y_bbox, r_bbox)
                if iou > best_iou:
                    best_iou  = iou
                    best_idx  = i
                    best_conf = r_conf

            if best_iou >= self.match_iou:
                merged_conf = (y_conf + best_conf) / 2.0
                if merged_conf >= self.merged_conf:
                    used_rtdetr.add(best_idx)
                    matched.append(Detection(
                        label=y_label,
                        bbox=y_bbox,
                        confidence=round(merged_conf, 4),
                    ))

        return matched