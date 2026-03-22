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
from typing import List                     # this is just to have the ability to say types
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

    # field makes it so that each detection gets its own list
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


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
        self.conf_thresh = cfg["perception"]["yolo"]["confidence"]
        self.iou_thresh  = cfg["perception"]["yolo"]["iou_threshold"]
        self.model = self._load_model(cfg["weights"]["yolo"])

    def _load_model(self, weights_path: str):
        # TODO: uncomment once ultralytics is installed on cluster
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.to(self.device)
        return model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run inference on one BGR frame.

        Returns
        -------
        list[Detection]
            One entry per detected vehicle or pedestrian, image coords.
        """
        # results is a list of one Result object per image, we pass one image we get a list of one Result object
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.cfg["perception"]["yolo"]["classes_phase1"],
            verbose=False,
        )

        detections = []

        # each box is what the network detected as something
        for box in results[0].boxes:
            cls_id = int(box.cls)
            
            # TODO: this is a simplified classification, will want to expand to be more specific
            if cls_id == _PERSON_ID:
                label = "person"
            elif cls_id in _VEHICLE_IDS:
                label = "car"
            else:
                continue

            # we make a detection object for each thing with the proper classicifaction above
            detections.append(Detection(
                label=label,
                bbox=box.xyxy[0].tolist(),
                confidence=float(box.conf),
            ))
        return detections
