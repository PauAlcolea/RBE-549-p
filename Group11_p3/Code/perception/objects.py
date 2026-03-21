# """
# perception/objects.py
# =====================
# Detects vehicles and pedestrians using YOLOv9 (via Ultralytics).

# Phase 1 outputs:
#   - All vehicles as class "car" (no sub-classification yet)
#   - Pedestrians as class "person"

# Phase 2 will add sub-classification (sedan/SUV/truck/etc.) in this same file.
# """

# from dataclasses import dataclass, field
# from typing import List
# import numpy as np


# # COCO class IDs we care about
# _VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# _PERSON_ID = 0


# @dataclass
# class Detection:
#     """Single detected object, in image coordinates."""
#     label: str                     # "car" | "person"
#     bbox: List[float]              # [x1, y1, x2, y2] pixels
#     confidence: float
#     depth_m: float = 0.0           # filled in by DepthEstimator.lift_to_3d
#     position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# class ObjectDetector:
#     """
#     Wraps an Ultralytics YOLO model for vehicle + pedestrian detection.

#     Usage
#     -----
#     detector = ObjectDetector(cfg, device="cuda")
#     results  = detector.detect(frame_bgr)   # list[Detection]
#     """

#     def __init__(self, cfg: dict, device: str = "cuda"):
#         self.cfg = cfg
#         self.device = device
#         self.conf_thresh = cfg["perception"]["yolo"]["confidence"]
#         self.iou_thresh  = cfg["perception"]["yolo"]["iou_threshold"]
#         self.model = self._load_model(cfg["weights"]["yolo"])

#     def _load_model(self, weights_path: str):
#         # TODO: uncomment once ultralytics is installed on cluster
#         # from ultralytics import YOLO
#         # model = YOLO(weights_path)
#         # model.to(self.device)
#         # return model
#         print(f"[ObjectDetector] STUB: would load YOLO from {weights_path}")
#         return None

#     def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
#         """
#         Run inference on one BGR frame.

#         Returns
#         -------
#         list[Detection]
#             One entry per detected vehicle or pedestrian, image coords.
#         """
#         if self.model is None:
#             # Stub return for development/testing
#             return []

#         # TODO: implement
#         # results = self.model.predict(
#         #     frame_bgr,
#         #     conf=self.conf_thresh,
#         #     iou=self.iou_thresh,
#         #     classes=self.cfg["perception"]["yolo"]["classes_phase1"],
#         #     verbose=False,
#         # )
#         # detections = []
#         # for box in results[0].boxes:
#         #     cls_id = int(box.cls)
#         #     label = "person" if cls_id == _PERSON_ID else "car"
#         #     detections.append(Detection(
#         #         label=label,
#         #         bbox=box.xyxy[0].tolist(),
#         #         confidence=float(box.conf),
#         #     ))
#         # return detections
#         raise NotImplementedError("ObjectDetector.detect not yet implemented")
