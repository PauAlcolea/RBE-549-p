# """
# perception/signs.py
# ===================
# Road sign detection — Phase 1 focuses on stop signs.

# Detection strategy:
#   - Primary:  YOLO (COCO class 11 = "stop sign") — fast and accurate.
#   - Fallback: HSV + shape filter for the red octagon, in case YOLO misses
#               partially occluded signs.

# Phase 2 will add speed limit signs (text recognition inside the bbox)
# and ground arrows (segmentation or homography-based detection).
# """

# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import List
# import numpy as np


# @dataclass
# class Sign:
#     label: str              # "stop_sign" (Phase 1); more types in Phase 2
#     bbox: List[float]       # [x1, y1, x2, y2]
#     confidence: float
#     depth_m: float = 0.0
#     position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# # COCO class ID for stop sign
# _STOP_SIGN_ID = 11


# class SignDetector:
#     """
#     Detects stop signs using YOLO.

#     Usage
#     -----
#     detector = SignDetector(cfg, device="cuda")
#     signs = detector.detect(frame_bgr, object_detections)
#     """

#     def __init__(self, cfg: dict, device: str = "cuda"):
#         self.cfg = cfg
#         self.device = device
#         self.model = self._load_model(cfg["weights"]["yolo"])

#     def _load_model(self, weights_path: str):
#         # TODO: same YOLO weights — filter for class 11
#         print(f"[SignDetector] STUB: would reuse YOLO from {weights_path}")
#         return None

#     def detect(self, frame_bgr: np.ndarray, object_detections: list = None) -> List[Sign]:
#         """
#         Detect stop signs in one BGR frame.

#         Returns
#         -------
#         list[Sign]
#         """
#         # TODO: implement
#         # Filter object_detections for class 11 (stop sign), or run separate YOLO pass.
#         # Optionally run _octagon_fallback for robustness.
#         return []

#     def _octagon_fallback(self, frame_bgr: np.ndarray) -> List[Sign]:
#         """
#         Backup detector using red HSV mask + contour approximation.
#         Looks for 8-sided red shapes in the image.
#         Returns list of Sign objects.
#         """
#         # TODO: implement
#         # import cv2
#         # hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
#         # red_mask = ... (same HSV ranges as traffic.py)
#         # contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # signs = []
#         # for cnt in contours:
#         #     approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
#         #     if len(approx) == 8 and cv2.contourArea(cnt) > 500:
#         #         x, y, w, h = cv2.boundingRect(cnt)
#         #         signs.append(Sign(label="stop_sign", bbox=[x, y, x+w, y+h], confidence=0.6))
#         # return signs
#         raise NotImplementedError
