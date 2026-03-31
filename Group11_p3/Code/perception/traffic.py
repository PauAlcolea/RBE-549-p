"""
perception/traffic.py
=====================
Traffic light detection and color classification.

Detection:  YOLO (same model as objects.py) already returns traffic light bboxes
            if we include COCO class 9 ("traffic light").
Color:      HSV thresholding on the top third of the detected bbox crop.
            (The illuminated bulb is at the top for red, middle for yellow,
            bottom for green — but thresholding is more robust than position.)
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import cv2


@dataclass
class TrafficLight:
    bbox: List[float]       # [x1, y1, x2, y2]
    color: str              # "red" | "yellow" | "green" | "unknown"
    confidence: float
    depth_m: float = 0.0
    label: str = "traffic_light"   # for consistency with Detection
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # optional, can be filled in by DepthEstimator.lift_to_3d

class TrafficLightDetector:
    """
    Detects traffic lights and classifies their color.

    Two-stage approach:
      1. YOLO bounding box (reuses the object detector model)
      2. HSV color classification inside the bbox crop

    Usage
    -----
    detector = TrafficLightDetector(cfg, device="cuda")
    lights = detector.detect(frame_bgr, all_detections)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hsv_ranges = self._build_hsv_ranges(cfg["perception"]["traffic_light"])
        # Minimum pixel count to declare a color (avoids noise on tiny/dark crops)
        self.min_pixels = cfg["perception"]["traffic_light"]["min_color_pixels"]

    def _build_hsv_ranges(self, tl_cfg: dict):
        """Package HSV range lists into a dict for color lookup."""
        return {
            "red": [
                (np.array(tl_cfg["hsv_red_low"],    dtype=np.uint8),
                 np.array(tl_cfg["hsv_red_high"],   dtype=np.uint8)),
                # Red wraps around hue=0 in OpenCV HSV, so we need a second range
                (np.array(tl_cfg["hsv_red_low2"],   dtype=np.uint8),
                 np.array(tl_cfg["hsv_red_high2"],  dtype=np.uint8)),
            ],
            "yellow": [
                (np.array(tl_cfg["hsv_yellow_low"],  dtype=np.uint8),
                 np.array(tl_cfg["hsv_yellow_high"], dtype=np.uint8)),
            ],
            "green": [
                (np.array(tl_cfg["hsv_green_low"],  dtype=np.uint8),
                 np.array(tl_cfg["hsv_green_high"], dtype=np.uint8)),
            ],
        }

    def detect(self, frame_bgr: np.ndarray, object_detections: list) -> List[TrafficLight]:
        """
        Detect traffic lights in one BGR frame and classify their color.

        Parameters
        ----------
        frame_bgr : np.ndarray
        object_detections : list[Detection], optional
            If provided, filter for traffic light class instead of re-running YOLO.

        Returns
        -------
        list[TrafficLight]
        """
        lights = []

        for det in object_detections:
            if det.label != "traffic_light":
                continue

            color = self._classify_color(frame_bgr, det.bbox)
            lights.append(TrafficLight(
                bbox=det.bbox,
                color=color,
                confidence=det.confidence,
                depth_m=det.depth_m,
                label=det.label
            ))

        return lights

    def _classify_color(self, frame_bgr: np.ndarray, bbox: List[float]) -> str:
        """
        Classify the lit color of a single traffic light crop using HSV.

        Strategy:
          - Crop the bbox (clamped to image bounds).
          - Convert to HSV.
          - Count pixels inside each color's HSV range(s).
          - Return the color with the highest count if it exceeds min_pixels,
            else "unknown".

        Returns "red", "yellow", "green", or "unknown".
        """
        h_img, w_img = frame_bgr.shape[:2]

        # Clamp bbox to image bounds
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w_img, int(bbox[2]))
        y2 = min(h_img, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        crop = frame_bgr[y1:y2, x1:x2]

        # HSV is unreliable on very small crops (far-away lights)
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Count masked pixels per color
        scores: dict = {}
        for color_name, ranges in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            # mask values are 0 or 255, divide to get real pixel count
            scores[color_name] = int(np.sum(mask) // 255)

        best_color = max(scores, key=scores.get)
        return best_color if scores[best_color] >= self.min_pixels else "unknown"
