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

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class TrafficLight:
    bbox: List[float]       # [x1, y1, x2, y2]
    color: str              # "red" | "yellow" | "green" | "unknown"
    confidence: float
    depth_m: float = 0.0


# COCO class ID for traffic light
_TRAFFIC_LIGHT_ID = 9


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

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.hsv_ranges = self._build_hsv_ranges(cfg["perception"]["traffic_light"])
        # We reuse the YOLO model from ObjectDetector — no second load needed.
        # The caller (run_perception.py) passes in the full object detections
        # which already contain traffic lights if class 9 was included.
        # Alternatively, we can run a second YOLO pass here.
        self.model = self._load_model(cfg["weights"]["yolo"])

    def _load_model(self, weights_path: str):
        # TODO: same YOLO weights as objects.py — but now include class 9
        # Consider passing in the already-loaded model to avoid loading twice.
        print(f"[TrafficLightDetector] STUB: would reuse YOLO from {weights_path}")
        return None

    def _build_hsv_ranges(self, tl_cfg: dict):
        """Package HSV range lists into a dict for color lookup."""
        import numpy as np
        return {
            "red": [
                (np.array(tl_cfg["hsv_red_low"]),  np.array(tl_cfg["hsv_red_high"])),
                (np.array(tl_cfg["hsv_red_low2"]), np.array(tl_cfg["hsv_red_high2"])),
            ],
            "yellow": [
                (np.array(tl_cfg["hsv_yellow_low"]), np.array(tl_cfg["hsv_yellow_high"])),
            ],
            "green": [
                (np.array(tl_cfg["hsv_green_low"]), np.array(tl_cfg["hsv_green_high"])),
            ],
        }

    def detect(self, frame_bgr: np.ndarray, object_detections: list = None) -> List[TrafficLight]:
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
        # TODO: implement
        # 1. Get traffic light bboxes (from YOLO class 9 or from object_detections)
        # 2. For each bbox: crop frame, classify color with _classify_color
        # 3. Return list of TrafficLight objects
        return []

    def _classify_color(self, frame_bgr: np.ndarray, bbox: List[float]) -> str:
        """
        Classify the lit color of a single traffic light crop using HSV.

        Returns "red", "yellow", "green", or "unknown".
        """
        # TODO: implement
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        scores = {}
        for color, ranges in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            scores[color] = mask.sum()
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"
        # raise NotImplementedError("TrafficLightDetector._classify_color not yet implemented")
