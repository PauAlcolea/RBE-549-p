"""
perception/signs.py
===================
Road sign detection — Phase 1 focuses on stop signs.

Detection strategy:
  - Primary:  YOLO (COCO class 11 = "stop sign") — fast and accurate.
  - Fallback: HSV + shape filter for the red octagon, in case YOLO misses
              partially occluded signs.

Phase 2 will add speed limit signs (text recognition inside the bbox)
and ground arrows (segmentation or homography-based detection).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import cv2


@dataclass
class Sign:
    label: str              # "stop_sign" (Phase 1); more types in Phase 2
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float
    depth_m: float = 0.0
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    track_id: Optional[int] = None

class SignDetector:
    """
    Detects stop signs using YOLO.

    Usage
    -----
    detector = SignDetector(cfg, device="cuda")
    signs = detector.detect(frame_bgr, object_detections)
    """

    def __init__(self, cfg: dict):
        sc = cfg["perception"]["stop_sign"]
        self.min_area         = sc["min_contour_area"]
        self.max_area            = sc["max_contour_area"]
        self.dp_epsilon       = sc["dp_epsilon_factor"]
        self.min_sides        = sc["min_sides"]
        self.max_sides        = sc["max_sides"]
        self.aspect_min       = sc["aspect_ratio_min"]
        self.aspect_max       = sc["aspect_ratio_max"]
        self.min_solidity        = sc["min_solidity"]
        self.fallback_conf    = sc["fallback_confidence"]
        self.fallback_window     = sc["fallback_window_frames"]

        # Reuse the red HSV ranges from traffic_light config for the octagon fallback
        tl_cfg = cfg["perception"]["traffic_light"]
        self.red_ranges = [
            (np.array(tl_cfg["hsv_red_low"],   dtype=np.uint8),
             np.array(tl_cfg["hsv_red_high"],  dtype=np.uint8)),
            (np.array(tl_cfg["hsv_red_low2"],  dtype=np.uint8),
             np.array(tl_cfg["hsv_red_high2"], dtype=np.uint8)),
        ]

        # How many frames have passed since YOLO last confirmed a stop sign.
        # Start at a large number so the fallback is inactive until YOLO fires.
        self._frames_since_yolo = self.fallback_window + 1

    def detect(self, frame_bgr: np.ndarray, object_detections: list) -> List[Sign]:
        """
        Detect stop signs in one BGR frame.

        Parameters
        ----------
        frame_bgr         : BGR image (H x W x 3, uint8)
        object_detections : list[Detection] from ObjectDetector.detect().

        Returns
        -------
        list[Sign]
        """
        signs: List[Sign] = []
        yolo_bboxes = set()
        yolo_found = False

        # --- Stage 1: YOLO detections ---
        for det in object_detections:
            if det.label == "stop_sign":
                signs.append(Sign(
                    label="stop_sign",
                    bbox=det.bbox,
                    confidence=det.confidence,
                    depth_m=det.depth_m,
                    position_3d=det.position_3d,
                    track_id=getattr(det, "track_id", None),
                ))
                yolo_bboxes.add(tuple(int(v) for v in det.bbox))
                yolo_found = True

        # Update the frame counter regardless of whether we use the fallback.
        # If YOLO found a sign this frame, reset to 0.
        # Otherwise increment — the fallback window counts up from the last sighting.
        if yolo_found:
            self._frames_since_yolo = 0
        else:
            self._frames_since_yolo += 1

        # --- Stage 2: fallback, only within the window after a YOLO sighting ---
        # _frames_since_yolo == 0 means YOLO just found one this frame (no need for fallback).
        # 1 <= _frames_since_yolo <= window means YOLO recently found one but not this frame
        # — this is the occlusion scenario the fallback is designed for.
        # > window means no sign has been seen recently — fallback stays off.
        if 1 <= self._frames_since_yolo <= self.fallback_window:
            for fs in self._octagon_fallback(frame_bgr):
                if not self._overlaps_any(fs.bbox, yolo_bboxes):
                    signs.append(fs)

        return signs

    def _octagon_fallback(self, frame_bgr: np.ndarray) -> List[Sign]:
        """
        Tightly-filtered fallback: red HSV mask + shape analysis.

        Only called when YOLO saw a stop sign recently (within fallback_window_frames).

        Filters applied in order (all must pass):
          1. Area bounds          — rejects noise and large red objects like vehicles
          2. Solidity             — rejects brake lights, irregular blobs, thin shapes
          3. Aspect ratio         — rejects elongated red shapes
          4. Vertex count         — rejects rectangles and overly complex shapes
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.red_ranges:
            red_mask |= cv2.inRange(hsv, lo, hi)

        # Morphological close to fill small gaps inside the sign housing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signs: List[Sign] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter 1: area bounds
            if not (self.min_area <= area <= self.max_area):
                continue

            # Filter 2: solidity — a filled octagon scores ~0.95; irregular shapes score lower
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            if (area / hull_area) < self.min_solidity:
                continue

            # Filter 3: aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if not (self.aspect_min < aspect < self.aspect_max):
                continue

            # Filter 4: vertex count after Douglas-Peucker approximation
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.dp_epsilon * peri, True)
            if not (self.min_sides <= len(approx) <= self.max_sides):
                continue

            signs.append(Sign(
                label="stop_sign",
                bbox=[float(x), float(y), float(x + w), float(y + h)],
                confidence=self.fallback_conf,
            ))

        return signs

    @staticmethod
    def _overlaps_any(bbox: List[float], existing: set, iou_thresh: float = 0.3) -> bool:
        """
        Returns True if bbox has IoU > iou_thresh with any bbox in existing.
        Used to suppress fallback detections already found by YOLO.
        """
        x1, y1, x2, y2 = bbox
        area_a = max(0, x2 - x1) * max(0, y2 - y1)
        if area_a == 0:
            return False
        for (ex1, ey1, ex2, ey2) in existing:
            inter_x1 = max(x1, ex1)
            inter_y1 = max(y1, ey1)
            inter_x2 = min(x2, ex2)
            inter_y2 = min(y2, ey2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_b = max(0, ex2 - ex1) * max(0, ey2 - ey1)
            union = area_a + area_b - inter_area
            if union > 0 and inter_area / union > iou_thresh:
                return True
        return False