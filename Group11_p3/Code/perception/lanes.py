# """
# perception/lanes.py
# ===================
# Lane detection using CLRNet (or LaneATT as fallback).

# Outputs per frame:
#   - List of lane polylines (image coordinates)
#   - Each lane tagged with color ("white" | "yellow") and style ("solid" | "dashed")

# Color classification uses HSV thresholding on the lane region.
# Dashed vs solid is inferred from the density/continuity of detected points.
# """

# from dataclasses import dataclass, field
# from typing import List, Tuple
# import numpy as np


# @dataclass
# class Lane:
#     """A single detected lane line."""
#     points: List[Tuple[float, float]]   # [(x, y), ...] in image pixels, top to bottom
#     color: str = "white"                # "white" | "yellow"
#     style: str = "solid"                # "solid" | "dashed"
#     confidence: float = 1.0


# class LaneDetector:
#     """
#     Wraps CLRNet for lane detection.

#     Usage
#     -----
#     detector = LaneDetector(cfg, device="cuda")
#     lanes = detector.detect(frame_bgr)   # list[Lane]
#     """

#     def __init__(self, cfg: dict, device: str = "cuda"):
#         self.cfg = cfg
#         self.device = device
#         self.model_name = cfg["perception"]["lanes"]["model"]
#         self.max_lanes  = cfg["perception"]["lanes"]["max_lanes"]
#         self.model = self._load_model()

#     def _load_model(self):
#         # TODO: load CLRNet checkpoint
#         # CLRNet requires mmcv + its own config system.
#         # See: https://github.com/Turoad/CLRNet
#         # from clrnet.utils.config import Config
#         # from clrnet.models.registry import build_net
#         # ...
#         print(f"[LaneDetector] STUB: would load {self.model_name}")
#         return None

#     def detect(self, frame_bgr: np.ndarray) -> List[Lane]:
#         """
#         Run lane detection on one BGR frame.

#         Returns
#         -------
#         list[Lane]
#             Detected lanes with points in image coordinates.
#         """
#         if self.model is None:
#             return []

#         # TODO: implement
#         # 1. Preprocess frame (resize, normalize per CLRNet requirements)
#         # 2. Run model forward pass
#         # 3. Post-process to get polyline coordinates
#         # 4. Classify each lane's color and style (call helpers below)
#         raise NotImplementedError("LaneDetector.detect not yet implemented")

#     # ── Helpers ───────────────────────────────────────────────────────────

#     def _classify_color(self, frame_bgr: np.ndarray, points: list) -> str:
#         """
#         Sample pixels along the lane and classify as white or yellow
#         using HSV thresholding.

#         Returns "white" or "yellow".
#         """
#         # TODO: implement
#         # import cv2
#         # Sample ~10 points along the lane, extract small patches,
#         # convert to HSV, check if hue falls in yellow range (~15-35 deg).
#         # If majority of samples are yellow → "yellow", else "white".
#         raise NotImplementedError

#     def _classify_style(self, points: list, img_height: int) -> str:
#         """
#         Infer dashed vs solid from the vertical distribution of detected points.
#         Gaps between point clusters indicate dashes.

#         Returns "solid" or "dashed".
#         """
#         # TODO: implement
#         # Sort points by y. Compute gaps between consecutive y values.
#         # If max gap > threshold → "dashed", else "solid".
#         raise NotImplementedError
