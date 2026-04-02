"""
perception/depth.py
===================
Monocular depth estimation using Depth Anything V2.

Key responsibilities:
  1. Produce a dense depth map (relative scale) for each frame.
  2. Convert relative depth to metric scale using known object sizes.
  3. Lift 2D bounding box centers to approximate 3D positions using
     the camera intrinsics from config.yaml.

Scale recovery strategy (from project hints):
  - Find objects in the scene with known real-world heights
    (cars ~1.5 m, people ~1.75 m, stop signs ~0.75 m).
  - Compute scale factor: scale = known_height / (bbox_height_px * depth_relative)
  - Average across multiple known objects for robustness.
"""

# from typing import List
from pathlib import Path
import numpy as np
from .Depth_Anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch
from utils.io_utils import download_file_if_missing

_GROUND_CONTACT_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck", "person", "sedan", "hatchback", "suv", "pickuptruck", "pickup_truck"}


class DepthEstimator:
    """
    Wraps Depth Anything V2 for per-frame dense depth estimation

    Usage
    -----
    estimator = DepthEstimator(cfg, device="cuda")
    depth_map = estimator.estimate(frame_bgr)         # H x W float32 array
    objects   = estimator.lift_to_3d(objects, depth_map, cfg)
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.known_heights = cfg["perception"]["depth"]["known_heights"]
        # Camera intrinsics
        cam = cfg["blender"]["camera"]
        self.fx = cam["fx"]
        self.fy = cam["fy"]
        self.cx = cam["cx"]
        self.cy = cam["cy"]
        self.model = self._load_model(cfg["weights"]["depth"])

    def _load_model(self, weights_path: str) -> DepthAnythingV2:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            depth_cfg = self.cfg.get("perception", {}).get("depth", {})
            download_url = depth_cfg.get("download_url")
            download_file_if_missing(weights_path, download_url)

        model = DepthAnythingV2(**{**self.cfg["perception"]["depth"]["model"]})
        model.load_state_dict(
            torch.load(str(weights_path), map_location=self.device, weights_only=False)
        )
        model = model.to(self.device).eval()
        return model

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on one BGR frame.

        Returns
        -------
        np.ndarray, shape (H, W), dtype float32
            Metric depth map in meters (direct output from Depth Anything V2
            metric model). Larger values = farther from camera.
        """
        depth = self.model.infer_image(frame_bgr)  # returns H x W float32
        return depth


    def lift_to_3d(self, detections: list, depth_map: np.ndarray) -> list:
        """
        Enrich each Detection with a metric depth estimate and 3D position.

        The depth model (Depth Anything V2 metric, VKITTI-trained) already outputs
        metric depth in meters. We read the median depth from the inner 50% of the
        bbox ROI as Z (shrinking the ROI avoids background bleed on object edges),
        then apply a scale correction computed from known-height objects to
        compensate for domain shift between VKITTI and our footage.

        Pinhole back-projection to camera-frame 3D coordinates:
            X = (u - cx) / fx * Z      # positive X = right
            Y = (v - cy) / fy * Z      # positive Y = down
            Z = depth_m                # positive Z = forward into scene

        Modifies detections in-place and returns them.
        """
        if not detections:
            return detections

        scale = self._estimate_scale(detections, depth_map)

        H_map, W_map = depth_map.shape

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Use a ground-contact pixel for road users so the lifted 3D point
            # better matches where the object touches the road, not its visual center.
            if det.label in _GROUND_CONTACT_LABELS:
                u = (x1 + x2) / 2.0
                v = y2
            else:
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0

            # Sample the inner 50% of the bbox to reduce background bleed
            # at object boundaries (common failure mode for monocular depth).
            bh = y2 - y1
            bw = x2 - x1
            if det.label in _GROUND_CONTACT_LABELS:
                # Focus the depth estimate near the lower middle of the bbox,
                # which better approximates wheel/foot contact with the road.
                y1i = max(0,       min(int(y1 + 0.70 * bh), H_map - 1))
                y2i = max(y1i + 1, min(int(y2), H_map))
                x1i = max(0,       min(int(u - 0.15 * bw), W_map - 1))
                x2i = max(x1i + 1, min(int(u + 0.15 * bw), W_map))
            else:
                y1i = max(0,         min(int(y1 + 0.25 * bh), H_map - 1))
                y2i = max(y1i + 1,   min(int(y2 - 0.25 * bh), H_map))
                x1i = max(0,         min(int(x1 + 0.25 * bw), W_map - 1))
                x2i = max(x1i + 1,   min(int(x2 - 0.25 * bw), W_map))

            roi = depth_map[y1i:y2i, x1i:x2i]
            Z_raw = float(np.median(roi)) if roi.size > 0 else float(np.median(depth_map))

            # Apply domain-shift correction (1.0 if no known-height objects found)
            Z = Z_raw * scale
            Z = max(Z, 0.1)  # sanity floor: nothing closer than 10 cm

            # Pinhole back-projection
            X = (u - self.cx) / self.fx * Z
            Y = (v - self.cy) / self.fy * Z

            det.depth_m = round(Z, 3)
            det.position_3d = [round(X, 3), round(Y, 3), round(Z, 3)]

        return detections

    def _estimate_scale(self, detections: list, depth_map: np.ndarray) -> float:
        """
        Compute a multiplicative scale correction for the metric depth map.

        Even though the model outputs metric depth, there is often a scale bias
        when the training domain (VKITTI) differs from the target footage.
        We recover a correction factor using objects of known real-world height:

            expected_Z  = fy * known_height_m / bbox_height_px   (pinhole model)
            measured_Z  = median depth inside bbox               (model output)
            correction  = expected_Z / measured_Z

        Corrections are clamped per-detection to [0.5, 2.0] before averaging
        to limit the impact of a single bad detection. Falls back to 1.0 if no
        known-height objects are present.
        """
        H_map, W_map = depth_map.shape
        corrections = []

        for det in detections:
            known_h = self.known_heights.get(det.label)
            if known_h is None:
                continue

            x1, y1, x2, y2 = det.bbox
            bbox_height_px = y2 - y1
            if bbox_height_px < 10:
                # Too small — pinhole estimate noisy and ROI unreliable
                continue

            # Expected metric depth from pinhole geometry
            expected_Z = self.fy * known_h / bbox_height_px

            # Measured depth: median over the full bbox ROI
            y1i = max(0,       min(int(y1), H_map - 1))
            y2i = max(y1i + 1, min(int(y2), H_map))
            x1i = max(0,       min(int(x1), W_map - 1))
            x2i = max(x1i + 1, min(int(x2), W_map))
            roi = depth_map[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                continue
            measured_Z = float(np.median(roi))
            if measured_Z < 0.01:
                continue  # degenerate model output, skip

            corrections.append(expected_Z / measured_Z)

        if not corrections:
            return 1.0

        # Clamp each correction before averaging to reduce outlier impact
        corrections = [max(0.5, min(c, 2.0)) for c in corrections]
        return float(np.mean(corrections))
