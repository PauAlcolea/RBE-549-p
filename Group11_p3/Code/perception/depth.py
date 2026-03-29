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

_GROUND_CONTACT_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck", "person"}


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

    def estimate_headings(self, detections: list, depth_map: np.ndarray) -> list:
        """
        Estimate the heading (yaw) of each vehicle detection from the depth map
        using PCA on a 3D point cloud cropped from the object's bounding box.
 
        Algorithm per detection
        -----------------------
        1. Crop the depth patch for the inner 80% of the bbox (reduce background bleed).
        2. Backproject every pixel (u, v, Z) → camera-space 3D point (X, Y, Z).
        3. Project points onto the ground plane (keep only X and Z — the horizontal axes).
        4. Run PCA on the XZ point cloud.
           - The first principal component = the car's dominant axis (long axis for cars).
        5. The raw PCA axis has a 180° ambiguity. Resolve it:
           - The car's front is more likely facing AWAY from the ego camera when it is
             in front of us (Z > 0, which is always true in camera space).
           - We use the horizontal position of the bbox center relative to the image
             center: a car on the LEFT side of the image that is elongated horizontally
             is most likely driving rightward (positive X), and vice versa.
           - Concretely: choose the PCA direction whose X component agrees with the
             sign of (bbox_center_x - image_cx). Ties / ambiguous cases default to
             heading = 0 (facing forward, away from ego).
        6. Convert the resolved (dX, dZ) direction vector to a Blender Z-up yaw angle:
               heading_rad = atan2(dX, dZ)
           This matches Blender's convention where 0 = +Y (forward) and +π/2 = +X (right).
 
        Only modifies detections whose label is a vehicle type. Pedestrians keep
        heading_rad = 0.0 (the project spec says pedestrians face the same way in Ph1).
 
        Modifies detections in-place and returns them.
        """
        _VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck"}
        H_map, W_map = depth_map.shape
 
        for det in detections:
            if det.label not in _VEHICLE_LABELS:
                continue
 
            x1, y1, x2, y2 = det.bbox
            bw = x2 - x1
            bh = y2 - y1
 
            # Use inner 80% of bbox to reduce background contamination
            x1i = max(0,       min(int(x1 + 0.10 * bw), W_map - 1))
            x2i = max(x1i + 1, min(int(x2 - 0.10 * bw), W_map))
            y1i = max(0,       min(int(y1 + 0.10 * bh), H_map - 1))
            y2i = max(y1i + 1, min(int(y2 - 0.10 * bh), H_map))
 
            depth_patch = depth_map[y1i:y2i, x1i:x2i]
            if depth_patch.size < 20:
                # Bbox too small for reliable PCA — leave heading at default 0
                continue
 
            # Build pixel coordinate grids for the patch
            us = np.arange(x1i, x2i, dtype=np.float32)   # shape (W_patch,)
            vs = np.arange(y1i, y2i, dtype=np.float32)   # shape (H_patch,)
            uu, vv = np.meshgrid(us, vs)                  # both shape (H_patch, W_patch)
 
            Z = depth_patch.astype(np.float32)
 
            # Filter out degenerate depth values (sky, missing data)
            valid = (Z > 0.1) & (Z < 200.0)
            Z = Z[valid]
            uu = uu[valid]
            vv = vv[valid]
 
            if Z.size < 20:
                continue
 
            # Pinhole backprojection → camera-space 3D
            X_cam = (uu - self.cx) / self.fx * Z   # lateral  (right = positive)
            # Y_cam = (vv - self.cy) / self.fy * Z  # vertical (down  = positive) — not needed for PCA on ground plane
            Z_cam = Z                               # forward  (away  = positive)
 
            # PCA on the XZ ground plane
            pts = np.stack([X_cam, Z_cam], axis=1)  # shape (N, 2)
            pts -= pts.mean(axis=0)                 # center
 
            if pts.shape[0] < 2:
                continue
 
            cov = np.cov(pts.T)                     # 2×2 covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # eigh returns eigenvalues in ascending order; take the LAST (largest) = PC1
            principal_axis = eigenvectors[:, -1]    # shape (2,): [dX, dZ]
 
            dX, dZ = float(principal_axis[0]), float(principal_axis[1])
 
            # ── 180° ambiguity resolution ──────────────────────────────────
            # Heuristic: the car's front faces away from the ego vehicle.
            # For a car directly in front, dZ should be positive (pointing away).
            # For a car to the side, use horizontal bbox position to decide left vs right.
            bbox_cx = (x1 + x2) / 2.0
            image_cx = self.cx
 
            # If the principal axis points toward the camera (dZ < 0), flip it
            if dZ < 0:
                dX, dZ = -dX, -dZ
 
            # Secondary disambiguation using horizontal position:
            # A car to the LEFT of image center (bbox_cx < image_cx) is more likely
            # moving/facing right (positive dX), and vice versa.
            # Only apply this flip if the axis is mostly lateral (|dX| > |dZ|),
            # i.e. the car is side-on to the camera.
            if abs(dX) > abs(dZ):
                expected_sign = 1.0 if bbox_cx < image_cx else -1.0
                if dX * expected_sign < 0:
                    dX, dZ = -dX, -dZ
 
            # Convert (dX, dZ) direction → Blender yaw angle
            # Blender Z-up, Y-forward: heading = atan2(X, Y_forward)
            # In our camera/Blender mapping: forward = +Z_cam = +Y_blender
            heading = float(np.arctan2(dX, dZ))
 
            det.heading_rad = round(heading, 4)
 
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
