# """
# perception/depth.py
# ===================
# Monocular depth estimation using Depth Anything V2.

# Key responsibilities:
#   1. Produce a dense depth map (relative scale) for each frame.
#   2. Convert relative depth to metric scale using known object sizes.
#   3. Lift 2D bounding box centers to approximate 3D positions using
#      the camera intrinsics from config.yaml.

# Scale recovery strategy (from project hints):
#   - Find objects in the scene with known real-world heights
#     (cars ~1.5 m, people ~1.75 m, stop signs ~0.75 m).
#   - Compute scale factor: scale = known_height / (bbox_height_px * depth_relative)
#   - Average across multiple known objects for robustness.
# """

# from typing import List
# import numpy as np


# class DepthEstimator:
#     """
#     Wraps Depth Anything V2 for per-frame dense depth estimation.

#     Usage
#     -----
#     estimator = DepthEstimator(cfg, device="cuda")
#     depth_map = estimator.estimate(frame_bgr)         # H x W float32 array
#     objects   = estimator.lift_to_3d(objects, depth_map, cfg)
#     """

#     def __init__(self, cfg: dict, device: str = "cuda"):
#         self.cfg = cfg
#         self.device = device
#         self.encoder = cfg["perception"]["depth"]["encoder"]
#         self.known_heights = cfg["perception"]["depth"]["known_heights"]
#         # Camera intrinsics
#         cam = cfg["blender"]["camera"]
#         self.fx = cam["fx"]
#         self.fy = cam["fy"]
#         self.cx = cam["cx"]
#         self.cy = cam["cy"]
#         self.model = self._load_model(cfg["weights"]["depth"])

#     def _load_model(self, weights_path: str):
#         # TODO: uncomment once Depth Anything V2 repo is cloned and installed
#         # Add the depth_anything_v2 repo to sys.path first:
#         # sys.path.insert(0, "../depth-anything-v2")
#         # from depth_anything_v2.dpt import DepthAnythingV2
#         # model_configs = {
#         #     'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
#         #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#         #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256,512,1024,1024]},
#         # }
#         # model = DepthAnythingV2(**model_configs[self.encoder])
#         # model.load_state_dict(torch.load(weights_path, map_location='cpu'))
#         # model = model.to(self.device).eval()
#         # return model
#         print(f"[DepthEstimator] STUB: would load Depth Anything V2 ({self.encoder})")
#         return None

#     def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
#         """
#         Run depth estimation on one BGR frame.

#         Returns
#         -------
#         np.ndarray, shape (H, W), dtype float32
#             Relative inverse depth map. Larger values = closer to camera.
#             Call lift_to_3d to convert to metric scale.
#         """
#         if self.model is None:
#             h, w = frame_bgr.shape[:2]
#             return np.ones((h, w), dtype=np.float32)

#         # TODO: implement
#         # depth = self.model.infer_image(frame_bgr)   # returns H x W float32
#         # return depth
#         raise NotImplementedError("DepthEstimator.estimate not yet implemented")

#     def lift_to_3d(self, detections: list, depth_map: np.ndarray, cfg: dict) -> list:
#         """
#         Enrich each Detection with a metric depth estimate and 3D position.

#         Strategy
#         --------
#         1. For each detection, read the median depth value inside the bbox.
#         2. Compute metric scale using known real-world heights where possible.
#         3. Compute 3D position using pinhole camera model:
#                X = (cx_px - fx) / fx * Z
#                Y = (cy_px - fy) / fy * Z
#                Z = depth_metric

#         Modifies detections in-place and returns them.
#         """
#         if not detections:
#             return detections

#         # TODO: implement
#         # scale = self._estimate_scale(detections, depth_map)
#         # for det in detections:
#         #     x1, y1, x2, y2 = det.bbox
#         #     cx_px = (x1 + x2) / 2
#         #     cy_px = (y1 + y2) / 2
#         #     roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
#         #     rel_depth = float(np.median(roi)) if roi.size > 0 else 1.0
#         #     Z = scale / (rel_depth + 1e-6)
#         #     X = (cx_px - self.cx) / self.fx * Z
#         #     Y = (cy_px - self.cy) / self.fy * Z
#         #     det.depth_m = Z
#         #     det.position_3d = [X, Y, Z]
#         return detections

#     def _estimate_scale(self, detections: list, depth_map: np.ndarray) -> float:
#         """
#         Compute a pixel-to-meter scale factor from detections of known-height objects.
#         Returns a float scale such that: metric_depth = scale / relative_depth.
#         """
#         # TODO: implement
#         # For each detection whose label is in self.known_heights:
#         #   bbox_height_px = y2 - y1
#         #   known_h = self.known_heights[det.label]
#         #   focal = self.fy
#         #   expected_depth = focal * known_h / bbox_height_px
#         #   roi = depth_map[y1:y2, x1:x2]
#         #   rel = np.median(roi)
#         #   scale_estimate = expected_depth * rel
#         # Average scale estimates. Fall back to 1.0 if none available.
#         return 1.0
