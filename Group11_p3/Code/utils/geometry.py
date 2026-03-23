# """
# utils/geometry.py
# =================
# Shared geometry math used by both perception (depth lifting) and
# Blender (unprojection to ground plane, coordinate transforms).

# All functions are pure numpy — no OpenCV or bpy dependency.
# """

# import numpy as np
# from typing import List, Tuple


# # ── Camera model ──────────────────────────────────────────────────────────────

# def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
#     """
#     Build a 3×3 camera intrinsic matrix K from focal lengths and principal point.

#     Returns
#     -------
#     K : np.ndarray, shape (3, 3)
#     """
#     return np.array([
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1],
#     ], dtype=np.float64)


# def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
#     """
#     Compute a unit ray direction in camera space for image pixel (u, v).

#     Parameters
#     ----------
#     u, v : pixel coordinates (x right, y down)
#     K    : 3×3 intrinsic matrix

#     Returns
#     -------
#     ray : np.ndarray shape (3,), unit vector in camera space
#     """
#     # K_inv = np.linalg.inv(K)
#     # ray = K_inv @ np.array([u, v, 1.0])
#     # return ray / np.linalg.norm(ray)
#     return


# def backproject_to_depth(
#     u: float, v: float,
#     depth_m: float,
#     K: np.ndarray,
# ) -> np.ndarray:
#     """
#     Backproject a single image pixel at a known metric depth to camera-space 3D.

#     Parameters
#     ----------
#     u, v    : pixel coordinates
#     depth_m : metric depth (Z in camera space, meters)
#     K       : 3×3 intrinsic matrix

#     Returns
#     -------
#     np.ndarray shape (3,) : [X, Y, Z] in camera coordinates
#     """
#     # K_inv = np.linalg.inv(K)
#     # ray = K_inv @ np.array([u, v, 1.0])
#     # return ray * depth_m
#     return


# def bbox_center_to_3d(
#     bbox: List[float],
#     depth_m: float,
#     K: np.ndarray,
# ) -> np.ndarray:
#     """
#     Lift the center of a 2D bounding box to a 3D camera-space point.

#     Parameters
#     ----------
#     bbox    : [x1, y1, x2, y2] in pixels
#     depth_m : estimated metric depth for this object
#     K       : 3×3 intrinsic matrix

#     Returns
#     -------
#     np.ndarray shape (3,) : [X, Y, Z] in camera space (meters)
#     """
#     # cx = (bbox[0] + bbox[2]) / 2.0
#     # cy = (bbox[1] + bbox[3]) / 2.0
#     # return backproject_to_depth(cx, cy, depth_m, K)
#     return


# # ── Ground plane unprojection ─────────────────────────────────────────────────

# def unproject_to_ground(
#     u: float, v: float,
#     K: np.ndarray,
#     cam_height_m: float,
# ) -> np.ndarray:
#     """
#     Unproject an image point onto the flat ground plane (Y = 0 in world space).

#     Assumes the camera is at height cam_height_m above the ground,
#     looking forward. Camera Y-axis points down in image space.

#     Parameters
#     ----------
#     u, v         : pixel coordinates
#     K            : 3×3 intrinsic matrix
#     cam_height_m : camera height above ground (meters)

#     Returns
#     -------
#     np.ndarray shape (3,) : [X, Y, Z] world coordinates, Y=0 (ground)
#                             X = lateral, Z = forward, Y = up
#     """
#     # K_inv = np.linalg.inv(K)
#     # ray_cam = K_inv @ np.array([u, v, 1.0])   # camera space: x right, y down, z fwd

#     # # Camera sits at (0, cam_height_m, 0) in world space (X right, Y up, Z fwd)
#     # # Ray in world: X = ray_cam[0], Y = -ray_cam[1], Z = ray_cam[2]
#     # ray_world = np.array([ray_cam[0], -ray_cam[1], ray_cam[2]])

#     # # Intersect with Y=0 plane: cam_pos + t * ray = (*, 0, *)
#     # #   cam_height_m + t * ray_world[1] = 0  =>  t = -cam_height_m / ray_world[1]
#     # if abs(ray_world[1]) < 1e-6:
#     #     # Ray is nearly horizontal — no valid ground intersection
#     #     return np.array([0.0, 0.0, 0.0])

#     # t = -cam_height_m / ray_world[1]
#     # if t < 0:
#     #     # Intersection is behind the camera
#     #     return np.array([0.0, 0.0, 0.0])

#     # world_pt = np.array([0.0, cam_height_m, 0.0]) + t * ray_world
#     # return world_pt  # [X, 0, Z] in world
#     return


# # ── Coordinate system transforms ──────────────────────────────────────────────

# def camera_to_blender(pos_cam: np.ndarray) -> Tuple[float, float, float]:
#     """
#     Convert a point from camera space to Blender world space.

#     Camera space: X right, Y down, Z forward
#     Blender space: X right, Y forward, Z up

#     Parameters
#     ----------
#     pos_cam : np.ndarray shape (3,) [X, Y, Z] in camera space

#     Returns
#     -------
#     (bx, by, bz) suitable for bpy object.location
#     """
#     # x, y, z = pos_cam
#     # return (float(x), float(z), float(-y))
#     return


# def blender_to_camera(pos_bl: Tuple[float, float, float]) -> np.ndarray:
#     """Inverse of camera_to_blender."""
#     # bx, by, bz = pos_bl
#     # return np.array([bx, -bz, by])
#     return


# # ── Bounding box helpers ───────────────────────────────────────────────────────

# def bbox_area(bbox: List[float]) -> float:
#     """Return pixel area of a bounding box [x1, y1, x2, y2]."""
#     # return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
#     return


# def bbox_iou(a: List[float], b: List[float]) -> float:
#     """
#     Compute Intersection over Union of two bounding boxes.

#     Parameters
#     ----------
#     a, b : [x1, y1, x2, y2]

#     Returns
#     -------
#     float in [0, 1]
#     """
#     # ix1 = max(a[0], b[0])
#     # iy1 = max(a[1], b[1])
#     # ix2 = min(a[2], b[2])
#     # iy2 = min(a[3], b[3])
#     # inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
#     # union = bbox_area(a) + bbox_area(b) - inter
#     # return inter / union if union > 0 else 0.0
#     return


# def estimate_metric_depth_from_bbox(
#     bbox: List[float],
#     known_height_m: float,
#     fy: float,
# ) -> float:
#     """
#     Estimate metric depth of an object from its bounding box height
#     and known real-world height, using the thin-lens formula:

#         Z = fy * known_height_m / bbox_height_px

#     Parameters
#     ----------
#     bbox            : [x1, y1, x2, y2] in pixels
#     known_height_m  : real-world height of the object (meters)
#     fy              : vertical focal length in pixels

#     Returns
#     -------
#     float : estimated depth in meters
#     """
#     # bbox_h = max(1.0, bbox[3] - bbox[1])
#     # return (fy * known_height_m) / bbox_h
#     return
