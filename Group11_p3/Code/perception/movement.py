
import numpy as np
import cv2

def estimate_ego_motion(flow, mask=None):
	"""
	Estimate global camera motion (homography) from dense optical flow, using only background (non-object) regions.
	Args:
		flow: (H, W, 2) numpy array, optical flow from prev to curr frame
		mask: (H, W) bool or uint8, True for background pixels (optional)
	Returns:
		H: 3x3 homography matrix (float32) or None if estimation fails
	"""
	H, W = flow.shape[:2]
	grid_y, grid_x = np.mgrid[0:H, 0:W]
	pts1 = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
	pts2 = pts1 + flow.reshape(-1, 2)
	if mask is not None:
		mask_flat = mask.reshape(-1)
		pts1 = pts1[mask_flat]
		pts2 = pts2[mask_flat]
	if len(pts1) < 16:
		return None
	Hmat, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
	return Hmat

def median_flow_in_box(flow, bbox):
	"""
	Compute median flow vector inside a bounding box.
	Args:
		flow: (H, W, 2) numpy array
		bbox: [x1, y1, x2, y2] (float)
	Returns:
		median_flow: (2,) float32
	"""
	x1, y1, x2, y2 = map(int, bbox)
	region = flow[max(0, y1):max(0, y2), max(0, x1):max(0, x2), :]
	if region.size == 0:
		return np.zeros(2, dtype=np.float32)
	return np.median(region.reshape(-1, 2), axis=0)

def warp_points(points, H):
	"""
	Warp 2D points using a homography.
	Args:
		points: (N, 2) array
		H: 3x3 homography
	Returns:
		warped: (N, 2) array
	"""
	pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
	pts_warp = (H @ pts_h.T).T
	pts_warp = pts_warp[:, :2] / np.clip(pts_warp[:, 2:3], 1e-6, None)
	return pts_warp

def is_vehicle_moving(
	det,
	flow,
	bg_mask,
	ego_H,
	dt,
	min_residual_px=1.5,
	min_3d_speed=0.7,
	edge_margin_px=20
):
	"""
	Decide if a vehicle is moving using ego-compensated flow and 3D velocity.
	Args:
		det: detection object with bbox, position_3d, prev_position_3d
		flow: (H, W, 2) optical flow
		bg_mask: (H, W) bool, True for background
		ego_H: 3x3 homography (ego motion)
		dt: time between frames (s)
		min_residual_px: threshold for residual flow
		min_3d_speed: threshold for 3D speed (m/s)
		edge_margin_px: ignore flow if box is near image edge
	Returns:
		is_moving: bool
		score: float (max of residual flow and 3D speed, normalized)
	"""
	bbox = det.bbox
	H, W = flow.shape[:2]
	x1, y1, x2, y2 = map(int, bbox)
	# Edge check
	if x1 < edge_margin_px or x2 > (W - edge_margin_px) or y1 < edge_margin_px or y2 > (H - edge_margin_px):
		use_flow = False
	else:
		use_flow = True
	# Residual flow
	median_f = median_flow_in_box(flow, bbox)
	if ego_H is not None:
		# Predict where box center should move if static
		cx = 0.5 * (x1 + x2)
		cy = 0.5 * (y1 + y2)
		pred = warp_points(np.array([[cx, cy]], dtype=np.float32), ego_H)[0] - np.array([cx, cy])
		residual = median_f - pred
	else:
		residual = median_f
	residual_mag = float(np.linalg.norm(residual)) if use_flow else 0.0
	# 3D velocity
	v3d = 0.0
	if hasattr(det, "position_3d") and hasattr(det, "prev_position_3d") and det.prev_position_3d is not None:
		v3d = np.linalg.norm(np.array(det.position_3d) - np.array(det.prev_position_3d)) / max(dt, 1e-6)
	# Decision
	is_moving = (residual_mag > min_residual_px) or (v3d > min_3d_speed)
	score = max(residual_mag / min_residual_px, v3d / min_3d_speed)
	return is_moving, float(score)