"""
perception/pose.py
==================
2D human pose estimation using Ultralytics YOLO pose models.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_COCO17_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6],
]


class PersonPoseEstimator:
    """Run a YOLO pose model on person crops and attach 2D keypoints."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        pose_cfg = cfg.get("perception", {}).get("pose", {})
        self.enabled = bool(pose_cfg.get("enabled", True))
        self.device = device
        self.confidence = float(pose_cfg.get("confidence", 0.25))
        self.iou_threshold = float(pose_cfg.get("iou_threshold", 0.45))
        self.keypoint_threshold = float(pose_cfg.get("keypoint_threshold", 0.2))
        self.min_bbox_w = int(pose_cfg.get("min_bbox_w", 20))
        self.min_bbox_h = int(pose_cfg.get("min_bbox_h", 40))
        self.crop_padding_ratio = float(pose_cfg.get("crop_padding_ratio", 0.08))
        self.pose_format = str(pose_cfg.get("format", "coco17"))

        self.model_ref = self._resolve_model_ref(cfg, pose_cfg)
        self.model = self._load_model(self.model_ref) if self.enabled else None

    def _resolve_model_ref(self, cfg: dict, pose_cfg: dict) -> str:
        weights_cfg = cfg.get("weights", {})
        model_ref = pose_cfg.get("model") or weights_cfg.get("pose2d") or "yolo11n-pose.pt"
        model_path = Path(str(model_ref))
        if model_path.exists():
            return str(model_path)
        return str(model_ref)

    def _load_model(self, model_ref: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not available in this environment. Install the ultralytics "
                "package or disable perception.pose.enabled in config.yaml."
            ) from exc

        try:
            model = YOLO(model_ref)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pose model '{model_ref}'. If this is an official model name, "
                "Ultralytics may need internet access to download it. Otherwise point "
                "weights.pose2d or perception.pose.model to a local .pt pose checkpoint."
            ) from exc

        model.to(self.device)
        return model

    def annotate_detections(self, frame_bgr: np.ndarray, detections: list) -> list:
        if not self.enabled or self.model is None:
            return detections

        for det in detections:
            if getattr(det, "label", None) != "person":
                continue

            crop, offset = self._crop_person(frame_bgr, det.bbox)
            if crop is None:
                det.keypoints_2d = []
                det.keypoint_scores = []
                det.skeleton_links = []
                det.pose_format = self.pose_format
                continue

            keypoints, scores = self._predict_crop_pose(crop)
            if not keypoints:
                det.keypoints_2d = []
                det.keypoint_scores = []
                det.skeleton_links = []
                det.pose_format = self.pose_format
                continue

            ox, oy = offset
            det.keypoints_2d = [
                [round(float(x + ox), 2), round(float(y + oy), 2)]
                for x, y in keypoints
            ]
            det.keypoint_scores = [round(float(score), 4) for score in scores]
            det.skeleton_links = list(_COCO17_SKELETON)
            det.pose_format = self.pose_format

        return detections

    def _crop_person(
        self,
        frame_bgr: np.ndarray,
        bbox: List[float],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = x2 - x1
        bh = y2 - y1
        if bw < self.min_bbox_w or bh < self.min_bbox_h:
            return None, (max(0, int(round(x1))), max(0, int(round(y1))))

        pad_x = self.crop_padding_ratio * bw
        pad_y = self.crop_padding_ratio * bh
        x1i = max(0, min(int(round(x1 - pad_x)), w_img - 1))
        x2i = max(x1i + 1, min(int(round(x2 + pad_x)), w_img))
        y1i = max(0, min(int(round(y1 - pad_y)), h_img - 1))
        y2i = max(y1i + 1, min(int(round(y2 + pad_y)), h_img))

        crop = frame_bgr[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            return None, (x1i, y1i)
        return crop, (x1i, y1i)

    def _predict_crop_pose(self, crop_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        try:
            results = self.model.predict(
                crop_bgr,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception:
            return [], []

        if not results:
            return [], []

        result = results[0]
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or len(keypoints_obj) == 0:
            return [], []

        best_idx = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
            confs = boxes.conf.detach().cpu().numpy().reshape(-1)
            best_idx = int(np.argmax(confs))

        xy = keypoints_obj.xy
        xy = xy.detach().cpu().numpy() if hasattr(xy, "detach") else np.asarray(xy)
        if xy.ndim != 3 or best_idx >= xy.shape[0]:
            return [], []
        points = xy[best_idx]

        conf = getattr(keypoints_obj, "conf", None)
        if conf is None:
            scores = np.ones((points.shape[0],), dtype=np.float32)
        else:
            conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
            conf = np.asarray(conf)
            if conf.ndim == 2 and best_idx < conf.shape[0]:
                scores = conf[best_idx]
            else:
                scores = np.ones((points.shape[0],), dtype=np.float32)

        keypoints = [[float(pt[0]), float(pt[1])] for pt in points]
        return keypoints, [float(s) for s in scores]
