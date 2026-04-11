"""
perception/movement.py
=====================
Vehicle parked vs moving classification using optical flow + Sampson distance.
"""

from __future__ import annotations

from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_clip(bbox: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1i = int(max(0, min(width - 1, round(x1))))
    y1i = int(max(0, min(height - 1, round(y1))))
    x2i = int(max(0, min(width, round(x2))))
    y2i = int(max(0, min(height, round(y2))))
    if x2i - x1i < 2 or y2i - y1i < 2:
        return None
    return x1i, y1i, x2i, y2i


def _sampson_distance(pts1: np.ndarray, pts2: np.ndarray, f_mat: np.ndarray) -> np.ndarray:
    """Compute Sampson distance for Nx2 correspondences."""
    if pts1.size == 0:
        return np.empty((0,), dtype=np.float32)

    ones = np.ones((pts1.shape[0], 1), dtype=np.float32)
    x1 = np.concatenate([pts1.astype(np.float32), ones], axis=1)
    x2 = np.concatenate([pts2.astype(np.float32), ones], axis=1)

    fx1 = (f_mat @ x1.T).T
    ftx2 = (f_mat.T @ x2.T).T
    numer = np.sum(x2 * fx1, axis=1) ** 2
    denom = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    denom = np.maximum(denom, 1e-6)
    return (numer / denom).astype(np.float32)


class MovementEstimator:
    """Annotate vehicle detections with parked/moving state."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        self._model = None
        self._flow_args = None
        self._track_state: Dict[int, dict] = {}
        self._last_flow_viz: Optional[np.ndarray] = None
        self._last_flow_source: str = "none"

        temporal_cfg = cfg.get("perception", {}).get("temporal_stability", {})
        motion_cfg = temporal_cfg.get("motion_detection", {})
        self.enabled = bool(motion_cfg.get("enabled", False))
        self.use_flow_anything = bool(motion_cfg.get("use_flow_anything", True))
        self.fallback_on_failed_flow = bool(motion_cfg.get("fallback_on_failed_flow", True))

        self.history_frames = max(3, int(motion_cfg.get("history_frames", 8)))
        self.switch_hysteresis_frames = max(1, int(motion_cfg.get("switch_hysteresis_frames", 2)))
        self.min_track_age_frames = max(1, int(motion_cfg.get("min_track_age_frames", 3)))

        self.parked_threshold = float(motion_cfg.get("parked_threshold", 0.35))
        self.moving_threshold = float(motion_cfg.get("moving_threshold", 0.60))
        if self.moving_threshold <= self.parked_threshold:
            self.moving_threshold = self.parked_threshold + 0.1

        self.flow_mag_moving_px = max(0.1, float(motion_cfg.get("flow_mag_moving_px", 1.5)))
        self.sampson_moving = max(1e-4, float(motion_cfg.get("sampson_moving_threshold", 1.5)))
        self.ring_scale = max(1.05, float(motion_cfg.get("local_ring_scale", 1.45)))
        self.min_ring_points = max(8, int(motion_cfg.get("min_ring_points", 30)))
        self.local_baseline_blend = _clamp01(float(motion_cfg.get("local_baseline_blend", 0.65)))
        self.comoving_delta_px = max(0.01, float(motion_cfg.get("comoving_delta_px", 0.25)))
        self.allow_unknown_state = bool(motion_cfg.get("allow_unknown_state", True))
        self.weak_geometry_inlier_ratio = _clamp01(float(motion_cfg.get("weak_geometry_inlier_ratio", 0.45)))
        self.force_unknown_on_weak_geometry = bool(motion_cfg.get("force_unknown_on_weak_geometry", True))
        self.track_velocity_moving_px = max(0.05, float(motion_cfg.get("track_velocity_moving_px", 0.9)))
        self.tracker_reliability_min = _clamp01(float(motion_cfg.get("tracker_reliability_min", 0.35)))
        self.track_gap_penalty_frames = max(1, int(motion_cfg.get("track_gap_penalty_frames", 2)))
        self.track_prior_weight = _clamp01(float(motion_cfg.get("track_prior_weight", 0.35)))
        self.edge_margin_ratio = _clamp01(float(motion_cfg.get("edge_margin_ratio", 0.18)))
        self.edge_flow_attenuation = _clamp01(float(motion_cfg.get("edge_flow_attenuation", 0.45)))
        self.edge_unknown_if_low_ring = bool(motion_cfg.get("edge_unknown_if_low_ring", True))
        self.edge_min_ring_points = max(4, int(motion_cfg.get("edge_min_ring_points", 20)))
        self.use_depth_attenuation = bool(motion_cfg.get("use_depth_attenuation", True))
        self.depth_ref_m = max(1.0, float(motion_cfg.get("depth_ref_m", 20.0)))
        self.depth_attenuation_strength = _clamp01(float(motion_cfg.get("depth_attenuation_strength", 0.35)))

        self.bg_sample_step = max(2, int(motion_cfg.get("bg_sample_step", 10)))
        self.box_sample_step = max(1, int(motion_cfg.get("box_sample_step", 4)))
        self.min_bg_points = max(16, int(motion_cfg.get("min_bg_points", 200)))
        self.min_points_per_box = max(6, int(motion_cfg.get("min_points_per_box", 40)))
        self.ransac_thresh = max(0.1, float(motion_cfg.get("sampson_ransac_thresh", 1.0)))

        self._flow_init_error = None
        if self.enabled and self.use_flow_anything:
            self._init_flow_anything(motion_cfg)
        self._print_startup_status()

    def reset(self) -> None:
        self._track_state.clear()
        self._last_flow_viz = None
        self._last_flow_source = "none"

    def get_last_flow_visualization(self) -> Tuple[Optional[np.ndarray], str]:
        if self._last_flow_viz is None:
            return None, self._last_flow_source
        return self._last_flow_viz.copy(), self._last_flow_source

    def _flow_to_bgr(self, flow: np.ndarray) -> np.ndarray:
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8((ang / 2.0) % 180.0)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _print_startup_status(self) -> None:
        if not self.enabled:
            print("[movement] disabled (perception.temporal_stability.motion_detection.enabled=false)")
            return

        if self._model is not None:
            flow_cfg = self.cfg.get("perception", {}).get("temporal_stability", {}).get("motion_detection", {}).get("flow_cfg", "")
            flow_ckpt = self.cfg.get("weights", {}).get("flow", "")
            print(
                f"[movement] enabled | backend=flow_anything | cfg={flow_cfg} | ckpt={flow_ckpt} "
                f"| tracker_fusion(reliability_min={self.tracker_reliability_min:.2f}, track_weight={self.track_prior_weight:.2f})"
            )
            return

        reason = self._flow_init_error or "FlowAnything unavailable"
        print(f"[movement] enabled | backend=unavailable | reason={reason}")

    def _init_flow_anything(self, motion_cfg: dict) -> None:
        try:
            root_dir = Path(__file__).resolve().parent
            flow_root = root_dir / "Flow_Anything"
            core_dir = flow_root / "core"
            if str(core_dir) not in sys.path:
                sys.path.insert(0, str(core_dir))

            from .Flow_Anything.config.parser import json_to_args
            from .Flow_Anything.core.raft import RAFT
            from .Flow_Anything.core.utils.utils import load_ckpt

            cfg_rel = str(motion_cfg.get("flow_cfg", "perception/Flow_Anything/config/eval/spring-M.json"))
            cfg_path = Path(__file__).resolve().parents[1] / cfg_rel
            if not cfg_path.exists():
                cfg_path = flow_root / "config" / "eval" / "spring-M.json"

            ckpt_rel = self.cfg.get("weights", {}).get("flow", "")
            ckpt_path = Path(__file__).resolve().parents[1] / str(ckpt_rel) if ckpt_rel else None
            if ckpt_path is None or not ckpt_path.exists():
                self._flow_init_error = "FlowAnything checkpoint not found"
                self._model = None
                self._flow_args = None
                return

            args = json_to_args(str(cfg_path))
            if not hasattr(args, "scale"):
                args.scale = 0
            if not hasattr(args, "iters"):
                args.iters = 12

            self._flow_args = args
            self._model = RAFT(args)
            load_ckpt(self._model, str(ckpt_path))
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            self._flow_init_error = f"FlowAnything init failed: {exc}"
            self._model = None
            self._flow_args = None

    def _flow_from_flow_anything(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None or self._flow_args is None:
            return None

        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        scale = float(getattr(self._flow_args, "scale", 0))
        if scale != 0:
            t1 = F.interpolate(t1, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
            t2 = F.interpolate(t2, scale_factor=2 ** scale, mode="bilinear", align_corners=False)

        with torch.no_grad():
            output = self._model(t1, t2, iters=self._flow_args.iters, test_mode=True)
            flow_t = output["flow"][-1]

        if scale != 0:
            down = 0.5 ** scale
            flow_t = F.interpolate(flow_t, scale_factor=down, mode="bilinear", align_corners=False) * down

        flow = flow_t[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return flow

    def _compute_flow(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        flow = None
        source = "none"

        try:
            flow = self._flow_from_flow_anything(prev_bgr, curr_bgr)
            if flow is not None:
                source = "flow_anything"
        except Exception:
            flow = None

        return flow, source

    def _build_background_points(
        self,
        flow: np.ndarray,
        boxes: List[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = flow.shape[:2]
        bg_mask = np.ones((h, w), dtype=np.uint8)
        for box in boxes:
            clipped = _bbox_clip(box, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            bg_mask[y1:y2, x1:x2] = 0

        ys = np.arange(0, h, self.bg_sample_step)
        xs = np.arange(0, w, self.bg_sample_step)
        grid_x, grid_y = np.meshgrid(xs, ys)

        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)
        keep = bg_mask[flat_y, flat_x] > 0
        if int(np.sum(keep)) < self.min_bg_points:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0.0

        p1 = np.stack([flat_x[keep], flat_y[keep]], axis=1).astype(np.float32)
        uv = flow[flat_y[keep], flat_x[keep]]
        p2 = p1 + uv.astype(np.float32)

        valid = (
            (p2[:, 0] >= 0.0)
            & (p2[:, 0] <= (w - 1.0))
            & (p2[:, 1] >= 0.0)
            & (p2[:, 1] <= (h - 1.0))
        )
        p1 = p1[valid]
        p2 = p2[valid]

        mags = np.linalg.norm(uv[valid], axis=1)
        bg_flow_median = float(np.median(mags)) if mags.size else 0.0
        return p1, p2, bg_flow_median

    def _estimate_fundamental(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        if p1.shape[0] < self.min_bg_points:
            return None, 0.0
        f_mat, mask = cv2.findFundamentalMat(
            p1,
            p2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            confidence=0.99,
            maxIters=3000,
        )
        if f_mat is None:
            return None, 0.0
        if not isinstance(f_mat, np.ndarray) or f_mat.shape != (3, 3):
            return None, 0.0
        inlier_ratio = 0.0
        if isinstance(mask, np.ndarray) and mask.size > 0:
            inlier_ratio = float(np.mean(mask.astype(np.float32)))
        return f_mat.astype(np.float32), inlier_ratio

    def _ring_flow_median(self, flow: np.ndarray, bbox: Sequence[float]) -> Tuple[float, int]:
        h, w = flow.shape[:2]
        clipped = _bbox_clip(bbox, w, h)
        if clipped is None:
            return 0.0, 0

        x1, y1, x2, y2 = clipped
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(2.0, float(x2 - x1))
        bh = max(2.0, float(y2 - y1))

        half_w = 0.5 * bw * self.ring_scale
        half_h = 0.5 * bh * self.ring_scale
        rx1 = int(max(0, min(w - 1, round(cx - half_w))))
        ry1 = int(max(0, min(h - 1, round(cy - half_h))))
        rx2 = int(max(0, min(w, round(cx + half_w))))
        ry2 = int(max(0, min(h, round(cy + half_h))))

        if rx2 - rx1 < 2 or ry2 - ry1 < 2:
            return 0.0, 0

        xs = np.arange(rx1, rx2, self.box_sample_step, dtype=np.int32)
        ys = np.arange(ry1, ry2, self.box_sample_step, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return 0.0, 0

        gx, gy = np.meshgrid(xs, ys)
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)

        in_box = (gx >= x1) & (gx < x2) & (gy >= y1) & (gy < y2)
        ring_keep = ~in_box
        if not np.any(ring_keep):
            return 0.0, 0

        uv = flow[gy[ring_keep], gx[ring_keep]]
        mags = np.linalg.norm(uv, axis=1)
        if mags.size == 0:
            return 0.0, 0
        return float(np.median(mags)), int(mags.size)

    def _box_metrics(self, flow: np.ndarray, bbox: Sequence[float], f_mat: Optional[np.ndarray]) -> Tuple[float, float, int]:
        h, w = flow.shape[:2]
        clipped = _bbox_clip(bbox, w, h)
        if clipped is None:
            return 0.0, 0.0, 0

        x1, y1, x2, y2 = clipped
        xs = np.arange(x1, x2, self.box_sample_step, dtype=np.int32)
        ys = np.arange(y1, y2, self.box_sample_step, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return 0.0, 0.0, 0

        grid_x, grid_y = np.meshgrid(xs, ys)
        gx = grid_x.reshape(-1)
        gy = grid_y.reshape(-1)
        uv = flow[gy, gx]
        mags = np.linalg.norm(uv, axis=1)
        flow_med = float(np.median(mags)) if mags.size else 0.0

        if f_mat is None:
            return flow_med, 0.0, int(mags.size)

        p1 = np.stack([gx, gy], axis=1).astype(np.float32)
        p2 = p1 + uv.astype(np.float32)
        valid = (
            (p2[:, 0] >= 0.0)
            & (p2[:, 0] <= (w - 1.0))
            & (p2[:, 1] >= 0.0)
            & (p2[:, 1] <= (h - 1.0))
        )
        if not np.any(valid):
            return flow_med, 0.0, int(mags.size)

        sd = _sampson_distance(p1[valid], p2[valid], f_mat)
        sampson_med = float(np.median(sd)) if sd.size else 0.0
        return flow_med, sampson_med, int(mags.size)

    def _motion_evidence(self, flow_med: float, sampson_med: float, baseline_flow_med: float, has_geometry: bool) -> float:
        flow_excess = max(0.0, flow_med - baseline_flow_med)
        flow_score = _clamp01(flow_excess / self.flow_mag_moving_px)

        if has_geometry:
            sampson_score = _clamp01(sampson_med / self.sampson_moving)
            return 0.6 * flow_score + 0.4 * sampson_score
        return 0.8 * flow_score

    def _depth_attenuation(self, depth_m: Optional[float]) -> float:
        if not self.use_depth_attenuation:
            return 1.0
        if depth_m is None:
            return 1.0
        d = float(depth_m)
        if not np.isfinite(d) or d <= 0.0:
            return 1.0

        # For far objects, downweight motion evidence because pixel flow noise and
        # small geometric errors can dominate true residual motion.
        far_ratio = max(0.0, (d - self.depth_ref_m) / max(1e-6, self.depth_ref_m))
        return 1.0 / (1.0 + self.depth_attenuation_strength * far_ratio)

    def _track_prior(self, feature: Optional[dict]) -> Tuple[Optional[float], float, bool]:
        if not feature:
            return None, 0.0, False

        reliability = _clamp01(float(feature.get("reliability", 0.0)))
        speed_px = max(0.0, float(feature.get("speed_px", 0.0)))
        gap_frames = int(feature.get("gap_frames", 0))

        if reliability <= 1e-3:
            return None, reliability, True

        prior = _clamp01(speed_px / self.track_velocity_moving_px)
        if gap_frames >= self.track_gap_penalty_frames:
            prior *= 0.6

        prefer_unknown = reliability < self.tracker_reliability_min
        return prior, reliability, prefer_unknown

    def _edge_strength(self, bbox: Sequence[float], width: int, height: int) -> Tuple[float, str]:
        if width <= 1 or height <= 1:
            return 0.0, "none"
        clipped = _bbox_clip(bbox, width, height)
        if clipped is None:
            return 0.0, "none"

        x1, y1, x2, y2 = clipped
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        edge_dist_px = min(cx, (width - 1) - cx, cy, (height - 1) - cy)
        norm_dist = edge_dist_px / float(max(1, min(width, height)))
        if norm_dist >= self.edge_margin_ratio:
            strength = 0.0
        else:
            strength = _clamp01(1.0 - (norm_dist / max(1e-6, self.edge_margin_ratio)))

        dists = {
            "left": cx,
            "right": (width - 1) - cx,
            "top": cy,
            "bottom": (height - 1) - cy,
        }
        side = min(dists, key=dists.get) if strength > 0.0 else "none"
        return strength, side

    def _update_track_state(self, track_id: int, evidence: float, frame_idx: int, prefer_unknown: bool) -> Tuple[str, float, int]:
        state = self._track_state.setdefault(
            int(track_id),
            {
                "history": deque(maxlen=self.history_frames),
                "stable": "unknown",
                "pending": None,
                "pending_count": 0,
                "age": 0,
                "last_seen": -1,
            },
        )
        state["history"].append(float(evidence))
        state["age"] = int(state.get("age", 0)) + 1
        state["last_seen"] = int(frame_idx)

        history = list(state["history"])
        avg = float(np.mean(history)) if history else 0.0
        stable = str(state.get("stable", "unknown"))

        if prefer_unknown and self.allow_unknown_state:
            target = "unknown"
        elif avg >= self.moving_threshold:
            target = "moving"
        elif avg <= self.parked_threshold:
            target = "parked"
        else:
            target = "unknown" if self.allow_unknown_state else None

        if state["age"] < self.min_track_age_frames:
            stable = "unknown"
            state["pending"] = None
            state["pending_count"] = 0
        elif target is None:
            state["pending"] = None
            state["pending_count"] = 0
        elif stable in {"unknown", target}:
            stable = target
            state["pending"] = None
            state["pending_count"] = 0
        else:
            if state.get("pending") == target:
                state["pending_count"] = int(state.get("pending_count", 0)) + 1
            else:
                state["pending"] = target
                state["pending_count"] = 1
            if int(state.get("pending_count", 0)) >= self.switch_hysteresis_frames:
                stable = target
                state["pending"] = None
                state["pending_count"] = 0

        state["stable"] = stable

        mid = 0.5 * (self.parked_threshold + self.moving_threshold)
        half = max(1e-3, 0.5 * (self.moving_threshold - self.parked_threshold))
        if stable == "unknown":
            confidence = _clamp01(1.0 - (abs(avg - mid) / half))
        else:
            confidence = _clamp01(abs(avg - mid) / half)
        return stable, confidence, int(state["age"])

    def annotate(
        self,
        frame_idx: int,
        prev_frame_bgr: Optional[np.ndarray],
        curr_frame_bgr: np.ndarray,
        objects: list,
        vehicle_labels: Sequence[str],
        track_features: Optional[Dict[int, dict]] = None,
    ) -> None:
        if not self.enabled:
            return

        vehicle_set = {str(v) for v in vehicle_labels}
        vehicle_dets = [d for d in objects if getattr(d, "label", "") in vehicle_set]
        if not vehicle_dets:
            return

        if prev_frame_bgr is None:
            self._last_flow_viz = None
            self._last_flow_source = "warmup"
            for det in vehicle_dets:
                det.is_moving = False
                det.is_parked = False
                det.motion_confidence = 0.0
                det.motion_source = "warmup"
                det.motion_age_frames = 0
            return

        flow, source = self._compute_flow(prev_frame_bgr, curr_frame_bgr)
        self._last_flow_source = source
        if flow is None:
            self._last_flow_viz = None
            for det in vehicle_dets:
                det.is_moving = False
                det.is_parked = False
                det.motion_confidence = 0.0
                det.motion_source = "unavailable"
                det.motion_age_frames = 0
            return

        self._last_flow_viz = self._flow_to_bgr(flow)

        all_boxes = [getattr(det, "bbox", None) for det in objects if getattr(det, "bbox", None) is not None]
        p1_bg, p2_bg, bg_flow_med = self._build_background_points(flow, all_boxes)
        f_mat, inlier_ratio = self._estimate_fundamental(p1_bg, p2_bg)
        has_geometry = f_mat is not None
        weak_geometry = bool(has_geometry and inlier_ratio < self.weak_geometry_inlier_ratio)
        frame_h, frame_w = flow.shape[:2]

        for det in vehicle_dets:
            track_id = getattr(det, "track_id", None)
            flow_med, sampson_med, point_count = self._box_metrics(flow, det.bbox, f_mat)
            ring_flow_med, ring_points = self._ring_flow_median(flow, det.bbox)
            enough_points = point_count >= self.min_points_per_box
            has_local_ring = ring_points >= self.min_ring_points

            baseline_flow_med = bg_flow_med
            if has_local_ring:
                baseline_flow_med = (
                    self.local_baseline_blend * ring_flow_med
                    + (1.0 - self.local_baseline_blend) * bg_flow_med
                )

            local_delta = abs(flow_med - baseline_flow_med)
            prefer_unknown = self.allow_unknown_state and local_delta <= self.comoving_delta_px
            if weak_geometry and self.force_unknown_on_weak_geometry:
                prefer_unknown = True

            evidence = self._motion_evidence(
                flow_med=flow_med,
                sampson_med=sampson_med,
                baseline_flow_med=baseline_flow_med,
                has_geometry=bool(has_geometry and enough_points),
            )
            depth_m = getattr(det, "depth_m", None)
            depth_weight = self._depth_attenuation(depth_m)
            evidence *= depth_weight

            edge_strength, edge_side = self._edge_strength(det.bbox, frame_w, frame_h)
            edge_effect = edge_strength * self.edge_flow_attenuation
            if edge_effect > 0.0:
                evidence *= (1.0 - edge_effect)
            if (
                self.allow_unknown_state
                and self.edge_unknown_if_low_ring
                and edge_strength >= 0.75
                and ring_points < self.edge_min_ring_points
            ):
                prefer_unknown = True

            track_prior = None
            track_reliability = 0.0
            track_unknown = False
            if track_id is not None and isinstance(track_features, dict):
                feature = track_features.get(int(track_id))
                track_prior, track_reliability, track_unknown = self._track_prior(feature)
                if track_prior is not None:
                    weight = self.track_prior_weight * max(0.3, track_reliability)
                    evidence = (1.0 - weight) * evidence + weight * track_prior
                if track_unknown and self.allow_unknown_state:
                    prefer_unknown = True

            if prefer_unknown:
                evidence = 0.5 * (self.parked_threshold + self.moving_threshold)

            if track_id is None:
                if prefer_unknown:
                    det.is_moving = False
                    det.is_parked = False
                    det.motion_state = "unknown"
                else:
                    det.is_moving = evidence >= self.moving_threshold
                    det.is_parked = evidence <= self.parked_threshold
                    det.motion_state = "moving" if det.is_moving else ("parked" if det.is_parked else "unknown")
                det.motion_confidence = _clamp01(abs(evidence - 0.5) * 2.0)
                det.motion_source = source
                det.motion_age_frames = 1
                det.motion_evidence = round(float(evidence), 4)
                det.motion_ring_flow_med = round(float(ring_flow_med), 4)
                det.motion_bg_flow_med = round(float(bg_flow_med), 4)
                det.motion_inlier_ratio = round(float(inlier_ratio), 4)
                det.motion_track_speed_px = round(float(track_features.get(int(track_id), {}).get("speed_px", 0.0)), 4) if (track_id is not None and isinstance(track_features, dict)) else None
                det.motion_track_reliability = round(float(track_reliability), 4)
                det.motion_state_source = "fused" if track_prior is not None else "flow"
                det.motion_edge_strength = round(float(edge_strength), 4)
                det.motion_edge_side = edge_side
                det.motion_depth_weight = round(float(depth_weight), 4)
                continue

            stable, confidence, age = self._update_track_state(int(track_id), evidence, frame_idx, prefer_unknown=prefer_unknown)
            det.is_moving = stable == "moving"
            det.is_parked = stable == "parked"
            det.motion_state = stable
            det.motion_confidence = confidence
            det.motion_source = source
            det.motion_age_frames = int(age)
            det.motion_evidence = round(float(evidence), 4)
            det.motion_ring_flow_med = round(float(ring_flow_med), 4)
            det.motion_bg_flow_med = round(float(bg_flow_med), 4)
            det.motion_inlier_ratio = round(float(inlier_ratio), 4)
            det.motion_track_speed_px = round(float(track_features.get(int(track_id), {}).get("speed_px", 0.0)), 4) if (track_id is not None and isinstance(track_features, dict)) else None
            det.motion_track_reliability = round(float(track_reliability), 4)
            det.motion_state_source = "fused" if track_prior is not None else "flow"
            det.motion_edge_strength = round(float(edge_strength), 4)
            det.motion_edge_side = edge_side
            det.motion_depth_weight = round(float(depth_weight), 4)