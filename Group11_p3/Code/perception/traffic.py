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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2


@dataclass
class TrafficLight:
    bbox: List[float]       # [x1, y1, x2, y2]
    color: str              # "red" | "yellow" | "green" | "unknown"
    confidence: float
    depth_m: float = 0.0
    label: str = "traffic_light"   # for consistency with Detection
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # optional, can be filled in by DepthEstimator.lift_to_3d

class TrafficLightDetector:
    """
    Detects traffic lights and classifies their color.

        One-stage color classifier on YOLO traffic-light boxes.

    Day mode: weighted region scoring over bulb ROIs.
    Night mode: choose region minimizing (normalized_saturation - normalized_value).

    Usage
    -----
    detector = TrafficLightDetector(cfg, device="cuda")
    lights = detector.detect(frame_bgr, all_detections)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        tl_cfg = cfg["perception"]["traffic_light"]
        self.hsv_ranges = self._build_hsv_ranges(tl_cfg)

        # Region-on classifier tuning (defaults keep behavior stable if keys are absent)
        self.force_best_guess = bool(tl_cfg.get("force_best_guess", True))
        self.region_margin_ratio = float(tl_cfg.get("region_margin_ratio", 0.08))
        self.center_focus_ratio = float(tl_cfg.get("center_focus_ratio", 0.6))
        self.bulb_roi_radius_ratio = float(tl_cfg.get("bulb_roi_radius_ratio", 0.38))
        self.wide_aspect_threshold = float(tl_cfg.get("wide_aspect_threshold", 0.62))
        self.wide_column_offset_ratio = float(tl_cfg.get("wide_column_offset_ratio", 0.22))
        self.night_mode = bool(tl_cfg.get("night_mode_default", False))
        self.on_v_percentile = float(tl_cfg.get("on_v_percentile", 85.0))
        self.min_on_v = float(tl_cfg.get("min_on_v", 140.0))
        self.min_on_s = float(tl_cfg.get("min_on_s", 50.0))
        self.unknown_score_epsilon = float(tl_cfg.get("unknown_score_epsilon", 0.02))
        self.degenerate_score_epsilon = float(tl_cfg.get("degenerate_score_epsilon", 1e-5))

        self.score_weights = {
            "bright_ratio": float(tl_cfg.get("weight_bright_ratio", 0.45)),
            "sv_energy": float(tl_cfg.get("weight_sv_energy", 0.35)),
            "contrast": float(tl_cfg.get("weight_contrast", 0.15)),
            "color_prior": float(tl_cfg.get("weight_color_prior", 0.05)),
        }

        self._region_to_color = ["red", "yellow", "green"]
        self.last_debug_info: List[Dict[str, Any]] = []

    def set_night_mode(self, enabled: bool) -> None:
        """Enable night classifier that favors low saturation and high value."""
        self.night_mode = bool(enabled)

    def _build_hsv_ranges(self, tl_cfg: dict):
        """Package HSV range lists into a dict for color lookup."""
        return {
            "red": [
                (np.array(tl_cfg["hsv_red_low"],    dtype=np.uint8),
                 np.array(tl_cfg["hsv_red_high"],   dtype=np.uint8)),
                # Red wraps around hue=0 in OpenCV HSV, so we need a second range
                (np.array(tl_cfg["hsv_red_low2"],   dtype=np.uint8),
                 np.array(tl_cfg["hsv_red_high2"],  dtype=np.uint8)),
            ],
            "yellow": [
                (np.array(tl_cfg["hsv_yellow_low"],  dtype=np.uint8),
                 np.array(tl_cfg["hsv_yellow_high"], dtype=np.uint8)),
            ],
            "green": [
                (np.array(tl_cfg["hsv_green_low"],  dtype=np.uint8),
                 np.array(tl_cfg["hsv_green_high"], dtype=np.uint8)),
            ],
        }

    def detect(self, frame_bgr: np.ndarray, object_detections: list) -> List[TrafficLight]:
        """
        Detect traffic lights in one BGR frame and classify their color.

        Parameters
        ----------
        frame_bgr : np.ndarray
        object_detections : list[Detection]
            Detections from the object detector; only traffic_light entries are used.

        Returns
        -------
        list[TrafficLight]
        """
        lights = []
        self.last_debug_info = []

        for det in object_detections:
            if det.label != "traffic_light":
                continue

            color, dbg = self._classify_color_region_on_debug(frame_bgr, det.bbox)
            dbg["bbox"] = [float(v) for v in det.bbox]
            dbg["predicted_color"] = color
            self.last_debug_info.append(dbg)

            lights.append(TrafficLight(
                bbox=det.bbox,
                color=color,
                confidence=det.confidence,
                depth_m=det.depth_m,
                label=det.label
            ))

        return lights

    def _clamp_bbox(self, image_shape: Tuple[int, int], bbox: List[float]) -> Tuple[int, int, int, int]:
        """Clamp float bbox coordinates to valid integer image bounds."""
        h_img, w_img = image_shape
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w_img, int(bbox[2]))
        y2 = min(h_img, int(bbox[3]))
        return x1, y1, x2, y2

    def _classify_color_region_on_debug(self, frame_bgr: np.ndarray, bbox: List[float]) -> Tuple[str, Dict[str, Any]]:
        """
        Infer active bulb by region illumination, not explicit color lookup.

        The crop is split into top/middle/bottom bands mapped to red/yellow/green.
        Each band receives an "on" score based on brightness, saturation, contrast,
        and a weak expected-color prior used only as a tie-breaker.
        """
        x1, y1, x2, y2 = self._clamp_bbox(frame_bgr.shape[:2], bbox)
        crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.empty((0, 0, 3), dtype=frame_bgr.dtype)

        debug_info: Dict[str, Any] = {
            "mode": "region_on",
            "night_mode": bool(self.night_mode),
            "crop_rect": [x1, y1, x2, y2],
            "region_scores": [],
            "winner_idx": None,
            "global_v_thr": None,
            "focus_rect": None,
            "band_lines_y": [],
            "roi_circles": [],
            "selection_metric": "score",
            "saturation_means": [],
            "value_means": [],
        }

        if crop.size == 0:
            return "unknown", debug_info

        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return "unknown", debug_info

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        region_scores, geom = self._score_bulb_regions(hsv)
        debug_info["region_scores"] = [float(s) for s in region_scores]
        debug_info["global_v_thr"] = float(geom.get("global_v_thr", 0.0))

        if len(region_scores) != 3:
            return "unknown", debug_info

        # Translate crop-local geometry into frame coordinates for visualization.
        fx1, fy1, fx2, fy2 = geom.get("focus_rect", [0, 0, 0, 0])
        debug_info["focus_rect"] = [x1 + int(fx1), y1 + int(fy1), x1 + int(fx2), y1 + int(fy2)]
        debug_info["band_lines_y"] = [y1 + int(v) for v in geom.get("band_lines_y", [])]
        debug_info["roi_circles"] = [
            [x1 + int(cx), y1 + int(cy), int(r), int(region_idx)]
            for (cx, cy, r, region_idx) in geom.get("roi_circles", [])
        ]
        debug_info["saturation_means"] = [float(v) for v in geom.get("saturation_means", [])]
        debug_info["value_means"] = [float(v) for v in geom.get("value_means", [])]

        if self.night_mode:
            sat_means = debug_info["saturation_means"]
            val_means = debug_info["value_means"]
            if len(sat_means) != 3 or len(val_means) != 3:
                return "unknown", debug_info

            sat_arr = np.array(sat_means, dtype=np.float32) / 255.0
            val_arr = np.array(val_means, dtype=np.float32) / 255.0
            # Lower metric is better: low saturation and high value.
            night_metric = sat_arr - val_arr
            winner_idx = int(np.argmin(night_metric))
            debug_info["winner_idx"] = winner_idx
            debug_info["selection_metric"] = "lowest_sat_high_val"
            debug_info["region_scores"] = [float(v) for v in night_metric.tolist()]
            return self._region_to_color[winner_idx], debug_info

        top_score = max(region_scores)
        # Degenerate: no meaningful evidence in any region
        if top_score <= self.degenerate_score_epsilon:
            return "unknown", debug_info

        winner_idx = int(np.argmax(region_scores))
        debug_info["winner_idx"] = winner_idx
        if (not self.force_best_guess) and top_score < self.unknown_score_epsilon:
            return "unknown", debug_info

        return self._region_to_color[winner_idx], debug_info

    def _score_bulb_regions(self, hsv_crop: np.ndarray) -> Tuple[List[float], Dict[str, Any]]:
        """Compute on-state scores for top/middle/bottom bulb regions."""
        h, w = hsv_crop.shape[:2]
        if h < 3 or w < 3:
            return [0.0, 0.0, 0.0], {}

        margin_x = int(w * self.region_margin_ratio)
        margin_y = int(h * self.region_margin_ratio)
        x1 = min(max(margin_x, 0), w - 1)
        x2 = max(min(w - margin_x, w), x1 + 1)
        y1 = min(max(margin_y, 0), h - 1)
        y2 = max(min(h - margin_y, h), y1 + 1)

        inner = hsv_crop[y1:y2, x1:x2]
        ih, iw = inner.shape[:2]
        inner_x0, inner_y0 = x1, y1
        if ih < 3 or iw < 3:
            inner = hsv_crop
            ih, iw = h, w
            inner_x0, inner_y0 = 0, 0

        # Focus on the central columns to suppress bright clutter near bbox edges.
        focus_ratio = min(max(self.center_focus_ratio, 0.2), 1.0)
        focus_w = max(int(iw * focus_ratio), 1)
        cx = iw // 2
        fx1 = max(cx - (focus_w // 2), 0)
        fx2 = min(fx1 + focus_w, iw)
        inner = inner[:, fx1:fx2, :]
        inner_x0 += fx1
        ih, iw = inner.shape[:2]
        if ih < 3 or iw < 1:
            return [0.0, 0.0, 0.0], {}

        # Use one threshold for the whole crop so only truly active regions survive.
        inner_val = inner[:, :, 2].astype(np.float32)
        global_adaptive_v = np.percentile(inner_val, self.on_v_percentile)
        global_v_thr = max(self.min_on_v, global_adaptive_v)

        y_line1 = int(round(inner_y0 + (ih / 3.0)))
        y_line2 = int(round(inner_y0 + (2.0 * ih / 3.0)))
        geom: Dict[str, Any] = {
            "focus_rect": [inner_x0, inner_y0, inner_x0 + iw, inner_y0 + ih],
            "band_lines_y": [y_line1, y_line2],
            "global_v_thr": float(global_v_thr),
            "saturation_means": [],
            "value_means": [],
        }

        scores, roi_geom = self._score_bulb_rois(inner, global_v_thr)
        # Shift ROI centers from inner-local to crop-local coords.
        geom["roi_circles"] = [
            [inner_x0 + int(cx), inner_y0 + int(cy), int(r), int(region_idx)]
            for (cx, cy, r, region_idx) in roi_geom.get("roi_circles", [])
        ]
        geom["saturation_means"] = [float(v) for v in roi_geom.get("saturation_means", [])]
        geom["value_means"] = [float(v) for v in roi_geom.get("value_means", [])]
        return scores, geom

    def _score_bulb_rois(self, hsv_crop: np.ndarray, global_v_thr: float) -> Tuple[List[float], Dict[str, Any]]:
        """Score compact circular ROIs near expected bulb centers (top/mid/bottom)."""
        h, w = hsv_crop.shape[:2]
        if h < 3 or w < 3:
            return [0.0, 0.0, 0.0], {"roi_circles": []}

        sat = hsv_crop[:, :, 1].astype(np.float32)
        val = hsv_crop[:, :, 2].astype(np.float32)
        on_global = (val >= global_v_thr) & (sat >= self.min_on_s)

        band_h = h / 3.0
        radius = max(1, int(self.bulb_roi_radius_ratio * band_h))
        cx = int(w * 0.5)

        yy, xx = np.ogrid[:h, :w]
        scores: List[float] = []
        circles: List[List[int]] = []
        sat_means: List[float] = []
        val_means: List[float] = []

        # Wide/arrow style traffic lights can have side-by-side bulbs per row.
        aspect = float(w) / max(float(h), 1.0)
        if aspect >= self.wide_aspect_threshold:
            off = int(round(w * self.wide_column_offset_ratio))
            col_centers = [int(np.clip(cx - off, 0, w - 1)), int(np.clip(cx + off, 0, w - 1))]
        else:
            col_centers = [cx]

        for idx, color_name in enumerate(self._region_to_color):
            cy = int((idx + 0.5) * band_h)
            roi = np.zeros((h, w), dtype=bool)
            for cx_col in col_centers:
                roi |= (((xx - cx_col) ** 2 + (yy - cy) ** 2) <= (radius * radius))
                circles.append([int(cx_col), int(cy), int(radius), int(idx)])

            scores.append(self._score_single_region(hsv_crop, color_name, roi, on_global))
            roi_pixels = np.count_nonzero(roi)
            if roi_pixels > 0:
                sat_means.append(float(np.mean(sat[roi])))
                val_means.append(float(np.mean(val[roi])))
            else:
                sat_means.append(255.0)
                val_means.append(0.0)
        return scores, {"roi_circles": circles, "saturation_means": sat_means, "value_means": val_means}

    def _score_single_region(
        self,
        hsv_region: np.ndarray,
        expected_color: str,
        roi_mask: np.ndarray,
        on_mask_global: np.ndarray,
    ) -> float:
        """Score one region for whether its expected bulb appears illuminated."""
        if hsv_region.size == 0 or roi_mask.size == 0:
            return 0.0

        sat = hsv_region[:, :, 1].astype(np.float32)
        val = hsv_region[:, :, 2].astype(np.float32)

        on_mask = on_mask_global & roi_mask
        roi_pixels = float(np.count_nonzero(roi_mask))
        if roi_pixels <= 0:
            return 0.0

        if not np.any(on_mask):
            return 0.0

        bright_ratio = float(np.count_nonzero(on_mask) / roi_pixels)
        sv_energy = float((np.mean(val[on_mask]) / 255.0) * (np.mean(sat[on_mask]) / 255.0))

        off_mask = roi_mask & (~on_mask)
        if np.any(off_mask):
            contrast = float(max(0.0, np.mean(val[on_mask]) - np.mean(val[off_mask])) / 255.0)
        else:
            contrast = float(np.mean(val[on_mask]) / 255.0)

        color_prior = self._expected_color_ratio_masked(hsv_region, expected_color, roi_mask)

        w = self.score_weights
        return (
            (w["bright_ratio"] * bright_ratio)
            + (w["sv_energy"] * sv_energy)
            + (w["contrast"] * contrast)
            + (w["color_prior"] * color_prior)
        )

    def _expected_color_ratio_masked(self, hsv_region: np.ndarray, color_name: str, roi_mask: np.ndarray) -> float:
        """Weak prior: fraction of ROI matching expected color HSV range."""
        ranges: List[Tuple[np.ndarray, np.ndarray]] = self.hsv_ranges.get(color_name, [])
        if not ranges:
            return 0.0

        mask = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv_region, lo, hi)

        roi_pixels = float(np.count_nonzero(roi_mask))
        if roi_pixels <= 0:
            return 0.0
        return float(np.count_nonzero((mask > 0) & roi_mask) / roi_pixels)
