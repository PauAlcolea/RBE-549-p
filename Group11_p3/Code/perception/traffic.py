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
    traffic_light_style: str = "standard_vertical"  # renderer hook: "standard_vertical" | "wide_green_arrow_candidate" | "square_arrow_signal_candidate"

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
        self.reject_if_hsv_comparable = bool(tl_cfg.get("reject_if_hsv_comparable", True))
        self.comparable_hue_range_max = float(tl_cfg.get("comparable_hue_range_max", 12.0))
        self.comparable_sat_range_max = float(tl_cfg.get("comparable_sat_range_max", 30.0))
        self.comparable_val_range_max = float(tl_cfg.get("comparable_val_range_max", 35.0))
        self.hue_compare_min_s = float(tl_cfg.get("hue_compare_min_s", 35.0))
        self.hue_compare_min_v = float(tl_cfg.get("hue_compare_min_v", 35.0))
        self.square_arrow_aspect_min = float(tl_cfg.get("square_arrow_aspect_min", 0.75))
        self.square_arrow_aspect_max = float(tl_cfg.get("square_arrow_aspect_max", 1.35))
        self.square_arrow_max_area_ratio = float(tl_cfg.get("square_arrow_max_area_ratio", 0.008))
        self.square_arrow_prior_weight = float(tl_cfg.get("square_arrow_prior_weight", 0.40))
        self.square_arrow_bluegreen_hue_center_deg = float(tl_cfg.get("square_arrow_bluegreen_hue_center_deg", 176.0))
        self.square_arrow_bluegreen_hue_half_width_deg = float(tl_cfg.get("square_arrow_bluegreen_hue_half_width_deg", 26.0))
        self.square_arrow_bluegreen_min_v = float(tl_cfg.get("square_arrow_bluegreen_min_v", 170.0))
        self.square_arrow_bluegreen_max_s = float(tl_cfg.get("square_arrow_bluegreen_max_s", 110.0))
        self.square_arrow_bluegreen_weight = float(tl_cfg.get("square_arrow_bluegreen_weight", 0.9))
        self.square_arrow_red_dark_max_v = float(tl_cfg.get("square_arrow_red_dark_max_v", 95.0))
        self.square_arrow_red_dark_min_s = float(tl_cfg.get("square_arrow_red_dark_min_s", 20.0))
        self.square_arrow_red_dark_weight = float(tl_cfg.get("square_arrow_red_dark_weight", 0.55))

        self.score_weights = {
            "bright_ratio": float(tl_cfg.get("weight_bright_ratio", 0.45)),
            "sv_energy": float(tl_cfg.get("weight_sv_energy", 0.35)),
            "contrast": float(tl_cfg.get("weight_contrast", 0.15)),
            "color_prior": float(tl_cfg.get("weight_color_prior", 0.05)),
        }

        self._region_to_color = ["red", "yellow", "green"]
        self.last_debug_info: List[Dict[str, Any]] = []
        self.scene_name: str = ""

    def set_night_mode(self, enabled: bool) -> None:
        """Enable night classifier that favors low saturation and high value."""
        self.night_mode = bool(enabled)

    def set_scene_context(self, scene_name: str) -> None:
        """Provide sequence context for scene-specific heuristics."""
        self.scene_name = str(scene_name).strip().lower()

    @staticmethod
    def _style_from_geometry(use_dual_columns: bool, is_square_arrow_signal: bool) -> str:
        """Map ROI layout heuristic to exported style metadata."""
        if is_square_arrow_signal:
            return "square_arrow_signal_candidate"
        return "wide_green_arrow_candidate" if use_dual_columns else "standard_vertical"

    @staticmethod
    def _bbox_metrics(image_shape: Tuple[int, int], bbox: List[float]) -> Dict[str, float]:
        """Return width/height/aspect/area-ratio for a clamped bbox."""
        h_img, w_img = image_shape
        x1 = max(0.0, float(bbox[0]))
        y1 = max(0.0, float(bbox[1]))
        x2 = min(float(w_img), float(bbox[2]))
        y2 = min(float(h_img), float(bbox[3]))

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        aspect = width / max(height, 1e-6)
        area_ratio = (width * height) / max(float(w_img * h_img), 1e-6)
        return {
            "width": width,
            "height": height,
            "aspect": aspect,
            "area_ratio": area_ratio,
        }

    def _is_square_arrow_signal_candidate(self, bbox_metrics: Dict[str, float]) -> bool:
        """Heuristic for lane-control arrow signal heads (small, near-square boxes)."""
        aspect = float(bbox_metrics.get("aspect", 0.0))
        area_ratio = float(bbox_metrics.get("area_ratio", 1.0))
        return (
            self.square_arrow_aspect_min <= aspect <= self.square_arrow_aspect_max
            and area_ratio <= self.square_arrow_max_area_ratio
        )

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

    def detect(self, frame_bgr: np.ndarray, traffic_detections: list) -> List[TrafficLight]:
        """
        Detect traffic lights in one BGR frame and classify their color.

        Parameters
        ----------
        frame_bgr : np.ndarray
        traffic_detections : list
            Traffic-light detections, each with at least bbox/confidence fields.

        Returns
        -------
        list[TrafficLight]
        """
        lights = []
        self.last_debug_info = []

        for det in traffic_detections:
            det_label = getattr(det, "label", "traffic_light")
            if det_label != "traffic_light":
                continue

            det_bbox = getattr(det, "bbox", None)
            if det_bbox is None:
                continue

            bbox_metrics = self._bbox_metrics(frame_bgr.shape[:2], det_bbox)
            square_arrow_scene_enabled = self.scene_name == "scene2"
            is_square_arrow_signal = (
                square_arrow_scene_enabled
                and self._is_square_arrow_signal_candidate(bbox_metrics)
            )
            if is_square_arrow_signal:
                color, dbg = self._classify_square_arrow_fullbox_debug(frame_bgr, det_bbox)
            else:
                color, dbg = self._classify_color_region_on_debug(frame_bgr, det_bbox)

            traffic_light_style = self._style_from_geometry(
                use_dual_columns=bool(dbg.get("use_dual_columns", False)),
                is_square_arrow_signal=is_square_arrow_signal,
            )
            dbg["bbox"] = [float(v) for v in det_bbox]
            dbg["predicted_color"] = color
            dbg["traffic_light_style"] = traffic_light_style
            dbg["bbox_aspect"] = float(bbox_metrics["aspect"])
            dbg["bbox_area_ratio"] = float(bbox_metrics["area_ratio"])
            dbg["square_arrow_candidate"] = bool(is_square_arrow_signal)
            dbg["square_arrow_scene_enabled"] = bool(square_arrow_scene_enabled)
            self.last_debug_info.append(dbg)

            lights.append(TrafficLight(
                bbox=det_bbox,
                color=color,
                confidence=float(getattr(det, "confidence", 0.0)),
                depth_m=float(getattr(det, "depth_m", 0.0)),
                label=det_label,
                traffic_light_style=traffic_light_style,
            ))

        return lights

    def _classify_square_arrow_fullbox_debug(self, frame_bgr: np.ndarray, bbox: List[float]) -> Tuple[str, Dict[str, Any]]:
        """Classify square arrow signals using full bbox and only red-vs-green evidence."""
        x1, y1, x2, y2 = self._clamp_bbox(frame_bgr.shape[:2], bbox)
        crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else np.empty((0, 0, 3), dtype=frame_bgr.dtype)
        frame_w = max(float(frame_bgr.shape[1]), 1.0)
        cx = 0.5 * (float(x1) + float(x2))
        x_norm = float(np.clip(cx / frame_w, 0.0, 1.0))
        green_prior = x_norm
        red_prior = 1.0 - x_norm
        prior_weight = float(self.square_arrow_prior_weight)
        prior_green_score = float(prior_weight * green_prior)
        prior_red_score = float(prior_weight * red_prior)

        debug_info: Dict[str, Any] = {
            "mode": "square_arrow_fullbox",
            "night_mode": bool(self.night_mode),
            "crop_rect": [x1, y1, x2, y2],
            "selection_metric": "red_green_fullbox_with_side_prior",
            "use_dual_columns": False,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
            "red_score": 0.0,
            "green_score": 0.0,
            "bluegreen_ratio": 0.0,
            "dark_red_ratio": 0.0,
            "left_right_norm": x_norm,
            "red_prior": float(red_prior),
            "green_prior": float(green_prior),
            "prior_weight": prior_weight,
            "winner_idx": None,
        }

        # For square arrow heads, always force a binary decision (red or green).
        if crop.size == 0:
            debug_info["red_score"] = prior_red_score
            debug_info["green_score"] = prior_green_score
            if prior_green_score >= prior_red_score:
                debug_info["winner_idx"] = 1
                return "green", debug_info
            debug_info["winner_idx"] = 0
            return "red", debug_info

        if crop.shape[0] < 6 or crop.shape[1] < 6:
            debug_info["red_score"] = prior_red_score
            debug_info["green_score"] = prior_green_score
            if prior_green_score >= prior_red_score:
                debug_info["winner_idx"] = 1
                return "green", debug_info
            debug_info["winner_idx"] = 0
            return "red", debug_info

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        roi_mask = np.ones(hsv.shape[:2], dtype=bool)
        hue = hsv[:, :, 0].astype(np.float32)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        red_ratio_base = self._expected_color_ratio_masked(hsv, "red", roi_mask)
        green_ratio_base = self._expected_color_ratio_masked(hsv, "green", roi_mask)

        # User-requested heuristic: green can skew blue/cyan with low saturation and high value.
        hue_center_cv = float(np.clip(self.square_arrow_bluegreen_hue_center_deg / 2.0, 0.0, 179.0))
        hue_half_cv = max(1.0, float(self.square_arrow_bluegreen_hue_half_width_deg / 2.0))
        hue_lo = max(0.0, hue_center_cv - hue_half_cv)
        hue_hi = min(179.0, hue_center_cv + hue_half_cv)
        bluegreen_mask = (
            (hue >= hue_lo)
            & (hue <= hue_hi)
            & (val >= self.square_arrow_bluegreen_min_v)
            & (sat <= self.square_arrow_bluegreen_max_s)
        )
        bluegreen_ratio = float(np.mean(bluegreen_mask))

        # User-requested heuristic: red can be very dark.
        # Important: require red hue so dark background does not dominate as "red".
        red_hue_mask = ((hue <= 10.0) | (hue >= 170.0))
        dark_red_mask = (
            red_hue_mask
            & (val <= self.square_arrow_red_dark_max_v)
            & (sat >= self.square_arrow_red_dark_min_s)
        )
        dark_red_ratio = float(np.mean(dark_red_mask))

        green_ratio = float(green_ratio_base + (self.square_arrow_bluegreen_weight * bluegreen_ratio))
        red_ratio = float(red_ratio_base + (self.square_arrow_red_dark_weight * dark_red_ratio))

        red_score = float(red_ratio + prior_red_score)
        green_score = float(green_ratio + prior_green_score)

        debug_info["red_ratio"] = float(red_ratio)
        debug_info["green_ratio"] = float(green_ratio)
        debug_info["bluegreen_ratio"] = float(bluegreen_ratio)
        debug_info["dark_red_ratio"] = float(dark_red_ratio)
        debug_info["red_score"] = red_score
        debug_info["green_score"] = green_score
        debug_info["left_right_norm"] = float(x_norm)
        debug_info["red_prior"] = float(red_prior)
        debug_info["green_prior"] = float(green_prior)

        if green_score >= red_score:
            debug_info["winner_idx"] = 1
            return "green", debug_info

        debug_info["winner_idx"] = 0
        return "red", debug_info

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
            "hsv_comparable_reject": False,
            "hsv_spreads": {},
            "hue_means": [],
            "saturation_means": [],
            "value_means": [],
            "use_dual_columns": False,
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
        debug_info["hue_means"] = [float(v) for v in geom.get("hue_means", [])]
        debug_info["saturation_means"] = [float(v) for v in geom.get("saturation_means", [])]
        debug_info["value_means"] = [float(v) for v in geom.get("value_means", [])]
        debug_info["use_dual_columns"] = bool(geom.get("use_dual_columns", False))

        comparable, spreads = self._is_hsv_comparable(
            debug_info["hue_means"],
            debug_info["saturation_means"],
            debug_info["value_means"],
        )
        debug_info["hsv_spreads"] = spreads
        if self.reject_if_hsv_comparable and comparable:
            debug_info["hsv_comparable_reject"] = True
            debug_info["selection_metric"] = "hsv_comparable_reject"
            return "unknown", debug_info

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

        # Decide wide-light behavior before focus cropping shrinks the width.
        pre_focus_aspect = float(iw) / max(float(ih), 1.0)
        use_dual_columns = pre_focus_aspect >= self.wide_aspect_threshold

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
            "use_dual_columns": bool(use_dual_columns),
        }

        scores, roi_geom = self._score_bulb_rois(inner, global_v_thr, use_dual_columns)
        # Shift ROI centers from inner-local to crop-local coords.
        geom["roi_circles"] = [
            [inner_x0 + int(cx), inner_y0 + int(cy), int(r), int(region_idx)]
            for (cx, cy, r, region_idx) in roi_geom.get("roi_circles", [])
        ]
        geom["hue_means"] = [float(v) for v in roi_geom.get("hue_means", [])]
        geom["saturation_means"] = [float(v) for v in roi_geom.get("saturation_means", [])]
        geom["value_means"] = [float(v) for v in roi_geom.get("value_means", [])]
        return scores, geom

    def _score_bulb_rois(
        self,
        hsv_crop: np.ndarray,
        global_v_thr: float,
        use_dual_columns: bool,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Score compact circular ROIs near expected bulb centers (top/mid/bottom)."""
        h, w = hsv_crop.shape[:2]
        if h < 3 or w < 3:
            return [0.0, 0.0, 0.0], {"roi_circles": []}

        sat = hsv_crop[:, :, 1].astype(np.float32)
        val = hsv_crop[:, :, 2].astype(np.float32)
        hue = hsv_crop[:, :, 0].astype(np.float32)
        on_global = (val >= global_v_thr) & (sat >= self.min_on_s)

        band_h = h / 3.0
        radius = max(1, int(self.bulb_roi_radius_ratio * band_h))
        cx = int(w * 0.5)

        yy, xx = np.ogrid[:h, :w]
        scores: List[float] = []
        circles: List[List[int]] = []
        hue_means: List[float] = []
        sat_means: List[float] = []
        val_means: List[float] = []

        # Wide/arrow style traffic lights can have side-by-side bulbs per row.
        if use_dual_columns:
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
                hue_mask = roi & (sat >= self.hue_compare_min_s) & (val >= self.hue_compare_min_v)
                if np.any(hue_mask):
                    hue_means.append(float(np.mean(hue[hue_mask])))
                else:
                    hue_means.append(float("nan"))
                sat_means.append(float(np.mean(sat[roi])))
                val_means.append(float(np.mean(val[roi])))
            else:
                hue_means.append(float("nan"))
                sat_means.append(255.0)
                val_means.append(0.0)
        return scores, {
            "roi_circles": circles,
            "hue_means": hue_means,
            "saturation_means": sat_means,
            "value_means": val_means,
        }

    def _hue_circular_spread(self, hue_values: List[float]) -> float:
        """Return circular spread for OpenCV hue values in [0, 179]. Lower means more similar."""
        vals = np.array(hue_values, dtype=np.float32)
        vals = np.mod(vals, 180.0)
        if vals.size < 2:
            return 0.0

        vals = np.sort(vals)
        wrap_gap = (vals[0] + 180.0) - vals[-1]
        gaps = np.diff(vals)
        max_gap = max(float(wrap_gap), float(np.max(gaps)) if gaps.size > 0 else 0.0)
        return float(180.0 - max_gap)

    def _is_hsv_comparable(
        self,
        hue_means: List[float],
        sat_means: List[float],
        val_means: List[float],
    ) -> Tuple[bool, Dict[str, float]]:
        """Check whether ROI HSV statistics are too similar to confidently pick a light color."""
        spreads = {"hue": float("inf"), "sat": float("inf"), "val": float("inf")}
        if len(hue_means) != 3 or len(sat_means) != 3 or len(val_means) != 3:
            return False, spreads

        hue_arr = np.array(hue_means, dtype=np.float32)
        sat_arr = np.array(sat_means, dtype=np.float32)
        val_arr = np.array(val_means, dtype=np.float32)
        if not np.all(np.isfinite(hue_arr)):
            return False, spreads
        if not np.all(np.isfinite(sat_arr)) or not np.all(np.isfinite(val_arr)):
            return False, spreads

        spreads["hue"] = self._hue_circular_spread(hue_means)
        spreads["sat"] = float(np.max(sat_arr) - np.min(sat_arr))
        spreads["val"] = float(np.max(val_arr) - np.min(val_arr))

        comparable = (
            spreads["hue"] <= self.comparable_hue_range_max
            and spreads["sat"] <= self.comparable_sat_range_max
            and spreads["val"] <= self.comparable_val_range_max
        )
        return comparable, spreads

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
