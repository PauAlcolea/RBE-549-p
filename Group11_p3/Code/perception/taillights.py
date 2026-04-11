"""
perception/taillights.py
========================
Detic-based taillight detection and per-vehicle brake-light state inference.

The detector is optional: if Detic is unavailable, this module keeps the
pipeline running and vehicle state falls back to "off".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os
import sys

import cv2
import numpy as np


_VALID_STATES = {"on", "off", "left indicator", "right indicator"}
_VALID_SIDE_STATES = {"on", "off", "not_detected"}


@dataclass
class TailLightDetection:
    label: str = "taillight"
    bbox: List[float] = field(default_factory=list)
    confidence: float = 0.0
    depth_m: float = 0.0
    position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    vehicle_track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    side: str = "unknown"  # left | right | unknown
    prompt: str = "taillight"
    activation_score: float = 0.0
    status: str = "off"  # on | off (HSV-based)


def _clamp_bbox_xyxy(box: List[float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if w <= 0 or h <= 0:
        return None
    x1 = max(0, min(int(round(float(box[0]))), w - 1))
    y1 = max(0, min(int(round(float(box[1]))), h - 1))
    x2 = max(0, min(int(round(float(box[2]))), w))
    y2 = max(0, min(int(round(float(box[3]))), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class TailLightDetector:
    """
    Detect taillights only inside already-detected vehicle ROIs.

    The detector also sets:
      vehicle.brake_light_state: "on" | "off" | "left indicator" | "right indicator"
      vehicle.brake_light_confidence: float
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = str(device)
        self.predictor = None
        self.active = False

        taillight_cfg = cfg.get("perception", {}).get("taillights_detic", {})
        self.enabled = bool(taillight_cfg.get("enabled", True))
        self.confidence = float(taillight_cfg.get("confidence", 0.20))
        self.max_per_vehicle = max(1, int(taillight_cfg.get("max_per_vehicle", 4)))
        self.min_width_px = float(taillight_cfg.get("min_width_px", 2.0))
        self.min_height_px = float(taillight_cfg.get("min_height_px", 2.0))
        self.min_vehicle_crop_side_px = int(taillight_cfg.get("min_vehicle_crop_side_px", 16))
        self.state_lit_score_threshold = float(taillight_cfg.get("state_lit_score_threshold", 0.05))
        self.state_indicator_ratio = float(taillight_cfg.get("state_indicator_ratio", 1.7))
        self.side_center_margin_ratio = float(taillight_cfg.get("side_center_margin_ratio", 0.10))
        self.state_red_ratio_weight = float(taillight_cfg.get("state_red_ratio_weight", 0.80))
        self.state_value_weight = float(taillight_cfg.get("state_value_weight", 0.20))
        weights_sum = self.state_red_ratio_weight + self.state_value_weight
        if weights_sum <= 1e-6:
            self.state_red_ratio_weight = 0.80
            self.state_value_weight = 0.20
        else:
            self.state_red_ratio_weight /= weights_sum
            self.state_value_weight /= weights_sum
        self.state_detection_conf_floor = float(taillight_cfg.get("state_detection_conf_floor", 0.60))
        self.state_detection_conf_gamma = float(taillight_cfg.get("state_detection_conf_gamma", 0.50))
        self.state_roi_margin_ratio = float(taillight_cfg.get("state_roi_margin_ratio", 0.12))
        self.state_detection_conf_floor = max(0.0, min(self.state_detection_conf_floor, 1.0))
        self.state_detection_conf_gamma = max(0.05, self.state_detection_conf_gamma)
        self.state_roi_margin_ratio = max(0.0, min(self.state_roi_margin_ratio, 0.45))
        self.night_mode = False

        self.red_low1 = np.array(taillight_cfg.get("hsv_red_low1", [0, 80, 80]), dtype=np.uint8)
        self.red_high1 = np.array(taillight_cfg.get("hsv_red_high1", [12, 255, 255]), dtype=np.uint8)
        self.red_low2 = np.array(taillight_cfg.get("hsv_red_low2", [170, 80, 80]), dtype=np.uint8)
        self.red_high2 = np.array(taillight_cfg.get("hsv_red_high2", [180, 255, 255]), dtype=np.uint8)
        night_cfg = taillight_cfg.get("night", {})
        self.night_state_lit_score_threshold = float(
            night_cfg.get("state_lit_score_threshold", self.state_lit_score_threshold)
        )
        self.night_red_low1 = np.array(
            night_cfg.get("hsv_red_low1", self.red_low1.tolist()),
            dtype=np.uint8,
        )
        self.night_red_high1 = np.array(
            night_cfg.get("hsv_red_high1", self.red_high1.tolist()),
            dtype=np.uint8,
        )
        self.night_red_low2 = np.array(
            night_cfg.get("hsv_red_low2", self.red_low2.tolist()),
            dtype=np.uint8,
        )
        self.night_red_high2 = np.array(
            night_cfg.get("hsv_red_high2", self.red_high2.tolist()),
            dtype=np.uint8,
        )
        # Night-specific anti-reflector controls (used only when --night).
        self.night_min_lit_value_mean = float(night_cfg.get("min_lit_value_mean", 185.0))
        self.night_max_red_ratio = float(night_cfg.get("max_red_ratio", 0.45))
        self.night_min_lit_value_mean = max(0.0, min(self.night_min_lit_value_mean, 255.0))
        self.night_max_red_ratio = max(0.01, min(self.night_max_red_ratio, 1.0))

        prompts = taillight_cfg.get(
            "prompts",
            [
                "left rear turn indicator lit",
                "right rear turn indicator lit",
                "vehicle brake light lit",
                "car taillight",
            ],
        )
        self.prompts = [str(p).strip() for p in prompts if str(p).strip()]
        if not self.prompts:
            self.prompts = ["car taillight"]

        self.code_dir = Path(__file__).resolve().parents[1]
        self.detectron2_dir = self.code_dir / "perception" / "detectron2"
        self.detic_dir = self._resolve_path(taillight_cfg.get("detic_repo", "perception/Detic"))
        self.config_path = self._resolve_repo_relative_path(
            self.detic_dir,
            taillight_cfg.get("config_file", "configs/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"),
        )
        self.weights_path = self._resolve_path(
            taillight_cfg.get("weights", "../Weights/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
        )

        if self.enabled:
            self._load_predictor()

    def is_active(self) -> bool:
        return bool(self.active and self.predictor is not None)

    def set_night_mode(self, enabled: bool) -> None:
        self.night_mode = bool(enabled)

    def _lit_threshold(self) -> float:
        return self.night_state_lit_score_threshold if self.night_mode else self.state_lit_score_threshold

    def _resolve_path(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None

        p = Path(raw)
        if p.is_absolute():
            return str(p)

        candidates = [self.code_dir / p, Path.cwd() / p]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(self.code_dir / p)

    @staticmethod
    def _resolve_repo_relative_path(repo_dir: Optional[str], value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute():
            return str(p)
        if repo_dir is None:
            return None
        return str(Path(repo_dir) / p)

    def _build_clip_classifier(self, torch_module):
        try:
            from detic.predictor import get_clip_embeddings  # type: ignore

            classifier = get_clip_embeddings(self.prompts)
            if classifier is None:
                return None
            if not torch_module.is_tensor(classifier):
                classifier = torch_module.tensor(classifier)
            if classifier.dim() != 2:
                return None
            if classifier.shape[1] == len(self.prompts):
                return classifier.float().cpu()
            if classifier.shape[0] == len(self.prompts):
                return classifier.t().contiguous().float().cpu()
            return None
        except Exception:
            return None

    def _load_predictor(self) -> None:
        if self.detic_dir is None or not Path(self.detic_dir).exists():
            print("[taillights] Detic repo path does not exist; detector disabled.")
            return

        detic_root = Path(self.detic_dir)
        detic_pkg_ok = (detic_root / "detic" / "config.py").exists()
        if not detic_pkg_ok:
            print(
                "[taillights] Detic package not found under configured repo. "
                "Expected detic/config.py. "
                f"Current path: {detic_root}"
            )
            print(
                "[taillights] Note: facebookresearch/detectron2 alone is not Detic. "
                "Detic additionally needs the Detic repo and CenterNet2."
            )
            return

        if str(detic_root) not in sys.path:
            sys.path.insert(0, str(detic_root))
        if self.detectron2_dir.exists() and str(self.detectron2_dir) not in sys.path:
            sys.path.insert(0, str(self.detectron2_dir))

        # CenterNet2 provides the `centernet` Python module that Detic imports.
        centernet_candidates = [
            self.code_dir / "perception" / "CenterNet2",
            detic_root / "third_party" / "CenterNet2",
            detic_root.parent / "CenterNet2",
        ]
        centernet_ok = False
        for root in centernet_candidates:
            if (root / "centernet").exists():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                centernet_ok = True
                break
        if not centernet_ok:
            print(
                "[taillights] CenterNet2 not found (missing `centernet` module). "
                "Expected one of: "
                f"{', '.join(str(p) for p in centernet_candidates)}"
            )
            return

        try:
            import torch
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from centernet.config import add_centernet_config  # type: ignore
            from detic.config import add_detic_config  # type: ignore
            from detic.modeling.utils import reset_cls_test  # type: ignore
        except Exception as exc:
            print(f"[taillights] Detic import failed ({exc}); detector disabled.")
            print(
                "[taillights] Required modules are: detectron2, detic, and centernet "
                "(from CenterNet2)."
            )
            return

        if self.config_path is None or not Path(self.config_path).exists():
            print(f"[taillights] Detic config missing at {self.config_path}; detector disabled.")
            return

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(str(self.config_path))
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        lvis_cat_info_path = detic_root / "datasets" / "metadata" / "lvis_v1_train_cat_info.json"
        if lvis_cat_info_path.exists() and hasattr(cfg.MODEL.ROI_BOX_HEAD, "CAT_FREQ_PATH"):
            cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(lvis_cat_info_path)
        if self.weights_path is not None and Path(self.weights_path).exists():
            cfg.MODEL.WEIGHTS = str(self.weights_path)

        try:
            # Detic uses some hardcoded relative metadata paths (datasets/...).
            # Initialize from the Detic repo root so those paths resolve.
            old_cwd = os.getcwd()
            try:
                os.chdir(str(detic_root))
                predictor = DefaultPredictor(cfg)
                classifier = self._build_clip_classifier(torch)
                if classifier is None:
                    print("[taillights] Could not build CLIP classifier; detector disabled.")
                    return
                reset_cls_test(predictor.model, classifier, len(self.prompts))
            finally:
                os.chdir(old_cwd)
        except Exception as exc:
            print(f"[taillights] Failed to initialize predictor ({exc}); detector disabled.")
            return

        self.predictor = predictor
        self.active = True
        print(
            "[taillights] active (vehicle-ROI mode) with prompts="
            f"{self.prompts}, conf={self.confidence:.2f}, max_per_vehicle={self.max_per_vehicle}"
        )

    def _prompt_side(self, prompt: str) -> str:
        txt = str(prompt).strip().lower()
        has_left = "left" in txt
        has_right = "right" in txt
        if has_left and not has_right:
            return "left"
        if has_right and not has_left:
            return "right"
        return "unknown"

    def _geo_side(self, box: List[float], vehicle_bbox: List[float]) -> str:
        vx1, _, vx2, _ = [float(v) for v in vehicle_bbox]
        vcx = 0.5 * (vx1 + vx2)
        vw = max(vx2 - vx1, 1.0)
        margin = self.side_center_margin_ratio * vw
        cx = 0.5 * (float(box[0]) + float(box[2]))
        if cx < (vcx - margin):
            return "left"
        if cx > (vcx + margin):
            return "right"
        return "unknown"

    def _red_activation_score(self, hsv_frame: np.ndarray, box: List[float], det_conf: float, prompt: str) -> float:
        h, w = hsv_frame.shape[:2]
        clamped = _clamp_bbox_xyxy(box, w, h)
        if clamped is None:
            return 0.0
        x1, y1, x2, y2 = clamped

        # Ignore bbox borders so the score reflects the lamp core more than body paint.
        bw = x2 - x1
        bh = y2 - y1
        mx = int(round(self.state_roi_margin_ratio * bw))
        my = int(round(self.state_roi_margin_ratio * bh))
        if (bw - (2 * mx)) >= 2 and (bh - (2 * my)) >= 2:
            x1 += mx
            y1 += my
            x2 -= mx
            y2 -= my

        roi = hsv_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        if self.night_mode:
            red_low1 = self.night_red_low1
            red_high1 = self.night_red_high1
            red_low2 = self.night_red_low2
            red_high2 = self.night_red_high2
        else:
            red_low1 = self.red_low1
            red_high1 = self.red_high1
            red_low2 = self.red_low2
            red_high2 = self.red_high2

        mask1 = cv2.inRange(roi, red_low1, red_high1)
        mask2 = cv2.inRange(roi, red_low2, red_high2)
        mask = cv2.bitwise_or(mask1, mask2)
        area = float(mask.shape[0] * mask.shape[1])
        if area <= 0.0:
            return 0.0
        red_ratio = float(np.count_nonzero(mask)) / area
        if red_ratio <= 0.0:
            return 0.0

        v = roi[:, :, 2]
        lit_vals = v[mask > 0]
        if lit_vals.size == 0:
            value_term = 0.0
        else:
            value_term = float(np.mean(lit_vals)) / 255.0

        # Night mode: suppress passive red reflectors that occupy too much area
        # and are not bright enough to be active lamps.
        if self.night_mode:
            if red_ratio > self.night_max_red_ratio:
                return 0.0
            if lit_vals.size == 0 or float(np.mean(lit_vals)) < self.night_min_lit_value_mean:
                return 0.0

        prompt_boost = 1.0
        low_prompt = str(prompt).lower()
        if "indicator" in low_prompt:
            prompt_boost = 1.10
        elif "brake" in low_prompt or "lit" in low_prompt:
            prompt_boost = 1.05

        # Confidence should softly gate color score, not dominate it.
        conf = max(0.0, min(float(det_conf), 1.0))
        conf_term = self.state_detection_conf_floor + (1.0 - self.state_detection_conf_floor) * (
            conf ** self.state_detection_conf_gamma
        )
        color_term = (self.state_red_ratio_weight * red_ratio) + (self.state_value_weight * value_term)
        raw = conf_term * color_term * prompt_boost
        return max(0.0, min(raw, 1.0))

    def _classify_state_from_scores(self, left_score: float, right_score: float) -> Tuple[str, float]:
        # Backward-compatible wrapper (legacy callsites).
        return self._classify_state_from_side_scores(
            left_detected=True,
            right_detected=True,
            left_score=left_score,
            right_score=right_score,
        )

    def _force_geo_side(self, box: List[float], vehicle_bbox: List[float]) -> str:
        """Always return left/right from geometry (no center unknown state)."""
        vx1, _, vx2, _ = [float(v) for v in vehicle_bbox]
        vcx = 0.5 * (vx1 + vx2)
        cx = 0.5 * (float(box[0]) + float(box[2]))
        return "left" if cx <= vcx else "right"

    def _collapse_vehicle_taillights(
        self,
        vehicle_dets: List[TailLightDetection],
        vehicle_bbox: List[float],
    ) -> Tuple[List[TailLightDetection], bool, bool, float, float]:
        """
        Reduce raw prompt detections to at most one detection per side.

        We keep the highest-activation detection per side so downstream JSON/debug
        and vehicle-state logic operate on simplified left/right taillights.
        """
        by_side = {"left": [], "right": []}
        for det in vehicle_dets:
            # Use geometry for final side assignment so prompt wording does not
            # create contradictory left/right duplicates for the same light.
            side = self._force_geo_side(getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0]), vehicle_bbox)
            det.side = side
            by_side[side].append(det)

        collapsed: List[TailLightDetection] = []
        left_detected = len(by_side["left"]) > 0
        right_detected = len(by_side["right"]) > 0
        left_score = 0.0
        right_score = 0.0

        if left_detected:
            best_left = max(
                by_side["left"],
                key=lambda d: (float(getattr(d, "activation_score", 0.0)), float(getattr(d, "confidence", 0.0))),
            )
            left_score = float(getattr(best_left, "activation_score", 0.0))
            best_left.status = self._light_status_from_activation(left_score)
            collapsed.append(best_left)

        if right_detected:
            best_right = max(
                by_side["right"],
                key=lambda d: (float(getattr(d, "activation_score", 0.0)), float(getattr(d, "confidence", 0.0))),
            )
            right_score = float(getattr(best_right, "activation_score", 0.0))
            best_right.status = self._light_status_from_activation(right_score)
            collapsed.append(best_right)

        return collapsed, left_detected, right_detected, left_score, right_score

    def _classify_state_from_side_scores(
        self,
        left_detected: bool,
        right_detected: bool,
        left_score: float,
        right_score: float,
    ) -> Tuple[str, float]:
        """
        Vehicle brake-light state rules:
          - no detections: off
          - one side detected and on: on (assume both are on)
          - both sides detected, one on: left/right indicator
          - both detected and both on: on
          - both detected and both off: off
        """
        thr = self._lit_threshold()
        left_on = left_detected and (left_score >= thr)
        right_on = right_detected and (right_score >= thr)

        if not left_detected and not right_detected:
            return "off", 0.0

        if left_detected and right_detected:
            if left_on and right_on:
                return "on", float(min(left_score, right_score))
            if left_on and not right_on:
                return "left indicator", float(left_score)
            if right_on and not left_on:
                return "right indicator", float(right_score)
            return "off", float(max(left_score, right_score))

        # Only one side detected.
        detected_score = float(left_score if left_detected else right_score)
        detected_on = bool(left_on if left_detected else right_on)
        if detected_on:
            return "on", detected_score
        return "off", detected_score

    def _light_status_from_activation(self, activation_score: float) -> str:
        return "on" if float(activation_score) >= self._lit_threshold() else "off"

    def _set_vehicle_state(self, vehicle, state: str, score: float) -> None:
        final_state = state if state in _VALID_STATES else "off"
        setattr(vehicle, "brake_light_state", final_state)
        setattr(vehicle, "brake_light_confidence", round(float(score), 4))

    def _side_status(self, detected: bool, score: float) -> str:
        if not detected:
            return "not_detected"
        return "on" if float(score) >= self._lit_threshold() else "off"

    def _set_vehicle_side_states(
        self,
        vehicle,
        left_status: str,
        right_status: str,
        left_score: float = 0.0,
        right_score: float = 0.0,
    ) -> None:
        left = str(left_status).strip().lower()
        right = str(right_status).strip().lower()
        if left not in _VALID_SIDE_STATES:
            left = "not_detected"
        if right not in _VALID_SIDE_STATES:
            right = "not_detected"
        setattr(vehicle, "brake_light_left_status", left)
        setattr(vehicle, "brake_light_right_status", right)
        setattr(vehicle, "brake_light_left_score", round(float(left_score), 4))
        setattr(vehicle, "brake_light_right_score", round(float(right_score), 4))

    def detect(self, frame_bgr: np.ndarray, vehicles: List[object]) -> List[TailLightDetection]:
        if len(vehicles) == 0:
            return []

        # Always define a state field for each vehicle.
        for vehicle in vehicles:
            if getattr(vehicle, "brake_light_state", None) not in _VALID_STATES:
                self._set_vehicle_state(vehicle, "off", 0.0)
            self._set_vehicle_side_states(vehicle, "not_detected", "not_detected", 0.0, 0.0)

        if not self.is_active():
            return []

        h, w = frame_bgr.shape[:2]
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        final: List[TailLightDetection] = []

        for vehicle in vehicles:
            vbox_raw = [float(v) for v in getattr(vehicle, "bbox", [0.0, 0.0, 0.0, 0.0])]
            vbox_clamped = _clamp_bbox_xyxy(vbox_raw, w, h)
            if vbox_clamped is None:
                self._set_vehicle_state(vehicle, "off", 0.0)
                continue

            vx1, vy1, vx2, vy2 = vbox_clamped
            if (vx2 - vx1) < self.min_vehicle_crop_side_px or (vy2 - vy1) < self.min_vehicle_crop_side_px:
                self._set_vehicle_state(vehicle, "off", 0.0)
                continue

            crop = frame_bgr[vy1:vy2, vx1:vx2]
            if crop.size == 0:
                self._set_vehicle_state(vehicle, "off", 0.0)
                continue

            try:
                outputs = self.predictor(crop)
            except Exception:
                self._set_vehicle_state(vehicle, "off", 0.0)
                continue

            instances = outputs.get("instances", None)
            if instances is None:
                self._set_vehicle_state(vehicle, "off", 0.0)
                continue

            instances = instances.to("cpu")
            vehicle_dets: List[TailLightDetection] = []
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                pred_classes = instances.pred_classes.numpy()

                for box_arr, score, cls_id in zip(boxes, scores, pred_classes):
                    score_f = float(score)
                    cls_idx = int(cls_id)
                    if score_f < self.confidence:
                        continue
                    if cls_idx < 0 or cls_idx >= len(self.prompts):
                        continue

                    x1c, y1c, x2c, y2c = [float(v) for v in box_arr.tolist()]
                    x1g, y1g = x1c + vx1, y1c + vy1
                    x2g, y2g = x2c + vx1, y2c + vy1
                    box_global = [x1g, y1g, x2g, y2g]
                    box_clamped = _clamp_bbox_xyxy(box_global, w, h)
                    if box_clamped is None:
                        continue

                    bw = float(box_clamped[2] - box_clamped[0])
                    bh = float(box_clamped[3] - box_clamped[1])
                    if bw < self.min_width_px or bh < self.min_height_px:
                        continue

                    prompt = self.prompts[cls_idx]
                    side = self._prompt_side(prompt)
                    if side == "unknown":
                        side = self._geo_side(box_global, vbox_raw)

                    activation = self._red_activation_score(hsv_frame, box_global, score_f, prompt)
                    vehicle_dets.append(
                        TailLightDetection(
                            label="taillight",
                            bbox=[float(box_clamped[0]), float(box_clamped[1]), float(box_clamped[2]), float(box_clamped[3])],
                            confidence=round(score_f, 4),
                            vehicle_track_id=getattr(vehicle, "track_id", None),
                            vehicle_class=str(getattr(vehicle, "label", "car")),
                            side=side,
                            prompt=prompt,
                            activation_score=round(float(activation), 4),
                            status=self._light_status_from_activation(activation),
                        )
                    )

            # Keep top raw detections per vehicle, then simplify to left/right.
            vehicle_dets = sorted(vehicle_dets, key=lambda d: float(d.confidence), reverse=True)[: self.max_per_vehicle]
            collapsed_dets, left_detected, right_detected, left_score, right_score = self._collapse_vehicle_taillights(
                vehicle_dets,
                vbox_raw,
            )
            state, conf = self._classify_state_from_side_scores(
                left_detected=left_detected,
                right_detected=right_detected,
                left_score=left_score,
                right_score=right_score,
            )
            self._set_vehicle_state(vehicle, state, conf)
            self._set_vehicle_side_states(
                vehicle,
                self._side_status(left_detected, left_score),
                self._side_status(right_detected, right_score),
                left_score,
                right_score,
            )
            final.extend(collapsed_dets)

        return final
