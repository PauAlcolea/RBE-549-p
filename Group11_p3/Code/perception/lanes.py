"""
perception/lanes.py
===================
Lane detection backed by DART/SAM3.

This module is designed to be called from run_perception.py on a per-frame basis
instead of invoking the DART CLI scripts.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class Lane:
    """A single detected lane line."""

    points: List[Tuple[float, float]]
    color: str = "white"
    
    confidence: float = 1.0


class LaneDetector:
    """
    Wraps DART for lane detection.

    Usage
    -----
    detector = LaneDetector(cfg, device="cuda")
    output = detector.detect(frame_bgr)
    lanes = output["lanes"]
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = self._normalize_device(device)

        self.code_dir = Path(__file__).resolve().parents[1]
        self.dart_dir = Path(__file__).resolve().parent / "DART"
        self.lanes_cfg = cfg.get("perception", {}).get("lanes", {})

        self.model_name = self.lanes_cfg.get("model", "DART")
        self.max_lanes = int(self.lanes_cfg.get("max_lanes", 8))
        self.lane_classes = list(
            self.lanes_cfg.get(
                "classes", ["yellow lane line painted on road", "white lane line painted on road", "crosswalk marking on road"]
            )
        )
        self.classes = list(self.lane_classes)

        self.confidence = float(self.lanes_cfg.get("confidence", 0.35))
        self.nms = float(self.lanes_cfg.get("nms", 0.4))
        self.imgsz = int(self.lanes_cfg.get("imgsz", 1008))
        self.compile_mode = self.lanes_cfg.get("compile_mode", "max-autotune")
        self.runtime_mode = str(self.lanes_cfg.get("runtime", "auto")).lower()
        self.use_masks = bool(self.lanes_cfg.get("use_masks", True))
        self.return_raw = bool(self.lanes_cfg.get("return_raw", True))
        self.trt_max_classes = int(self.lanes_cfg.get("trt_max_classes", 4))

        self.polyline_points = int(self.lanes_cfg.get("polyline_points", 20))
        self.color_samples = int(self.lanes_cfg.get("color_samples", 12))
        self.sample_patch = int(self.lanes_cfg.get("color_patch", 3))
        self.white_vote_ratio = float(self.lanes_cfg.get("white_vote_ratio", 0.5))
        self.dash_gap_px = float(self.lanes_cfg.get("dash_gap_px", 60.0))
        self.min_length_px = float(self.lanes_cfg.get("min_length_px", 50.0))

        self.white_low = np.array(
            self.lanes_cfg.get("hsv_white_low", [0, 0, 160]), dtype=np.uint8
        )
        self.white_high = np.array(
            self.lanes_cfg.get("hsv_white_high", [179, 80, 255]), dtype=np.uint8
        )

        requested_weights = cfg.get("weights", {}).get("lanes")
        self.checkpoint_path = self._resolve_path(
            self.lanes_cfg.get("checkpoint", requested_weights)
        )
        self.trt_backbone_path = self._resolve_path(
            self.lanes_cfg.get("trt_backbone", "perception/DART/hf_backbone_fp16.engine")
        )
        self.trt_enc_dec_path = self._resolve_path(
            self.lanes_cfg.get("trt_enc_dec", "perception/DART/enc_dec_fp16.engine")
        )
        self.text_cache_path = self._resolve_path(self.lanes_cfg.get("text_cache", None))

        self.predictor = None
        self.model = None
        self.active_runtime = "uninitialized"
        self.model = self._load_model(self.checkpoint_path)

    def _normalize_device(self, device: str) -> str:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_path(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if value == "" or value.upper() == "TBD":
            return None

        p = Path(value)
        if p.is_absolute():
            return str(p) if p.exists() else None

        candidates = [
            self.code_dir / p,
            self.dart_dir / p,
            Path.cwd() / p,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(self.code_dir / p)

    def _load_model(self, weights_path: Optional[str] = None):
        if self.imgsz % 14 != 0:
            raise ValueError(f"lanes.imgsz must be divisible by 14, got {self.imgsz}")

        if str(self.dart_dir) not in sys.path:
            sys.path.insert(0, str(self.dart_dir))

        from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
        from sam3.model_builder import (
            build_pruned_sam3_image_model,
            build_sam3_image_model,
            load_pruned_config,
        )

        use_trt_backbone, use_trt_enc_dec, compile_mode = self._select_runtime_mode()
        detection_only = not self.use_masks or use_trt_enc_dec

        model = None
        checkpoint_path = weights_path

        if checkpoint_path is not None:
            pruned_config = load_pruned_config(checkpoint_path)
            if pruned_config is not None:
                model = build_pruned_sam3_image_model(
                    checkpoint_path=checkpoint_path,
                    pruning_config=pruned_config,
                    device=self.device,
                    eval_mode=True,
                )
                if model.transformer.decoder.presence_token is not None:
                    model.transformer.decoder.presence_token = None

        if model is None:
            model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint_path,
                eval_mode=True,
            )

        if self.imgsz != 1008:
            pos_enc = model.backbone.vision_backbone.position_encoding
            pos_enc.precompute_for_resolution(self.imgsz)

        predictor = Sam3MultiClassPredictorFast(
            model,
            device=self.device,
            resolution=self.imgsz,
            use_fp16=(self.device == "cuda"),
            detection_only=detection_only,
            trt_engine_path=self.trt_backbone_path if use_trt_backbone else None,
            compile_mode=compile_mode,
            trt_enc_dec_engine_path=self.trt_enc_dec_path if use_trt_enc_dec else None,
            trt_max_classes=self.trt_max_classes,
        )

        predictor.set_classes(self.classes, text_cache=self.text_cache_path)
        if compile_mode is not None:
            self._warmup_predictor(predictor)

        self.predictor = predictor
        return model

    def _select_runtime_mode(self) -> Tuple[bool, bool, Optional[str]]:
        has_trt_backbone = bool(self.trt_backbone_path and Path(self.trt_backbone_path).exists())
        has_trt_enc_dec = bool(self.trt_enc_dec_path and Path(self.trt_enc_dec_path).exists())

        mode = self.runtime_mode
        if mode not in {"auto", "trt", "compile", "pytorch"}:
            mode = "auto"

        use_trt_backbone = False
        use_trt_enc_dec = False
        compile_mode = None

        if mode == "trt":
            if not has_trt_backbone:
                raise FileNotFoundError(
                    "lanes.runtime=trt requires a valid lanes.trt_backbone engine"
                )
            use_trt_backbone = True
            use_trt_enc_dec = has_trt_enc_dec and not self.use_masks
            self.active_runtime = "trt"
        elif mode == "compile":
            compile_mode = self.compile_mode
            self.active_runtime = "compile"
        elif mode == "pytorch":
            self.active_runtime = "pytorch"
        else:
            if has_trt_backbone:
                use_trt_backbone = True
                use_trt_enc_dec = has_trt_enc_dec and not self.use_masks
                self.active_runtime = "trt"
            elif self.device == "cuda":
                compile_mode = self.compile_mode
                self.active_runtime = "compile"
            else:
                self.active_runtime = "pytorch"

        if use_trt_enc_dec and self.use_masks:
            use_trt_enc_dec = False

        print(
            "[lanes] runtime="
            f"{self.active_runtime}, device={self.device}, "
            f"trt_backbone={use_trt_backbone}, trt_enc_dec={use_trt_enc_dec}, "
            f"compile_mode={compile_mode}, use_masks={self.use_masks}"
        )
        return use_trt_backbone, use_trt_enc_dec, compile_mode

    def _warmup_predictor(self, predictor) -> None:
        print("[lanes] warming up compiled lane predictor...")
        start = time.perf_counter()
        dummy_img = Image.new("RGB", (self.imgsz, self.imgsz))
        with torch.inference_mode():
            for _ in range(3):
                state = predictor.set_image(dummy_img)
                predictor.predict(
                    state,
                    confidence_threshold=self.confidence,
                    nms_threshold=self.nms,
                )
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"[lanes] warmup done in {time.perf_counter() - start:.1f}s")

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Run lane detection on one BGR frame.

        Returns a dict containing normalized lane output and optional raw model output.
        """
        if self.predictor is None:
            return {"lanes": [], "raw": {}}

        # Sam3 predictor tracks original size correctly for PIL inputs.
        # Passing raw HWC numpy arrays can misreport width/height in this build.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        with torch.inference_mode():
            state = self.predictor.set_image(frame_pil)
            raw_results = self.predictor.predict(
                state,
                confidence_threshold=self.confidence,
                nms_threshold=self.nms,
            )

        lanes = self._results_to_lanes(frame_bgr, raw_results)
        payload: Dict[str, Any] = {
            "lanes": [asdict(l) for l in lanes],
            "runtime": self.active_runtime,
        }
        if self.return_raw:
            payload["raw"] = raw_results
        return payload

    def _results_to_lanes(self, frame_bgr: np.ndarray, results: Dict[str, Any]) -> List[Lane]:
        scores = results.get("scores")
        if scores is None or len(scores) == 0:
            return []

        masks = results.get("masks")
        boxes = results.get("boxes")
        class_names = results.get("class_names", [])
        h = int(frame_bgr.shape[0])

        lanes: List[Lane] = []
        for i in range(len(scores)):
            score = float(scores[i].item())
            if score < self.confidence:
                continue

            cls_name = class_names[i] if i < len(class_names) else "lane line"
            cls_name_lower = cls_name.lower()
            is_lane = ("lane" in cls_name_lower)
            if not is_lane:
                continue

            points: List[Tuple[float, float]] = []
            if masks is not None and len(masks) > i:
                mask = masks[i].detach().cpu().numpy().astype(np.uint8)
                points = self._mask_to_polyline(mask)

            if not points and boxes is not None and len(boxes) > i:
                points = self._box_to_polyline(boxes[i].detach().cpu().numpy())

            if len(points) < 2:
                continue

            # Filter by polyline length
            if self._polyline_length(points) < self.min_length_px:
                continue

            cls_name = class_names[i] if i < len(class_names) else "lane line"
            cls_name_lower = cls_name.lower()
            if "yellow" in cls_name_lower or "non-white" in cls_name_lower:
                color = "yellow"
            elif "white" in cls_name_lower:
                color = "white"
            else:
                color = self._classify_color(frame_bgr, points)
            # style = self._classify_style(points, h)

            lanes.append(
                Lane(
                    points=points,
                    color=color,
                    confidence=score,
                )
            )

        lanes.sort(key=lambda lane: lane.confidence, reverse=True)
        return lanes

    def _polyline_length(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total Euclidean length of a polyline."""
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total

    def _mask_to_polyline(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return []

        y_min = int(ys.min())
        y_max = int(ys.max())
        if y_max <= y_min:
            return []

        samples = max(8, self.polyline_points)
        y_samples = np.linspace(y_min, y_max, num=samples).astype(np.int32)

        points: List[Tuple[float, float]] = []
        for y in y_samples:
            row_x = np.where(mask[y] > 0)[0]
            if row_x.size == 0:
                continue
            x_center = float(row_x.mean())
            points.append((x_center, float(y)))

        return points

    def _box_to_polyline(self, box_xyxy: Sequence[float]) -> List[Tuple[float, float]]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        if x2 <= x1 or y2 <= y1:
            return []

        x_center = 0.5 * (x1 + x2)
        y_samples = np.linspace(y1, y2, num=max(8, self.polyline_points // 2))
        return [(x_center, float(y)) for y in y_samples]

    def _classify_color(self, frame_bgr: np.ndarray, points: List[Tuple[float, float]]) -> str:
        if len(points) == 0:
            return "white"

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        idx = np.linspace(0, len(points) - 1, num=min(self.color_samples, len(points))).astype(int)
        white_votes = 0
        valid_samples = 0

        r = max(1, self.sample_patch)
        for i in idx:
            x, y = points[i]
            cx = int(np.clip(round(x), 0, w - 1))
            cy = int(np.clip(round(y), 0, h - 1))

            y1 = max(0, cy - r)
            y2 = min(h, cy + r + 1)
            x1 = max(0, cx - r)
            x2 = min(w, cx + r + 1)
            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            valid_samples += 1

            mean_hsv = patch.reshape(-1, 3).mean(axis=0)
            mean_hsv_u8 = np.array(mean_hsv, dtype=np.uint8)

            is_white = np.all(mean_hsv_u8 >= self.white_low) and np.all(
                mean_hsv_u8 <= self.white_high
            )

            white_votes += int(is_white)

        if valid_samples == 0:
            return "white"

        # White-first logic: anything not sufficiently white is treated as yellow.
        white_fraction = white_votes / float(valid_samples)
        if white_fraction >= self.white_vote_ratio:
            return "white"
        return "yellow"

    def _classify_style(
        self,
        points: List[Tuple[float, float]],
        img_height: int,
    ) -> str:
        if len(points) < 3:
            return "solid"

        ys = sorted(set(float(p[1]) for p in points))
        if len(ys) < 3:
            return "solid"

        gaps = np.diff(np.array(ys, dtype=np.float32))
        max_gap = float(gaps.max()) if gaps.size else 0.0

        adaptive_gap = max(self.dash_gap_px, 0.06 * float(img_height))
        if max_gap > adaptive_gap:
            return "dashed"
        return "solid"
