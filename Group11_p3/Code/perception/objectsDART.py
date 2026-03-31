"""
perception/objectsDART.py
=========================
DART-based traffic cone detection for non-COCO objects.

The detector returns object-style records compatible with DepthEstimator.lift_to_3d
and JSON export flow used by run_perception.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class Cone:
	label: str = "traffic_cone"
	bbox: List[float] = field(default_factory=list)
	confidence: float = 0.0
	depth_m: float = 0.0
	position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class ConeDetector:
	"""Wraps DART/SAM3 for prompt-based traffic-cone detection."""

	def __init__(self, cfg: dict, device: str = "cuda"):
		self.cfg = cfg
		self.device = self._normalize_device(device)

		self.code_dir = Path(__file__).resolve().parents[1]
		self.dart_dir = Path(__file__).resolve().parent / "DART"
		self.cones_cfg = cfg.get("perception", {}).get("cones", {})

		self.enabled = bool(self.cones_cfg.get("enabled", False))
		self.classes = list(self.cones_cfg.get("classes", ["traffic cone"]))
		self.confidence = float(self.cones_cfg.get("confidence", 0.35))
		self.nms = float(self.cones_cfg.get("nms", 0.40))
		self.imgsz = int(self.cones_cfg.get("imgsz", 1008))
		self.compile_mode = str(self.cones_cfg.get("compile_mode", "max-autotune"))
		self.runtime_mode = str(self.cones_cfg.get("runtime", "auto")).lower()
		self.use_masks = bool(self.cones_cfg.get("use_masks", True))
		self.trt_max_classes = int(self.cones_cfg.get("trt_max_classes", 4))
		self.min_width_px = float(self.cones_cfg.get("min_width_px", 8.0))
		self.min_height_px = float(self.cones_cfg.get("min_height_px", 8.0))
		self.max_instances = int(self.cones_cfg.get("max_instances", 30))

		requested_weights = self.cones_cfg.get("checkpoint", cfg.get("weights", {}).get("lanes"))
		self.checkpoint_path = self._resolve_path(requested_weights)
		self.trt_backbone_path = self._resolve_path(
			self.cones_cfg.get("trt_backbone", "perception/DART/hf_backbone_fp16.engine")
		)
		self.trt_enc_dec_path = self._resolve_path(
			self.cones_cfg.get("trt_enc_dec", "perception/DART/enc_dec_fp16.engine")
		)
		self.text_cache_path = self._resolve_path(self.cones_cfg.get("text_cache", None))

		self.predictor = None
		self.active_runtime = "disabled"
		if self.enabled:
			self._load_predictor()

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
					"cones.runtime=trt requires a valid cones.trt_backbone engine"
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
			"[cones] runtime="
			f"{self.active_runtime}, device={self.device}, "
			f"trt_backbone={use_trt_backbone}, trt_enc_dec={use_trt_enc_dec}, "
			f"compile_mode={compile_mode}, use_masks={self.use_masks}"
		)
		return use_trt_backbone, use_trt_enc_dec, compile_mode

	def _load_predictor(self) -> None:
		if self.imgsz % 14 != 0:
			raise ValueError(f"cones.imgsz must be divisible by 14, got {self.imgsz}")

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
		if self.checkpoint_path is not None:
			pruned_config = load_pruned_config(self.checkpoint_path)
			if pruned_config is not None:
				model = build_pruned_sam3_image_model(
					checkpoint_path=self.checkpoint_path,
					pruning_config=pruned_config,
					device=self.device,
					eval_mode=True,
				)
				if model.transformer.decoder.presence_token is not None:
					model.transformer.decoder.presence_token = None

		if model is None:
			model = build_sam3_image_model(
				device=self.device,
				checkpoint_path=self.checkpoint_path,
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
		self.predictor = predictor

	def detect(self, frame_bgr: np.ndarray) -> List[Cone]:
		"""Run DART cone detection on a single BGR frame."""
		if not self.enabled or self.predictor is None:
			return []

		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		frame_pil = Image.fromarray(frame_rgb)

		with torch.inference_mode():
			state = self.predictor.set_image(frame_pil)
			raw = self.predictor.predict(
				state,
				confidence_threshold=self.confidence,
				nms_threshold=self.nms,
			)

		cones = self._results_to_cones(raw)
		if self.max_instances > 0:
			cones = cones[: self.max_instances]
		return cones

	def _results_to_cones(self, results: dict) -> List[Cone]:
		boxes = results.get("boxes")
		scores = results.get("scores")

		if boxes is None or scores is None or len(scores) == 0:
			return []

		cones: List[Cone] = []
		for i in range(len(scores)):
			score = float(scores[i].item())
			if score < self.confidence:
				continue

			box = boxes[i].detach().cpu().tolist()
			x1, y1, x2, y2 = [float(v) for v in box]
			if x2 <= x1 or y2 <= y1:
				continue

			if (x2 - x1) < self.min_width_px or (y2 - y1) < self.min_height_px:
				continue

			cones.append(
				Cone(
					label="traffic_cone",
					bbox=[x1, y1, x2, y2],
					confidence=round(score, 4),
				)
			)

		cones.sort(key=lambda d: d.confidence, reverse=True)
		return cones
