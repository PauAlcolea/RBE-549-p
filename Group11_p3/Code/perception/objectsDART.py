"""
perception/objectsDART.py
=========================
Generic DART-based detector for non-COCO objects.

The detector returns object-style records compatible with DepthEstimator.lift_to_3d
and export flow. Labels and export buckets are config-driven so new classes can
be added without creating one module per object type.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class NonCocoObject:
	label: str = "non_coco_object"
	bbox: List[float] = field(default_factory=list)
	confidence: float = 0.0
	depth_m: float = 0.0
	position_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
	export_bucket: str = "non_coco_objects"


def _compute_iou(a: List[float], b: List[float]) -> float:
	"""Compute IoU between two [x1, y1, x2, y2] boxes."""
	ix1 = max(a[0], b[0])
	iy1 = max(a[1], b[1])
	ix2 = min(a[2], b[2])
	iy2 = min(a[3], b[3])
	inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
	if inter == 0:
		return 0.0
	area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
	area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
	den = area_a + area_b - inter
	if den <= 0:
		return 0.0
	return inter / den


class NonCocoDartDetector:
	"""Wraps DART/SAM3 for prompt-based non-COCO object detection."""

	def __init__(self, cfg: dict, device: str = "cuda"):
		self.cfg = cfg
		self.device = self._normalize_device(device)

		self.code_dir = Path(__file__).resolve().parents[1]
		self.dart_dir = Path(__file__).resolve().parent / "DART"

		# New key for extensibility, with backward-compatible fallback.
		self.detector_cfg = (
			cfg.get("perception", {}).get("non_coco_dart")
			or cfg.get("perception", {}).get("cones", {})
		)

		self.enabled = bool(self.detector_cfg.get("enabled", False))
		self.confidence = float(self.detector_cfg.get("confidence", 0.35))
		self.nms = float(self.detector_cfg.get("nms", 0.40))
		self.imgsz = int(self.detector_cfg.get("imgsz", 1008))
		self.compile_mode = str(self.detector_cfg.get("compile_mode", "max-autotune"))
		self.runtime_mode = str(self.detector_cfg.get("runtime", "auto")).lower()
		self.use_masks = bool(self.detector_cfg.get("use_masks", True))
		self.trt_max_classes = int(self.detector_cfg.get("trt_max_classes", 4))
		self.min_width_px = float(self.detector_cfg.get("min_width_px", 8.0))
		self.min_height_px = float(self.detector_cfg.get("min_height_px", 8.0))
		self.max_instances = int(self.detector_cfg.get("max_instances", 30))
		self.default_export_bucket = str(
			self.detector_cfg.get("default_export_bucket", "non_coco_objects")
		)

		self.class_specs = self._parse_class_specs(self.detector_cfg)
		self.prompt_to_spec = {
			spec["prompt"].lower(): spec for spec in self.class_specs
		}
		self.prompts = [spec["prompt"] for spec in self.class_specs]

		requested_weights = self.detector_cfg.get("checkpoint", cfg.get("weights", {}).get("lanes"))
		self.checkpoint_path = self._resolve_path(requested_weights)
		self.trt_backbone_path = self._resolve_path(
			self.detector_cfg.get("trt_backbone", "perception/DART/hf_backbone_fp16.engine")
		)
		self.trt_enc_dec_path = self._resolve_path(
			self.detector_cfg.get("trt_enc_dec", "perception/DART/enc_dec_fp16.engine")
		)
		self.text_cache_path = self._resolve_path(self.detector_cfg.get("text_cache", None))

		self.predictor = None
		self.active_runtime = "disabled"
		if self.enabled and len(self.prompts) > 0:
			self._load_predictor()

	@staticmethod
	def _slugify(name: str) -> str:
		clean = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
		return clean.strip("_") or "non_coco_object"

	def _default_bucket(self, label: str) -> str:
		return f"{label}s"

	def _parse_class_specs(self, detector_cfg: dict) -> List[Dict[str, Any]]:
		"""Build class mapping from config. Supports strings and dict entries."""
		raw_classes = detector_cfg.get("classes", ["traffic cone"])
		specs: List[Dict[str, Any]] = []

		for item in raw_classes:
			if isinstance(item, dict):
				prompt = str(item.get("prompt", "")).strip()
				if not prompt:
					continue
				label = str(item.get("label", self._slugify(prompt))).strip()
				export_bucket = str(
					item.get("export_bucket", self._default_bucket(label))
				).strip()
				raw_excludes = item.get("exclude_labels", item.get("negative_prompts", []))
				exclude_labels: List[str] = []
				if isinstance(raw_excludes, (list, tuple)):
					exclude_labels = [str(v).strip() for v in raw_excludes if str(v).strip()]
				elif raw_excludes is not None:
					exclude_labels = [str(raw_excludes).strip()]
				exclude_labels = [self._slugify(v) for v in exclude_labels]
				specs.append(
					{
						"prompt": prompt,
						"label": label,
						"export_bucket": export_bucket or self.default_export_bucket,
						"exclude_labels": exclude_labels,
					}
				)
				continue

			prompt = str(item).strip()
			if not prompt:
				continue

			# Backward-compatible default: old cones config maps all prompts to traffic_cone.
			if "non_coco_dart" not in self.cfg.get("perception", {}):
				label = "traffic_cone"
				export_bucket = "traffic_cones"
			else:
				label = self._slugify(prompt)
				export_bucket = self._default_bucket(label)

			specs.append(
				{
					"prompt": prompt,
					"label": label,
					"export_bucket": export_bucket,
					"exclude_labels": [],
				}
			)

		return specs

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
					"non_coco_dart.runtime=trt requires a valid trt_backbone engine"
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
			"[non-coco-dart] runtime="
			f"{self.active_runtime}, device={self.device}, "
			f"trt_backbone={use_trt_backbone}, trt_enc_dec={use_trt_enc_dec}, "
			f"compile_mode={compile_mode}, use_masks={self.use_masks}"
		)
		return use_trt_backbone, use_trt_enc_dec, compile_mode

	def _load_predictor(self) -> None:
		if self.imgsz % 14 != 0:
			raise ValueError(f"non_coco_dart.imgsz must be divisible by 14, got {self.imgsz}")

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
		predictor.set_classes(self.prompts, text_cache=self.text_cache_path)
		self.predictor = predictor

	def detect(self, frame_bgr: np.ndarray) -> List[NonCocoObject]:
		"""Run DART non-COCO detection on a single BGR frame."""
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

		objects = self._results_to_objects(raw)
		if self.max_instances > 0:
			objects = objects[: self.max_instances]
		return objects

	def _spec_for_prediction(self, i: int, class_names: list, class_ids) -> Dict[str, Any]:
		prompt_name = ""
		if i < len(class_names):
			prompt_name = str(class_names[i]).strip().lower()
		if prompt_name in self.prompt_to_spec:
			return self.prompt_to_spec[prompt_name]

		if class_ids is not None and len(class_ids) > i:
			cls_idx = int(class_ids[i].item())
			if 0 <= cls_idx < len(self.class_specs):
				return self.class_specs[cls_idx]

		return {
			"prompt": prompt_name or "unknown",
			"label": "non_coco_object",
			"export_bucket": self.default_export_bucket,
			"exclude_labels": [],
		}

	def _apply_negative_label_suppression(self, objects: List[NonCocoObject], iou_thr: float) -> List[NonCocoObject]:
		"""Drop detections that overlap labels listed in each class spec's exclude_labels."""
		if not objects:
			return objects

		label_excludes = {
			str(spec.get("label", "")).strip(): set(spec.get("exclude_labels", []) or [])
			for spec in self.class_specs
		}
		if not any(label_excludes.values()):
			return objects

		kept: List[NonCocoObject] = []
		for obj in objects:
			excludes = label_excludes.get(obj.label, set())
			if not excludes:
				kept.append(obj)
				continue

			suppressed = False
			for other in objects:
				if other is obj:
					continue
				if other.label not in excludes:
					continue
				if _compute_iou(obj.bbox, other.bbox) >= iou_thr:
					suppressed = True
					break

			if not suppressed:
				kept.append(obj)

		return kept

	def _results_to_objects(self, results: dict) -> List[NonCocoObject]:
		boxes = results.get("boxes")
		scores = results.get("scores")
		class_names = results.get("class_names", [])
		class_ids = results.get("class_ids")

		if boxes is None or scores is None or len(scores) == 0:
			return []

		negative_overlap_iou = float(self.detector_cfg.get("negative_overlap_iou", 0.35))

		objects: List[NonCocoObject] = []
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

			spec = self._spec_for_prediction(i, class_names, class_ids)
			objects.append(
				NonCocoObject(
					label=spec["label"],
					bbox=[x1, y1, x2, y2],
					confidence=round(score, 4),
					export_bucket=spec["export_bucket"],
				)
			)

		objects.sort(key=lambda d: d.confidence, reverse=True)
		objects = self._apply_negative_label_suppression(objects, iou_thr=negative_overlap_iou)
		return objects


# Backward-compatible aliases while the rest of the pipeline transitions.
Cone = NonCocoObject


class ConeDetector(NonCocoDartDetector):
	pass
