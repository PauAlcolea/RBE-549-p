from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import sys
import torch
import numpy as np

# The forked 3D-BoundingBox repo still uses np.float internally.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


_DEFAULT_CLASS_MAP = {
    "car": "car",
    "sedan": "car",
    "hatchback": "car",
    "suv": "car",
    "truck": "truck",
    "pickuptruck": "truck",
    "pickup_truck": "truck",
    "bus": "truck",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
}


@dataclass
class OrientationEstimate:
    """
    One 3D box/orientation prediction aligned with an upstream 2D detection.
    """

    detection_index: int
    input_label: str
    model_class: str
    bbox: list[float]
    confidence: float
    alpha_rad: float
    theta_ray_rad: float
    yaw_rad: float
    dimensions_m: list[float]
    location_m: list[float]

    def to_dict(self) -> dict:
        return {
            "detection_index": self.detection_index,
            "input_label": self.input_label,
            "model_class": self.model_class,
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "alpha_rad": round(self.alpha_rad, 6),
            "theta_ray_rad": round(self.theta_ray_rad, 6),
            "yaw_rad": round(self.yaw_rad, 6),
            "dimensions_m": [round(v, 4) for v in self.dimensions_m],
            "location_m": [round(v, 4) for v in self.location_m],
        }


class OrientationEstimator:
    """
    Wrapper around the forked `BoundingBox_3D` repo that consumes our existing
    2D detections instead of re-running the repo's internal YOLO detector.
    """

    def __init__(self, cfg: dict, device: str = "cuda", strict: bool = False):
        self.cfg = cfg
        self.device = self._resolve_device(device)
        self.code_dir = Path(__file__).resolve().parents[1]
        self.submodule_dir = Path(__file__).resolve().parent / "BoundingBox_3D"
        self.weights_dir = self.submodule_dir / "weights"

        orientation_cfg = cfg.get("perception", {}).get("orientation") or {}
        self.bins = int(orientation_cfg.get("bins", 2))

        self.class_map = dict(_DEFAULT_CLASS_MAP)
        self.class_map.update(orientation_cfg.get("class_map") or {})

        self.proj_matrix = self._build_projection_matrix(cfg)
        self.weights_path = self._resolve_weights_path(cfg, orientation_cfg)

        self.model = None
        self.averages = None
        self.angle_bins = None
        self._DetectedObject = None
        self._ModelModule = None
        self._calc_location = None
        self.disabled_reason: Optional[str] = None

        try:
            self._load_runtime()
        except Exception as exc:
            self.disabled_reason = str(exc)
            if strict:
                raise RuntimeError(self.disabled_reason) from exc

    def is_ready(self) -> bool:
        return self.disabled_reason is None

    def ensure_ready(self) -> None:
        if not self.is_ready():
            raise RuntimeError(self.disabled_reason or "Orientation estimator is not ready.")

    def estimate(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Any],
    ) -> list[OrientationEstimate]:
        """
        Run the forked 3D box model on a frame plus the pipeline's 2D detections.

        Only detections whose labels can be mapped onto the repo's supported
        classes are processed. Others are skipped.
        """
        self.ensure_ready()
        assert self.model is not None
        assert self.averages is not None
        assert self.angle_bins is not None
        assert self._DetectedObject is not None
        assert self._calc_location is not None

        import torch

        candidates: list[tuple[int, Any, str, Any, list[tuple[int, int]]]] = []

        for idx, det in enumerate(detections):
            label = str(getattr(det, "label", "")).lower()
            model_class = self.class_map.get(label)
            if not model_class:
                continue
            if not self.averages.recognized_class(model_class):
                continue

            bbox = getattr(det, "bbox", None)
            box_2d = self._bbox_to_box_2d(bbox, frame_bgr.shape)
            if box_2d is None:
                continue

            try:
                detected_object = self._DetectedObject(
                    frame_bgr,
                    model_class,
                    box_2d,
                    self.proj_matrix,
                )
            except Exception:
                continue

            candidates.append((idx, det, model_class, detected_object, box_2d))

        if not candidates:
            return []

        batch = torch.stack([item[3].img for item in candidates], dim=0).float().to(self.device)

        with torch.no_grad():
            orient_batch, conf_batch, dim_batch = self.model(batch)

        orient_np = orient_batch.detach().cpu().numpy()
        conf_np = conf_batch.detach().cpu().numpy()
        dim_np = dim_batch.detach().cpu().numpy()

        estimates: list[OrientationEstimate] = []
        for row, (idx, det, model_class, detected_object, box_2d) in enumerate(candidates):
            dim = dim_np[row] + self.averages.get_item(model_class)

            bin_idx = int(np.argmax(conf_np[row]))
            orient = orient_np[row, bin_idx, :]
            alpha = float(np.arctan2(orient[1], orient[0]) + self.angle_bins[bin_idx] - np.pi)
            theta_ray = float(detected_object.theta_ray)
            yaw = float(alpha + theta_ray)

            location, _ = self._calc_location(
                dim,
                detected_object.proj_matrix,
                box_2d,
                alpha,
                theta_ray,
            )

            estimates.append(
                OrientationEstimate(
                    detection_index=idx,
                    input_label=str(getattr(det, "label", model_class)),
                    model_class=model_class,
                    bbox=[float(v) for v in getattr(det, "bbox", [])],
                    confidence=float(getattr(det, "confidence", 0.0)),
                    alpha_rad=alpha,
                    theta_ray_rad=theta_ray,
                    yaw_rad=yaw,
                    dimensions_m=[float(v) for v in dim.tolist()],
                    location_m=[float(v) for v in location],
                )
            )

        return estimates

    def annotate_detections(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Any],
    ) -> list[OrientationEstimate]:
        """
        Attach orientation fields onto the existing detection objects in-place.

        Added attributes:
          - `heading_rad`
          - `alpha_rad`
          - `theta_ray_rad`
          - `dimensions_3d`
          - `bbox_3d_location`
          - `orientation_model_class`
          - `orientation_source`
        """
        estimates = self.estimate(frame_bgr, detections)

        for estimate in estimates:
            det = detections[estimate.detection_index]
            det.heading_rad = estimate.yaw_rad
            det.alpha_rad = estimate.alpha_rad
            det.theta_ray_rad = estimate.theta_ray_rad
            det.dimensions_3d = estimate.dimensions_m
            det.bbox_3d_location = estimate.location_m
            det.orientation_model_class = estimate.model_class
            det.orientation_source = "BoundingBox_3D"

        return estimates

    def _load_runtime(self):
        if not self.submodule_dir.exists():
            raise FileNotFoundError(
                f"BoundingBox_3D submodule not found at {self.submodule_dir}"
            )
        if self.weights_path is None or not self.weights_path.exists():
            raise FileNotFoundError(
                f"No BoundingBox_3D checkpoint found. Expected a .pkl file in {self.weights_dir}"
            )

        if str(self.submodule_dir) not in sys.path:
            sys.path.insert(0, str(self.submodule_dir))

        from torchvision.models import vgg

        from library.Math import calc_location
        from torch_lib import ClassAverages, Model
        from torch_lib.Dataset import DetectedObject, generate_bins
        
        self._DetectedObject = DetectedObject
        self._ModelModule = Model
        self._calc_location = calc_location
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(self.bins)

        try:
            backbone = vgg.vgg19_bn(weights=None)
        except TypeError:
            backbone = vgg.vgg19_bn(pretrained=False)

        model = Model.Model(features=backbone.features, bins=self.bins)
        checkpoint = torch.load(str(self.weights_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

    def _resolve_device(self, device: str) -> str:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def _resolve_weights_path(self, cfg: dict, orientation_cfg: dict):
        configured = (
        cfg.get("weights", {}).get("orientation")
        or orientation_cfg.get("weights")
        or orientation_cfg.get("weights_path")
        )
        if configured:
            p = self._resolve_path(configured)
            if not p.exists():
                raise FileNotFoundError(f"Configured orientation weights not found: {p}")
            return p

        checkpoints = sorted(self.weights_dir.glob("*.pkl"))
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(
            f"No orientation weights (.pkl) found in {self.weights_dir}. "
            "Download the BoundingBox_3D checkpoint and place it there."
        )

    def _resolve_path(self, path_like: str) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (self.code_dir / path).resolve()

    def _build_projection_matrix(self, cfg: dict) -> np.ndarray:
        camera_cfg = cfg["blender"]["camera"]
        fx = float(camera_cfg["fx"])
        fy = float(camera_cfg["fy"])
        cx = float(camera_cfg["cx"])
        cy = float(camera_cfg["cy"])
        return np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def _bbox_to_box_2d(
        self,
        bbox: Optional[Sequence[float]],
        image_shape: Sequence[int],
    ) -> Optional[list[tuple[int, int]]]:
        if bbox is None or len(bbox) != 4:
            return None

        height, width = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return [(x1, y1), (x2, y2)]
