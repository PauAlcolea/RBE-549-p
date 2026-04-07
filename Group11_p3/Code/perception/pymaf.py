"""
perception/pymaf.py
===================
Optional PyMAF integration for pedestrian SMPL pose/shape estimation.

Design goals
------------
- Keep the main perception pipeline runnable even when PyMAF is not installed.
- Run PyMAF once per scene/camera video (cached), not once per frame.
- Match PyMAF track bboxes to existing "person" detections and attach fields
  directly to those detection objects for JSON export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import subprocess
import shutil
import numpy as np


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """IoU for [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _resolve_path(value: Optional[str], primary_base: Path, fallback_base: Optional[Path] = None) -> Path:
    """Resolve config path with sensible bases."""
    if value is None:
        return primary_base
    p = Path(value)
    if p.is_absolute():
        return p
    p_primary = (primary_base / p).resolve()
    if p_primary.exists():
        return p_primary
    if fallback_base is not None:
        p_fallback = (fallback_base / p).resolve()
        if p_fallback.exists():
            return p_fallback
    return p_primary


def _discover_repo_dir(config_value: Optional[str], code_root: Path) -> Path:
    """Find PyMAF repo directory from config or known in-repo locations."""
    candidates = []
    if config_value:
        candidates.append(_resolve_path(config_value, code_root))
    candidates.extend(
        [
            (code_root / "perception" / "PyMAF").resolve(),
            (code_root / "../External/PyMAF").resolve(),
        ]
    )

    for c in candidates:
        if (c / "demo.py").exists():
            return c
    return candidates[0]


class PymafEstimator:
    """Lightweight bridge between this project and PyMAF's demo pipeline."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.code_root = Path(__file__).resolve().parent.parent
        pymaf_cfg = cfg.get("perception", {}).get("pymaf", {})

        self.enabled = bool(pymaf_cfg.get("enabled", False))
        self.repo_dir = _discover_repo_dir(pymaf_cfg.get("repo_dir"), self.code_root)
        self.output_dir = _resolve_path(pymaf_cfg.get("output_dir", "../Outputs/PyMAF"), self.code_root)
        self.cfg_file = str(pymaf_cfg.get("cfg_file", "configs/pymaf_config.yaml"))
        self.python_exec = str(pymaf_cfg.get("python_exec", "python3"))
        self.detector = str(pymaf_cfg.get("detector", "yolov8"))
        self.tracking_method = str(pymaf_cfg.get("tracking_method", "bbox"))
        self.yolo_img_size = int(pymaf_cfg.get("yolo_img_size", 416))
        self.tracker_batch_size = int(pymaf_cfg.get("tracker_batch_size", 12))
        self.model_batch_size = int(pymaf_cfg.get("model_batch_size", 8))
        self.no_render = bool(pymaf_cfg.get("no_render", True))
        self.reuse_cache = bool(pymaf_cfg.get("reuse_cache", True))
        self.match_iou = float(pymaf_cfg.get("match_iou", 0.20))
        self.min_bbox_side_px = float(pymaf_cfg.get("min_bbox_side_px", 20.0))
        self.include_joints3d = bool(pymaf_cfg.get("include_joints3d", True))
        self.strict_assets = bool(pymaf_cfg.get("strict_assets", False))
        self.smpl_fallback_pkl = _resolve_path(
            str(pymaf_cfg.get("smpl_fallback_pkl", "../Weights/pymaf_male.pkl")),
            self.code_root,
        )

        ckpt_cfg = pymaf_cfg.get("checkpoint")
        if ckpt_cfg:
            self.checkpoint = _resolve_path(str(ckpt_cfg), self.repo_dir, fallback_base=self.code_root)
        else:
            checkpoint_candidates = [
                self.repo_dir / "data/pretrained_model/PyMAF_model_checkpoint.pt",
                (self.code_root / "../Weights/PyMAF_model_checkpoint.pt").resolve(),
                (self.code_root / "../Weights/pymaf_model_checkpoint.pt").resolve(),
            ]
            self.checkpoint = checkpoint_candidates[0]
            for cand in checkpoint_candidates:
                if cand.exists():
                    self.checkpoint = cand
                    break

        self._frame_results: Dict[int, List[dict]] = {}
        self._prepared_key: Optional[str] = None

        if not self.enabled:
            return

        self._bootstrap_smpl_models()
        missing = self._collect_missing_assets()
        if missing:
            level = "error" if self.strict_assets else "warn"
            print(f"[{level}] PyMAF preflight failed. Missing required assets:")
            for m in missing:
                print(f"[{level}]   - {m}")
            print(
                f"[{level}] Set perception.pymaf paths in config.yaml or add the files above. "
                "PyMAF will be disabled for this run."
            )
            self.enabled = False
            if self.strict_assets:
                raise RuntimeError("PyMAF strict asset mode enabled and required files are missing.")
            return

    def is_active(self) -> bool:
        return self.enabled

    def prepare_scene(self, scene_name: str, camera: str, scene_dir: Path):
        """
        Ensure PyMAF output for this scene/camera is available and loaded.
        """
        if not self.enabled:
            return

        video_path = self._find_video(scene_dir, camera)
        key = f"{scene_name}:{camera}:{video_path}"
        if self._prepared_key == key:
            return

        output_pkl = self._ensure_output_pkl(scene_name, camera, video_path)
        self._frame_results = self._load_output(output_pkl)
        self._prepared_key = key
        print(
            f"[pymaf] Loaded {sum(len(v) for v in self._frame_results.values())} "
            f"track-frames from {output_pkl}"
        )

    def annotate_person_detections(self, frame_idx: int, detections: list) -> int:
        """
        Attach PyMAF SMPL fields to person detections in-place.
        Returns number of matched detections.
        """
        if not self.enabled or not detections:
            return 0

        candidates = self._frame_results.get(int(frame_idx), [])
        if not candidates:
            return 0

        used = set()
        matched = 0

        for det in detections:
            det_bbox = getattr(det, "bbox", None)
            if det_bbox is None or len(det_bbox) != 4:
                continue

            width = float(det_bbox[2]) - float(det_bbox[0])
            height = float(det_bbox[3]) - float(det_bbox[1])
            if min(width, height) < self.min_bbox_side_px:
                continue

            best_idx = -1
            best_iou = 0.0
            for idx, cand in enumerate(candidates):
                if idx in used:
                    continue
                iou = _bbox_iou(det_bbox, cand["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx < 0 or best_iou < self.match_iou:
                continue

            used.add(best_idx)
            cand = candidates[best_idx]
            matched += 1

            det.pymaf_track_id = int(cand["track_id"])
            det.pymaf_match_iou = float(best_iou)
            if cand.get("pose") is not None:
                det.smpl_pose = cand["pose"]
            if cand.get("betas") is not None:
                det.smpl_betas = cand["betas"]
            if self.include_joints3d and cand.get("joints3d") is not None:
                det.smpl_joints3d = cand["joints3d"]

        return matched

    def _find_video(self, scene_dir: Path, camera: str) -> Path:
        undist_dir = Path(scene_dir) / "Undist"
        matches = sorted(undist_dir.glob(f"*-{camera}_undistort.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"PyMAF expected video '*-{camera}_undistort.mp4' in {undist_dir}"
            )
        # demo.py is executed with cwd=self.repo_dir, so ensure absolute path.
        return matches[0].resolve()

    def _bootstrap_smpl_models(self):
        """
        If SMPL files are missing but a fallback pickle exists, populate
        data/smpl/SMPL_{MALE,FEMALE,NEUTRAL}.pkl from that fallback.
        """
        smpl_dir = self.repo_dir / "data" / "smpl"
        smpl_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            smpl_dir / "SMPL_MALE.pkl",
            smpl_dir / "SMPL_FEMALE.pkl",
            smpl_dir / "SMPL_NEUTRAL.pkl",
        ]

        if all(t.exists() for t in targets):
            return
        if not self.smpl_fallback_pkl.exists():
            return

        for tgt in targets:
            if tgt.exists():
                continue
            try:
                os.symlink(self.smpl_fallback_pkl, tgt)
            except OSError:
                shutil.copy2(self.smpl_fallback_pkl, tgt)
        print(
            "[pymaf] Bootstrapped missing SMPL model files from fallback: "
            f"{self.smpl_fallback_pkl}"
        )

    def _collect_missing_assets(self) -> List[str]:
        cfg_path = Path(self.cfg_file)
        if not cfg_path.is_absolute():
            cfg_path = self.repo_dir / cfg_path

        required = [
            self.repo_dir / "demo.py",
            cfg_path,
            self.checkpoint,
            self.repo_dir / "data/mesh_downsampling.npz",
            self.repo_dir / "data/UV_data/UV_Processed.mat",
            self.repo_dir / "data/UV_data/UV_symmetry_transforms.mat",
            self.repo_dir / "data/J_regressor_h36m.npy",
            self.repo_dir / "data/J_regressor_extra.npy",
            self.repo_dir / "data/smpl_mean_params.npz",
            self.repo_dir / "data/smpl/SMPL_NEUTRAL.pkl",
            self.repo_dir / "data/smpl/SMPL_MALE.pkl",
            self.repo_dir / "data/smpl/SMPL_FEMALE.pkl",
        ]

        missing = [str(p) for p in required if not p.exists()]
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg (executable not found in PATH)")
        return missing

    def _ensure_output_pkl(self, scene_name: str, camera: str, video_path: Path) -> Path:
        scene_out_root = (self.output_dir / scene_name / camera).resolve()
        clip_out_dir = scene_out_root / video_path.stem
        output_pkl = clip_out_dir / "output.pkl"

        if self.reuse_cache and output_pkl.exists():
            return output_pkl

        scene_out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_exec,
            "demo.py",
            "--vid_file",
            str(video_path),
            "--output_folder",
            str(scene_out_root),
            "--tracking_method",
            self.tracking_method,
            "--detector",
            self.detector,
            "--cfg_file",
            self.cfg_file,
            "--checkpoint",
            str(self.checkpoint),
            "--model_batch_size",
            str(self.model_batch_size),
            "--tracker_batch_size",
            str(self.tracker_batch_size),
            "--yolo_img_size",
            str(self.yolo_img_size),
        ]
        if self.no_render:
            cmd.append("--no_render")

        env = os.environ.copy()
        # Respect explicit CPU override from the parent pipeline.
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        print(f"[pymaf] Running: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            cwd=str(self.repo_dir),
            check=True,
            env=env,
        )

        if not output_pkl.exists():
            raise RuntimeError(f"PyMAF finished without producing expected output: {output_pkl}")
        return output_pkl

    def _load_output(self, output_pkl: Path) -> Dict[int, List[dict]]:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "PyMAF integration requires joblib to load output.pkl. "
                "Install joblib in the run_perception environment."
            ) from exc

        pred_results: Any = joblib.load(str(output_pkl))
        frame_map: Dict[int, List[dict]] = {}

        if not isinstance(pred_results, dict):
            return frame_map

        for fallback_track_id, (track_id, track) in enumerate(pred_results.items()):
            try:
                track_id_int = int(track_id)
            except (TypeError, ValueError):
                track_id_int = int(fallback_track_id)

            frame_ids = np.asarray(track.get("frame_ids", []), dtype=np.int64)
            bboxes = np.asarray(track.get("bboxes", []), dtype=np.float32)
            poses = np.asarray(track.get("pose", []), dtype=np.float32)
            betas = np.asarray(track.get("betas", []), dtype=np.float32)
            joints3d = np.asarray(track.get("joints3d", []), dtype=np.float32)

            for i, fid in enumerate(frame_ids):
                if i >= len(bboxes):
                    continue
                cx, cy, w, h = [float(v) for v in bboxes[i]]
                if w <= 1e-3 or h <= 1e-3:
                    continue

                det = {
                    "track_id": track_id_int,
                    "bbox": [cx - (w * 0.5), cy - (h * 0.5), cx + (w * 0.5), cy + (h * 0.5)],
                }
                if i < len(poses):
                    det["pose"] = poses[i].astype(float).tolist()
                if i < len(betas):
                    det["betas"] = betas[i].astype(float).tolist()
                if self.include_joints3d and i < len(joints3d):
                    det["joints3d"] = joints3d[i].astype(float).tolist()

                frame_map.setdefault(int(fid), []).append(det)

        return frame_map
