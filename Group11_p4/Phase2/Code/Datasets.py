import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisualDataset(Dataset):
    """
    Dataset for visual-only odometry from the generated sequence format.

    Each sample is an adjacent frame pair (t, t+1) with relative pose target

    Output format:
    - {'image_t', 'image_tp1', 'target_rel_pose': [dx, dy, qw, qx, qy, qz]}
    Note: dz is excluded for ground plane motion assumption
    """

    def __init__(
        self,
        data_dir,
        split: Optional[str] = None,
        transform=None,
        image_height: int = 360,
        image_width: int = 480,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.strict = strict
        self.mode = mode
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.image_height = image_height
        self.image_width = image_width

        if mode not in ["pairs", "sequences"]:
            raise ValueError(f"mode must be 'pairs' or 'sequences', got '{mode}'")

        if mode == "sequences" and sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")

        self.generated_root, self.split = self._resolve_generated_root_and_split(
            self.data_dir, split
        )
        self.index_path = self.generated_root / "index.csv"

        if not self.index_path.exists():
            raise FileNotFoundError(f"Could not find index.csv at {self.index_path}")

        self.sequences: List[Dict] = []
        self.samples: List[Tuple[int, int]] = []
        self._build_index()

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in split='{self.split}' under "
                f"{self.generated_root}"
            )

    @staticmethod
    def _resolve_generated_root_and_split(
        data_dir: Path, split: Optional[str]
    ) -> Tuple[Path, str]:
        if (data_dir / "index.csv").exists():
            generated_root = data_dir
            resolved_split = split or "train"
            return generated_root, resolved_split

        if (data_dir.parent / "index.csv").exists():
            generated_root = data_dir.parent
            resolved_split = split or data_dir.name
            return generated_root, resolved_split

        raise FileNotFoundError(
            "Could not resolve Generated root. Provide either the Generated root "
            "(contains index.csv) or a split folder like Generated/train."
        )

    def _build_index(self) -> None:
        with self.index_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get("split") == self.split]

        if not rows:
            raise ValueError(
                f"No sequences found for split='{self.split}' in {self.index_path}"
            )

        for row in rows:
            seq = self._load_sequence(row)
            seq_idx = len(self.sequences)
            self.sequences.append(seq)

            for local_idx in range(seq["num_samples"]):
                self.samples.append((seq_idx, local_idx))

    def _load_sequence(self, row: Dict[str, str]) -> Dict:
        seq_id = row["sequence_id"]
        seq_path = self.generated_root / row["rel_path"]
        frames_dir = seq_path / "frames"
        poses_path = seq_path / "poses.csv"

        required = [frames_dir, poses_path]
        for req_path in required:
            if not req_path.exists():
                msg = f"Missing required path for {seq_id}: {req_path}"
                raise FileNotFoundError(msg)

        poses = self._read_poses(poses_path)
        frames = poses["frame_ids"]
        t_abs = poses["translations"]
        q_abs = poses["quaternions"]

        frame_paths = []
        for frame_id in frames:
            frame_path = frames_dir / f"frame_{frame_id:04d}.png"
            if not frame_path.exists():
                msg = f"Missing frame file for {seq_id}: {frame_path}"
                raise FileNotFoundError(msg)
            frame_paths.append(frame_path)

        if len(frame_paths) != len(frames):
            raise ValueError(
                f"Frame/pose mismatch in {seq_id}: "
                f"{len(frame_paths)} frame files vs {len(frames)} pose rows"
            )

        if len(frame_paths) < 2:
            msg = f"Sequence {seq_id} has fewer than 2 valid frames."
            raise ValueError(msg)

        expected_num_frames = int(float(row.get("num_frames", len(frame_paths))))
        if expected_num_frames != len(frame_paths):
            raise ValueError(
                f"num_frames mismatch for {seq_id}: index.csv={expected_num_frames}, "
                f"actual={len(frame_paths)}"
            )

        return {
            "sequence_id": seq_id,
            "shape": row.get("shape", ""),
            "texture": row.get("texture", ""),
            "height_m": float(row.get("height_m", 0.0)),
            "speed_mps": float(row.get("speed_mps", 0.0)),
            "frame_ids": frames,
            "frame_paths": frame_paths,
            "t_abs": t_abs,
            "q_abs": q_abs,
            "num_samples": len(frame_paths) - 1,
            "seq_path": seq_path,
        }

    @staticmethod
    def _empty_sequence(row: Dict[str, str], seq_path: Path) -> Dict:
        return {
            "sequence_id": row.get("sequence_id", ""),
            "shape": row.get("shape", ""),
            "texture": row.get("texture", ""),
            "height_m": float(row.get("height_m", 0.0)),
            "speed_mps": float(row.get("speed_mps", 0.0)),
            "frame_ids": np.array([], dtype=np.int64),
            "frame_paths": [],
            "t_abs": np.empty((0, 3), dtype=np.float32),
            "q_abs": np.empty((0, 4), dtype=np.float32),
            "num_samples": 0,
            "seq_path": seq_path,
        }

    @staticmethod
    def _read_poses(poses_path: Path) -> Dict[str, np.ndarray]:
        frame_ids = []
        translations = []
        quaternions = []

        with poses_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing column '{col}' in {poses_path}")

            for row in reader:
                frame_ids.append(int(row["frame"]))
                translations.append(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])]
                )
                quaternions.append(
                    [
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                    ]
                )

        frame_ids_np = np.array(frame_ids, dtype=np.int64)
        t_abs_np = np.array(translations, dtype=np.float32)
        q_abs_np = np.array(quaternions, dtype=np.float32)
        q_abs_np = VisualDataset._normalize_quaternion_np(q_abs_np)

        return {
            "frame_ids": frame_ids_np,
            "translations": t_abs_np,
            "quaternions": q_abs_np,
        }

    @staticmethod
    def _normalize_quaternion_np(q: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(q, axis=-1, keepdims=True)
        denom = np.clip(denom, a_min=1e-12, a_max=None)
        return q / denom

    @staticmethod
    def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
        q = VisualDataset._normalize_quaternion_np(q[None, :])[0]
        w, x, y, z = q
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _relative_pose(
        cls, t0: np.ndarray, q0: np.ndarray, t1: np.ndarray, q1: np.ndarray
    ) -> np.ndarray:
        # Express translation in the local frame at time t.
        r0 = cls._quat_to_rotmat_np(q0)
        dt_world = t1 - t0
        dt_local = r0.T @ dt_world

        # Relative orientation from frame t to frame t+1.
        q_rel = cls._quat_mul_np(cls._quat_conjugate_np(q0), q1)
        q_rel = cls._normalize_quaternion_np(q_rel[None, :])[0]

        # Return 6D: [dx, dy, qw, qx, qy, qz] (ground plane motion, ignore dz)
        return np.concatenate(
            [dt_local[:2].astype(np.float32), q_rel.astype(np.float32)], axis=0
        )

    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_width, self.image_height), resample=Image.BILINEAR
        )
        arr = np.array(image, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index out of range: {idx}")

        seq_idx, local_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        return self._get_pair_sample(seq, local_idx)

    def _get_pair_sample(self, seq: Dict, local_idx: int) -> Dict:
        """Get a single frame pair sample."""
        frame_t_path = seq["frame_paths"][local_idx]
        frame_tp1_path = seq["frame_paths"][local_idx + 1]

        image_t = self._load_image_as_tensor(frame_t_path)
        image_tp1 = self._load_image_as_tensor(frame_tp1_path)

        if self.transform is not None:
            image_t = self.transform(image_t)
            image_tp1 = self.transform(image_tp1)

        t0 = seq["t_abs"][local_idx]
        q0 = seq["q_abs"][local_idx]
        t1 = seq["t_abs"][local_idx + 1]
        q1 = seq["q_abs"][local_idx + 1]

        target_rel_pose = torch.from_numpy(self._relative_pose(t0, q0, t1, q1))

        return {
            "image_t": image_t,
            "image_tp1": image_tp1,
            "target_rel_pose": target_rel_pose,
            "sequence_id": seq["sequence_id"],
            "frame_t": int(seq["frame_ids"][local_idx]),
            "frame_tp1": int(seq["frame_ids"][local_idx + 1]),
            "shape": seq["shape"],
            "texture": seq["texture"],
            "height_m": float(seq["height_m"]),
        }


class InertialDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("InertialDataset is not implemented yet.")


class VisualInertialDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("VisualInertialDataset is not implemented yet.")
