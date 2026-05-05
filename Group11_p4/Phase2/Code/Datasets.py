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

    Modes:
    - Frame-pair mode (sequence_length=None): Returns (t, t+1) pairs
    - Sequence mode (sequence_length>1): Returns sequences for LSTM

    Output format (pair mode):
    - {'image_t', 'image_tp1', 'target_rel_pose': [dx, dy, qw, qx, qy, qz]}
    
    Output format (sequence mode):
    - {'image_seq': (seq_len, 3, H, W), 'target_poses': (seq_len-1, 6)}
    
    Note: dz is excluded for ground plane motion assumption
    """

    def __init__(
        self,
        data_dir,
        split: Optional[str] = None,
        transform=None,
        image_height: int = 360,
        image_width: int = 480,
        use_augmentation: bool = False,
        sequence_length: Optional[int] = None,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.use_augmentation = use_augmentation
        self.sequence_length = sequence_length
        self.stride = stride
        self.is_sequence_mode = sequence_length is not None and sequence_length > 1

        # Initialize augmentation if enabled
        self.augmentation = None
        if self.use_augmentation:
            from Augmentation import VisualOdometryAugmentation, PairAugmentation

            base_aug = VisualOdometryAugmentation(
                brightness_range=(0.7, 1.3),
                contrast_range=(0.7, 1.3),
                gaussian_noise_std=0.02,
                apply_prob=0.8,
            )
            self.augmentation = PairAugmentation(base_aug)


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

            if self.is_sequence_mode:
                # Sequence mode: create sliding windows
                num_frames = len(seq["frame_paths"])
                for start_idx in range(0, num_frames - self.sequence_length * self.stride + 1, self.stride):
                    self.samples.append((seq_idx, start_idx))
            else:
                # Frame-pair mode: each consecutive pair
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
        
        if self.is_sequence_mode:
            return self._get_sequence_sample(seq, local_idx)
        else:
            return self._get_pair_sample(seq, local_idx)
    
    def _get_sequence_sample(self, seq: Dict, start_idx: int) -> Dict:
        """Get a sequence of frames for LSTM training."""
        images = []
        target_poses = []
        
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.stride
            image = self._load_image_as_tensor(seq["frame_paths"][frame_idx])
            
            if self.transform is not None:
                image = self.transform(image)
            
            images.append(image)
            
            # Compute relative pose for consecutive pairs
            if i < self.sequence_length - 1:
                next_frame_idx = start_idx + (i + 1) * self.stride
                t0 = seq["t_abs"][frame_idx]
                q0 = seq["q_abs"][frame_idx]
                t1 = seq["t_abs"][next_frame_idx]
                q1 = seq["q_abs"][next_frame_idx]
                
                rel_pose = torch.from_numpy(self._relative_pose(t0, q0, t1, q1))
                target_poses.append(rel_pose)
        
        # Stack into tensors
        image_seq = torch.stack(images, dim=0)  # (seq_len, 3, H, W)
        target_poses = torch.stack(target_poses, dim=0)  # (seq_len-1, 6)
        
        return {
            "image_seq": image_seq,
            "target_poses": target_poses,
            "sequence_id": seq["sequence_id"],
            "start_frame": int(seq["frame_ids"][start_idx]),
            "shape": seq["shape"],
            "texture": seq["texture"],
            "height_m": float(seq["height_m"]),
        }

    def _get_pair_sample(self, seq: Dict, local_idx: int) -> Dict:
        """Get a single frame pair sample."""
        frame_t_path = seq["frame_paths"][local_idx]
        frame_tp1_path = seq["frame_paths"][local_idx + 1]

        image_t = self._load_image_as_tensor(frame_t_path)
        image_tp1 = self._load_image_as_tensor(frame_tp1_path)

        if self.transform is not None:
            image_t = self.transform(image_t)
            image_tp1 = self.transform(image_tp1)

        # Apply augmentation to both images with same parameters
        if self.augmentation is not None:
            image_t, image_tp1 = self.augmentation(image_t, image_tp1)

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
    """
    Dataset for inertial-only odometry from generated sequence format.

    Returns samples in sequence mode:
    {
        'imu': (seq_len, 6),                 # [ax, ay, az, wx, wy, wz]
        'target_rel_pose': (seq_len-1, 7),   # [dx, dy, dz, qw, qx, qy, qz]
        ...
    }
    """

    def __init__(
        self,
        data_dir,
        split: Optional[str] = None,
        strict: bool = False,
        sequence_length: int = 10,
        sequence_stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.strict = strict
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")

        self.generated_root, self.split = VisualDataset._resolve_generated_root_and_split(
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
                f"No valid samples found in split='{self.split}' under {self.generated_root}"
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

            num_frames = len(seq["frame_ids"])
            for start_idx in range(
                0, num_frames - self.sequence_length + 1, self.sequence_stride
            ):
                self.samples.append((seq_idx, start_idx))

    def _load_sequence(self, row: Dict[str, str]) -> Dict:
        seq_id = row["sequence_id"]
        seq_path = self.generated_root / row["rel_path"]
        poses_path = seq_path / "poses.csv"
        imu_path = seq_path / f"{seq_id}_imu.csv"

        required = [poses_path, imu_path]
        for req_path in required:
            if not req_path.exists():
                msg = f"Missing required path for {seq_id}: {req_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                return self._empty_sequence(row, seq_path)

        poses = VisualDataset._read_poses(poses_path)
        imu = self._read_imu(imu_path)

        pose_frame_to_idx = {int(fid): i for i, fid in enumerate(poses["frame_ids"])}
        imu_frame_to_idx = {int(fid): i for i, fid in enumerate(imu["frame_ids"])}
        common_frames = sorted(set(pose_frame_to_idx.keys()) & set(imu_frame_to_idx.keys()))

        if len(common_frames) < 2:
            msg = f"Sequence {seq_id} has fewer than 2 aligned IMU/pose frames."
            if self.strict:
                raise ValueError(msg)
            return self._empty_sequence(row, seq_path)

        pose_keep = [pose_frame_to_idx[fid] for fid in common_frames]
        imu_keep = [imu_frame_to_idx[fid] for fid in common_frames]

        frame_ids = np.array(common_frames, dtype=np.int64)
        t_abs = poses["translations"][pose_keep]
        q_abs = poses["quaternions"][pose_keep]
        imu_values = imu["values"][imu_keep]

        return {
            "sequence_id": seq_id,
            "shape": row.get("shape", ""),
            "texture": row.get("texture", ""),
            "height_m": float(row.get("height_m", 0.0)),
            "speed_mps": float(row.get("speed_mps", 0.0)),
            "frame_ids": frame_ids,
            "imu": imu_values,
            "t_abs": t_abs,
            "q_abs": q_abs,
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
            "imu": np.empty((0, 6), dtype=np.float32),
            "t_abs": np.empty((0, 3), dtype=np.float32),
            "q_abs": np.empty((0, 4), dtype=np.float32),
            "seq_path": seq_path,
        }

    @staticmethod
    def _read_imu(imu_path: Path) -> Dict[str, np.ndarray]:
        frame_ids = []
        values = []

        with imu_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "ax", "ay", "az", "wx", "wy", "wz"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing column '{col}' in {imu_path}")

            for row in reader:
                frame_ids.append(int(row["frame"]))
                values.append(
                    [
                        float(row["ax"]),
                        float(row["ay"]),
                        float(row["az"]),
                        float(row["wx"]),
                        float(row["wy"]),
                        float(row["wz"]),
                    ]
                )

        return {
            "frame_ids": np.array(frame_ids, dtype=np.int64),
            "values": np.array(values, dtype=np.float32),
        }

    @staticmethod
    def _relative_pose_7d(
        t0: np.ndarray, q0: np.ndarray, t1: np.ndarray, q1: np.ndarray
    ) -> np.ndarray:
        """
        Compute 7D relative pose in local frame.
        
        Returns:
            7D array: [dx, dy, dz, qw, qx, qy, qz]
        """
        # Express translation in the local frame at time t.
        r0 = VisualDataset._quat_to_rotmat_np(q0)
        dt_world = t1 - t0
        dt_local = r0.T @ dt_world

        # Relative orientation from frame t to frame t+1.
        q_rel = VisualDataset._quat_mul_np(VisualDataset._quat_conjugate_np(q0), q1)
        q_rel = VisualDataset._normalize_quaternion_np(q_rel[None, :])[0]

        # Return 7D: [dx, dy, dz, qw, qx, qy, qz]
        return np.concatenate(
            [dt_local.astype(np.float32), q_rel.astype(np.float32)], axis=0
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index out of range: {idx}")

        seq_idx, start_idx = self.samples[idx]
        seq = self.sequences[seq_idx]
        end_idx = start_idx + self.sequence_length

        imu_tensor = torch.from_numpy(seq["imu"][start_idx:end_idx])

        target_rel_poses = []
        for i in range(start_idx, end_idx - 1):
            rel_pose = InertialDataset._relative_pose_7d(
                seq["t_abs"][i], seq["q_abs"][i], seq["t_abs"][i + 1], seq["q_abs"][i + 1]
            )
            target_rel_poses.append(rel_pose)
        target_rel_poses_tensor = torch.from_numpy(np.stack(target_rel_poses, axis=0))

        return {
            "imu": imu_tensor,
            "target_rel_pose": target_rel_poses_tensor,
            "sequence_id": seq["sequence_id"],
            "frame_start": int(seq["frame_ids"][start_idx]),
            "frame_end": int(seq["frame_ids"][end_idx - 1]),
            "shape": seq["shape"],
            "texture": seq["texture"],
            "height_m": float(seq["height_m"]),
        }


class VisualInertialDataset(Dataset):
    """
    Dataset for visual-inertial odometry combining images and IMU data.
    
    Images are sampled at 100Hz, IMU at 1000Hz.
    Returns image pairs with corresponding IMU sequences between frames.
    
    Output format:
    {
        'image_t': (3, H, W),
        'image_tp1': (3, H, W),
        'imu_seq': (N, 6),  # N IMU samples between frames (typically ~10)
        'target_rel_pose': (6,)  # [dx, dy, qw, qx, qy, qz]
    }
    """

    def __init__(
        self,
        data_dir,
        split: Optional[str] = None,
        transform=None,
        image_height: int = 360,
        image_width: int = 480,
        use_augmentation: bool = False,
        imu_hz: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.use_augmentation = use_augmentation
        self.imu_hz = imu_hz

        # Initialize augmentation if enabled
        self.augmentation = None
        if self.use_augmentation:
            from Augmentation import VisualOdometryAugmentation, PairAugmentation

            base_aug = VisualOdometryAugmentation(
                brightness_range=(0.7, 1.3),
                contrast_range=(0.7, 1.3),
                gaussian_noise_std=0.02,
                apply_prob=0.8,
            )
            self.augmentation = PairAugmentation(base_aug)

        self.generated_root, self.split = VisualDataset._resolve_generated_root_and_split(
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
            if seq["num_samples"] > 0:
                seq_idx = len(self.sequences)
                self.sequences.append(seq)
                # Create samples for each consecutive frame pair
                for local_idx in range(seq["num_samples"]):
                    self.samples.append((seq_idx, local_idx))

    def _load_sequence(self, row: Dict[str, str]) -> Dict:
        seq_id = row["sequence_id"]
        seq_path = self.generated_root / row["rel_path"]
        frames_dir = seq_path / "frames"
        poses_path = seq_path / "poses.csv"
        imu_path = seq_path / f"{seq_id}_imu_{self.imu_hz}hz.csv"

        required = [frames_dir, poses_path, imu_path]
        for req_path in required:
            if not req_path.exists():
                print(f"Warning: Missing path for {seq_id}: {req_path}")
                return self._empty_sequence(row, seq_path)

        # Read poses (100 Hz)
        poses = VisualDataset._read_poses(poses_path)
        frames = poses["frame_ids"]
        t_abs = poses["translations"]
        q_abs = poses["quaternions"]

        # Read IMU data (1000 Hz)
        imu_data = self._read_imu(imu_path)

        # Verify frames exist
        frame_paths = []
        for frame_id in frames:
            frame_path = frames_dir / f"frame_{frame_id:04d}.png"
            if not frame_path.exists():
                print(f"Warning: Missing frame file for {seq_id}: {frame_path}")
                return self._empty_sequence(row, seq_path)
            frame_paths.append(frame_path)

        if len(frame_paths) < 2:
            return self._empty_sequence(row, seq_path)

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
            "imu_frame_ids": imu_data["frame_ids"],
            "imu_values": imu_data["values"],
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
            "imu_frame_ids": np.array([], dtype=np.int64),
            "imu_values": np.empty((0, 6), dtype=np.float32),
            "num_samples": 0,
            "seq_path": seq_path,
        }

    @staticmethod
    def _read_imu(imu_path: Path) -> Dict[str, np.ndarray]:
        """Read IMU data from CSV file."""
        frame_ids = []
        values = []

        with imu_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "ax", "ay", "az", "wx", "wy", "wz"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing column '{col}' in {imu_path}")

            for row in reader:
                frame_ids.append(int(row["frame"]))
                values.append(
                    [
                        float(row["ax"]),
                        float(row["ay"]),
                        float(row["az"]),
                        float(row["wx"]),
                        float(row["wy"]),
                        float(row["wz"]),
                    ]
                )

        return {
            "frame_ids": np.array(frame_ids, dtype=np.int64),
            "values": np.array(values, dtype=np.float32),
        }

    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_width, self.image_height), resample=Image.BILINEAR
        )
        arr = np.array(image, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _get_imu_between_frames(
        self, seq: Dict, frame_idx_t: int, frame_idx_tp1: int
    ) -> np.ndarray:
        """
        Get IMU samples between two consecutive image frames.
        
        Args:
            seq: Sequence dictionary
            frame_idx_t: Index of frame t in the sequence
            frame_idx_tp1: Index of frame t+1 in the sequence
        
        Returns:
            IMU values between the two frames, shape (N, 6)
        """
        frame_id_t = seq["frame_ids"][frame_idx_t]
        frame_id_tp1 = seq["frame_ids"][frame_idx_tp1]

        # Find IMU samples with frame IDs in range [frame_id_t, frame_id_tp1)
        # The IMU CSV has frame IDs matching the image frame IDs
        imu_frame_ids = seq["imu_frame_ids"]
        mask = (imu_frame_ids >= frame_id_t) & (imu_frame_ids < frame_id_tp1)
        imu_samples = seq["imu_values"][mask]

        # If no samples found, return empty array
        if len(imu_samples) == 0:
            # Fallback: return zeros to avoid breaking the training
            # In practice, this shouldn't happen if data is properly generated
            return np.zeros((1, 6), dtype=np.float32)

        return imu_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index out of range: {idx}")

        seq_idx, local_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        # Load images
        frame_t_path = seq["frame_paths"][local_idx]
        frame_tp1_path = seq["frame_paths"][local_idx + 1]

        image_t = self._load_image_as_tensor(frame_t_path)
        image_tp1 = self._load_image_as_tensor(frame_tp1_path)

        if self.transform is not None:
            image_t = self.transform(image_t)
            image_tp1 = self.transform(image_tp1)

        # Apply augmentation to both images with same parameters
        if self.augmentation is not None:
            image_t, image_tp1 = self.augmentation(image_t, image_tp1)

        # Get IMU sequence between the two frames
        imu_seq = self._get_imu_between_frames(seq, local_idx, local_idx + 1)
        imu_seq_tensor = torch.from_numpy(imu_seq)

        # Compute relative pose (6D: dx, dy, qw, qx, qy, qz)
        t0 = seq["t_abs"][local_idx]
        q0 = seq["q_abs"][local_idx]
        t1 = seq["t_abs"][local_idx + 1]
        q1 = seq["q_abs"][local_idx + 1]

        target_rel_pose = torch.from_numpy(
            VisualDataset._relative_pose(t0, q0, t1, q1)
        )

        return {
            "image_t": image_t,
            "image_tp1": image_tp1,
            "imu_seq": imu_seq_tensor,
            "target_rel_pose": target_rel_pose,
            "sequence_id": seq["sequence_id"],
            "frame_t": int(seq["frame_ids"][local_idx]),
            "frame_tp1": int(seq["frame_ids"][local_idx + 1]),
            "shape": seq["shape"],
            "texture": seq["texture"],
            "height_m": float(seq["height_m"]),
        }
