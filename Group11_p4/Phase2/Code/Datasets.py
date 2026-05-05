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
        'target_rel_poses': (seq_len-1, 7),  # [dx, dy, dz, qw, qx, qy, qz]
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
            rel_pose = VisualDataset._relative_pose(
                seq["t_abs"][i], seq["q_abs"][i], seq["t_abs"][i + 1], seq["q_abs"][i + 1]
            )
            target_rel_poses.append(rel_pose)
        target_rel_poses_tensor = torch.from_numpy(np.stack(target_rel_poses, axis=0))

        return {
            "imu": imu_tensor,
            "target_rel_poses": target_rel_poses_tensor,
            "sequence_id": seq["sequence_id"],
            "frame_start": int(seq["frame_ids"][start_idx]),
            "frame_end": int(seq["frame_ids"][end_idx - 1]),
            "shape": seq["shape"],
            "texture": seq["texture"],
            "height_m": float(seq["height_m"]),
        }


class VisualInertialDataset(VisualDataset):
    """
    Dataset for visual-inertial odometry with synchronized camera and IMU data.
    
    Extends VisualDataset to include IMU measurements (1000 Hz) synchronized with
    camera frames (100 Hz). Each image frame has an associated IMU window.
    
    Output format (sequences mode):
        {
            'images': (seq_len, 3, H, W),
            'target_rel_poses': (seq_len-1, 7),
            'imu_windows': (seq_len, imu_window_size, 6),  # 6 = [ax, ay, az, wx, wy, wz]
            'imu_timestamps': (seq_len, imu_window_size),
            ...
        }
    
    IMU window: For each frame i, provides IMU samples from previous frame to current frame.
    With 100 Hz images and 1000 Hz IMU, each window contains 10 IMU samples.
    """
    
    def __init__(
        self,
        data_dir,
        split: Optional[str] = None,
        transform=None,
        strict: bool = False,
        mode: str = "sequences",
        sequence_length: int = 10,
        sequence_stride: int = 1,
        image_height: int = 360,
        image_width: int = 480,
        imu_hz: float = 1000.0,
        image_hz: float = 100.0,
    ):
        # Initialize visual dataset components
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            strict=strict,
            mode=mode,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
            image_height=image_height,
            image_width=image_width,
        )
        
        self.imu_hz = imu_hz
        self.image_hz = image_hz
        self.imu_window_size = int(imu_hz / image_hz)
        
        # Load IMU data for all sequences
        self._load_imu_data()
    
    def _load_imu_data(self) -> None:
        """Load IMU data for all sequences."""
        for seq in self.sequences:
            seq_path = seq["seq_path"]
            
            # Try to find IMU file (prefer 1000hz version)
            hz_suffix = f"_{int(self.imu_hz)}hz" if self.imu_hz != 100 else ""
            imu_candidates = [
                seq_path / f"{seq['sequence_id']}_imu{hz_suffix}.csv",
                seq_path / f"{seq['sequence_id']}_imu.csv",
                seq_path / f"imu_gt{hz_suffix}.csv",
                seq_path / "imu_gt.csv",
            ]
            
            imu_path = None
            for candidate in imu_candidates:
                if candidate.exists():
                    imu_path = candidate
                    break
            
            if imu_path is None:
                if self.strict:
                    raise FileNotFoundError(
                        f"No IMU file found for sequence {seq['sequence_id']} in {seq_path}"
                    )
                # Create empty IMU data
                seq["imu_data"] = np.empty((0, 6), dtype=np.float32)
                seq["imu_timestamps"] = np.empty((0,), dtype=np.float32)
                seq["has_imu"] = False
                continue
            
            # Read IMU CSV
            imu_data, imu_timestamps = self._read_imu_csv(imu_path)
            seq["imu_data"] = imu_data
            seq["imu_timestamps"] = imu_timestamps
            seq["has_imu"] = True
    
    @staticmethod
    def _read_imu_csv(imu_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read IMU CSV file.
        
        Expected columns: frame, t, ax, ay, az, wx, wy, wz
        
        Returns:
            imu_data: (N, 6) array of [ax, ay, az, wx, wy, wz]
            timestamps: (N,) array of timestamps
        """
        imu_data = []
        timestamps = []
        
        with imu_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "t", "ax", "ay", "az", "wx", "wy", "wz"]
            
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing column '{col}' in {imu_path}")
            
            for row in reader:
                timestamps.append(float(row["t"]))
                imu_data.append([
                    float(row["ax"]),
                    float(row["ay"]),
                    float(row["az"]),
                    float(row["wx"]),
                    float(row["wy"]),
                    float(row["wz"]),
                ])
        
        return (
            np.array(imu_data, dtype=np.float32),
            np.array(timestamps, dtype=np.float32),
        )
    
    def _get_imu_window(
        self, seq: Dict, frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IMU window for a specific frame.
        
        For frame i at 100 Hz, get IMU samples from frame i-1 to i (10 samples at 1000 Hz).
        For the first frame, return the first window.
        
        Args:
            seq: Sequence dictionary
            frame_idx: Index into frame_paths/frame_ids
        
        Returns:
            imu_window: (window_size, 6) IMU measurements
            timestamps: (window_size,) timestamps
        """
        if not seq["has_imu"]:
            # Return zeros if no IMU data
            return (
                np.zeros((self.imu_window_size, 6), dtype=np.float32),
                np.zeros((self.imu_window_size,), dtype=np.float32),
            )
        
        imu_data = seq["imu_data"]
        imu_timestamps = seq["imu_timestamps"]
        
        # IMU indices: for frame i, get samples [i*window_size : (i+1)*window_size]
        start_idx = frame_idx * self.imu_window_size
        end_idx = (frame_idx + 1) * self.imu_window_size
        
        # Handle boundary cases
        if end_idx > len(imu_data):
            # Pad with zeros if not enough IMU samples
            available = imu_data[start_idx:]
            available_ts = imu_timestamps[start_idx:]
            pad_size = self.imu_window_size - len(available)
            
            if pad_size > 0:
                imu_window = np.vstack([
                    available,
                    np.zeros((pad_size, 6), dtype=np.float32)
                ])
                timestamps = np.hstack([
                    available_ts,
                    np.zeros((pad_size,), dtype=np.float32)
                ])
            else:
                imu_window = available
                timestamps = available_ts
        else:
            imu_window = imu_data[start_idx:end_idx]
            timestamps = imu_timestamps[start_idx:end_idx]
        
        return imu_window, timestamps
    
    def _get_pair_sample(self, seq: Dict, local_idx: int) -> Dict:
        """Get a single frame pair sample with IMU data."""
        # Get visual data from parent class
        sample = super()._get_pair_sample(seq, local_idx)
        
        # Add IMU window for frame t+1
        imu_window, imu_ts = self._get_imu_window(seq, local_idx + 1)
        sample["imu_window"] = torch.from_numpy(imu_window)
        sample["imu_timestamps"] = torch.from_numpy(imu_ts)
        
        return sample
    
    def _get_sequence_sample(self, seq: Dict, start_idx: int) -> Dict:
        """Get a subsequence sample with multiple frames and IMU windows."""
        # Get visual data from parent class
        sample = super()._get_sequence_sample(seq, start_idx)
        
        # Add IMU windows for all frames in the sequence
        end_idx = start_idx + self.sequence_length
        imu_windows = []
        imu_timestamps_list = []
        
        for i in range(start_idx, end_idx):
            imu_window, imu_ts = self._get_imu_window(seq, i)
            imu_windows.append(imu_window)
            imu_timestamps_list.append(imu_ts)
        
        # Stack: (seq_len, window_size, 6)
        sample["imu_windows"] = torch.from_numpy(np.stack(imu_windows, axis=0))
        # Stack: (seq_len, window_size)
        sample["imu_timestamps"] = torch.from_numpy(np.stack(imu_timestamps_list, axis=0))
        
        return sample

