#!/usr/bin/env python
"""
Test.py - DL-based odometry model evaluation framework

Evaluates trained visual/inertial/visual-inertial odometry models on test sequences.
Computes ATE, RPE metrics and generates trajectory plots similar to Phase1's EVO evaluation.
"""

import sys

sys.dont_write_bytecode = True

import csv
import json
import os
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from Train import ModelTypes, Pipeline

current_dir = Path(__file__).parent
output_dir = current_dir.parent / "Output" / "Testing"


class SequenceEvaluator:
    """
    Evaluates a trained model on a single test sequence.

    Handles:
    - Loading ground truth from poses.csv
    - Running model inference with sliding windows
    - Accumulating relative poses into absolute trajectory
    - Computing metrics (ATE, RPE)
    - Generating plots and outputs
    """

    def __init__(
        self,
        model,
        model_type: ModelTypes,
        sequence_path: Path,
        image_height: int,
        image_width: int,
        device: str,
    ):
        self.model = model
        self.model_type = model_type
        self.sequence_path = sequence_path
        self.image_height = image_height
        self.image_width = image_width
        self.device = device

        # Load sequence metadata
        self.sequence_id = sequence_path.name
        self.metadata = self._load_metadata()

        # Load ground truth
        self.gt_poses = self._load_ground_truth()
        self.num_frames = len(self.gt_poses["frames"])

    def _load_metadata(self) -> Dict:
        """Load sequence metadata from metadata.json."""
        metadata_path = self.sequence_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _load_ground_truth(self) -> Dict[str, np.ndarray]:
        """
        Load ground truth poses from poses.csv.

        Returns:
            Dict with:
                - 'frames': (N,) frame IDs
                - 'positions': (N, 3) absolute positions [tx, ty, tz]
                - 'quaternions': (N, 4) absolute rotations [qw, qx, qy, qz]
        """
        poses_path = self.sequence_path / "poses.csv"
        if not poses_path.exists():
            raise FileNotFoundError(f"Ground truth poses not found: {poses_path}")

        frames = []
        positions = []
        quaternions = []

        with open(poses_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(int(row["frame"]))
                positions.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
                quaternions.append(
                    [
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                    ]
                )

        return {
            "frames": np.array(frames, dtype=np.int64),
            "positions": np.array(positions, dtype=np.float64),
            "quaternions": np.array(quaternions, dtype=np.float64),
        }

    def _load_image(self, frame_id: int) -> torch.Tensor:
        """Load and preprocess a single image frame."""
        frame_path = self.sequence_path / "frames" / f"frame_{frame_id:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # Load image
        img = Image.open(frame_path).convert("RGB")

        # Resize to model input size
        img = img.resize((self.image_width, self.image_height), Image.BILINEAR)

        # Convert to tensor and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)

        return img_tensor

    def _load_aligned_imu(self) -> np.ndarray:
        """
        Load IMU readings aligned to ground-truth frame IDs.

        Returns:
            imu: (N, 6) IMU sequence [ax, ay, az, wx, wy, wz], where N == num_frames
        """
        imu_path = self.sequence_path / f"{self.sequence_id}_imu.csv"
        if not imu_path.exists():
            raise FileNotFoundError(f"IMU file not found: {imu_path}")

        imu_by_frame = {}
        with open(imu_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "ax", "ay", "az", "wx", "wy", "wz"]
            for col in required_cols:
                if col not in (reader.fieldnames or []):
                    raise ValueError(f"Missing IMU column '{col}' in {imu_path}")

            for row in reader:
                frame_id = int(row["frame"])
                imu_by_frame[frame_id] = [
                    float(row["ax"]),
                    float(row["ay"]),
                    float(row["az"]),
                    float(row["wx"]),
                    float(row["wy"]),
                    float(row["wz"]),
                ]

        aligned_imu = []
        for frame_id in self.gt_poses["frames"]:
            if int(frame_id) not in imu_by_frame:
                raise ValueError(
                    f"Missing IMU sample for frame {int(frame_id)} in {imu_path}"
                )
            aligned_imu.append(imu_by_frame[int(frame_id)])

        return np.asarray(aligned_imu, dtype=np.float32)

    def _load_vi_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load high-rate IMU data for visual-inertial inference.

        Tries the 1000Hz file first ({seq_id}_imu_1000hz.csv), then falls back
        to the base IMU file ({seq_id}_imu.csv).

        Returns:
            imu_values: (M, 6) float32 array of [ax, ay, az, wx, wy, wz]
            imu_frame_ids: (M,) int64 array of frame IDs corresponding to each IMU sample
        """
        imu_path = self.sequence_path / f"{self.sequence_id}_imu_1000hz.csv"
        if not imu_path.exists():
            imu_path = self.sequence_path / f"{self.sequence_id}_imu.csv"
        if not imu_path.exists():
            raise FileNotFoundError(
                f"No IMU file found for VI inference in {self.sequence_path}. "
                f"Expected {self.sequence_id}_imu_1000hz.csv or {self.sequence_id}_imu.csv"
            )

        frame_ids = []
        values = []

        with open(imu_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["frame", "ax", "ay", "az", "wx", "wy", "wz"]
            for col in required_cols:
                if col not in (reader.fieldnames or []):
                    raise ValueError(f"Missing IMU column '{col}' in {imu_path}")
            for row in reader:
                frame_ids.append(int(row["frame"]))
                values.append([
                    float(row["ax"]), float(row["ay"]), float(row["az"]),
                    float(row["wx"]), float(row["wy"]), float(row["wz"]),
                ])

        return (
            np.array(values, dtype=np.float32),
            np.array(frame_ids, dtype=np.int64),
        )

    def run_inference(self) -> Dict[str, np.ndarray]:
        """
        Run model inference on the full sequence using consecutive frame pairs.

        Returns:
            Dict with:
                - 'rel_positions': (N-1, 3) predicted relative positions
                - 'rel_quaternions': (N-1, 4) predicted relative rotations
                - 'abs_positions': (N, 3) accumulated absolute positions
                - 'abs_quaternions': (N, 4) accumulated absolute quaternions
        """
        self.model.eval()

        # Collect relative poses by processing consecutive frame pairs
        rel_positions = []
        rel_quaternions = []

        frames = self.gt_poses["frames"]

        with torch.no_grad():
            if self.model_type == ModelTypes.VISUAL:
                images = [self._load_image(int(fid)) for fid in frames]
                images = torch.stack(images, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
                batch = {"images": images}
                pred_poses = self.model(batch).squeeze(0).cpu().numpy()  # (N-1, 6)

                # Split [dx, dy, qw, qx, qy, qz] into translation and quaternion
                # Note: dz=0 for ground plane motion
                for i in range(pred_poses.shape[0]):
                    rel_positions.append(np.array([pred_poses[i, 0], pred_poses[i, 1], 0.0], dtype=np.float64))
                    rel_quaternions.append(pred_poses[i, 2:].astype(np.float64))  # [qw, qx, qy, qz]

            elif self.model_type == ModelTypes.INERTIAL:
                imu = self._load_aligned_imu()
                print(f"DEBUG: Loaded IMU shape: {imu.shape}")
                print(f"DEBUG: First 3 IMU samples:\n{imu[:3]}")
                
                imu = torch.from_numpy(imu).unsqueeze(0).to(self.device)  # (1, N, 6)
                batch = {"imu": imu}
                pred_poses = self.model(batch).squeeze(0).cpu().numpy()  # (N-1, 7)
                
                print(f"\nDEBUG: pred_poses shape: {pred_poses.shape}")
                print(f"DEBUG: First 3 predictions:\n{pred_poses[:3]}")
                
                # Compute GT using InertialDataset's method
                print(f"\nDEBUG: Ground truth (7D) first 3 relative poses:")
                from Datasets import InertialDataset
                for i in range(min(3, len(self.gt_poses["positions"]) - 1)):
                    gt_t0 = self.gt_poses["positions"][i]
                    gt_q0 = self.gt_poses["quaternions"][i]
                    gt_t1 = self.gt_poses["positions"][i + 1]
                    gt_q1 = self.gt_poses["quaternions"][i + 1]
                    gt_rel = InertialDataset._relative_pose_7d(gt_t0, gt_q0, gt_t1, gt_q1)
                    print(f"  GT {i}: {gt_rel}")
                print()

                # Split [dx, dy, dz, qw, qx, qy, qz] into translation and quaternion
                for i in range(pred_poses.shape[0]):
                    rel_positions.append(pred_poses[i, :3])
                    rel_quaternions.append(pred_poses[i, 3:])

            elif self.model_type == ModelTypes.VISUAL_INERTIAL:
                # Load high-rate IMU data for VI inference
                imu_values, imu_frame_ids = self._load_vi_imu()

                for i in tqdm(range(len(frames) - 1), desc="  VI inference", leave=False):
                    frame_t = int(frames[i])
                    frame_tp1 = int(frames[i + 1])

                    # Load image pair
                    img_t = self._load_image(frame_t).unsqueeze(0).to(self.device)    # (1, 3, H, W)
                    img_tp1 = self._load_image(frame_tp1).unsqueeze(0).to(self.device)  # (1, 3, H, W)

                    # Slice IMU samples in range [frame_t, frame_tp1)
                    mask = (imu_frame_ids >= frame_t) & (imu_frame_ids < frame_tp1)
                    imu_slice = imu_values[mask]
                    if len(imu_slice) == 0:
                        imu_slice = np.zeros((1, 6), dtype=np.float32)
                    imu_tensor = torch.from_numpy(imu_slice).unsqueeze(0).to(self.device)  # (1, N_imu, 6)

                    batch = {"image_t": img_t, "image_tp1": img_tp1, "imu_seq": imu_tensor}
                    pred = self.model(batch).squeeze(0).cpu().numpy()  # (6,) [dx, dy, qw, qx, qy, qz]

                    # VI model predicts 2D translation [dx, dy]; pad dz=0 for 3D accumulation
                    rel_positions.append(np.array([pred[0], pred[1], 0.0], dtype=np.float64))
                    rel_quaternions.append(pred[2:].astype(np.float64))  # [qw, qx, qy, qz]

            else:
                raise NotImplementedError(f"Inference for {self.model_type.name} is not implemented.")

        # Stack all predictions
        rel_positions = np.stack(rel_positions, axis=0)  # (N-1, 3)
        rel_quaternions = np.stack(rel_quaternions, axis=0)  # (N-1, 4)

        # Accumulate into absolute trajectory
        abs_positions, abs_quaternions = self._accumulate_trajectory(
            rel_positions, rel_quaternions
        )

        return {
            "rel_positions": rel_positions,
            "rel_quaternions": rel_quaternions,
            "abs_positions": abs_positions,
            "abs_quaternions": abs_quaternions,
        }

    def _accumulate_trajectory(
        self,
        rel_positions: np.ndarray,
        rel_quaternions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accumulate relative poses into absolute trajectory.

        Args:
            rel_positions: (N-1, 3) relative positions [dx, dy, dz]
            rel_quaternions: (N-1, 4) relative rotations [qw, qx, qy, qz]

        Returns:
            abs_positions: (N, 3) absolute positions
            abs_quaternions: (N, 4) absolute rotations
        """
        N = len(rel_positions) + 1

        # Initialize with ground truth initial pose
        abs_positions = np.zeros((N, 3), dtype=np.float64)
        abs_quaternions = np.zeros((N, 4), dtype=np.float64)

        abs_positions[0] = self.gt_poses["positions"][0]
        abs_quaternions[0] = self.gt_poses["quaternions"][0]

        # Accumulate poses
        for i in range(N - 1):
            # Current absolute rotation as quaternion
            q_abs = abs_quaternions[i]

            # Relative motion
            dt = rel_positions[i]
            dq = rel_quaternions[i]

            # Rotate relative translation by current absolute rotation
            dt_world = self._rotate_vector_by_quaternion(q_abs, dt)

            # Update absolute position
            abs_positions[i + 1] = abs_positions[i] + dt_world

            # Compose rotations: q_new = q_abs * dq
            abs_quaternions[i + 1] = self._quaternion_multiply(q_abs, dq)

            # Normalize quaternion to prevent drift
            abs_quaternions[i + 1] = self._normalize_quaternion(abs_quaternions[i + 1])

        return abs_positions, abs_quaternions

    @staticmethod
    def _rotate_vector_by_quaternion(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate vector v by quaternion q.

        Args:
            q: (4,) quaternion [qw, qx, qy, qz]
            v: (3,) vector [x, y, z]

        Returns:
            v_rotated: (3,) rotated vector
        """
        qw, qx, qy, qz = q
        x, y, z = v

        # Convert to rotation matrix
        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

        return R @ v

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q_result = q1 * q2

        Args:
            q1: (4,) quaternion [qw, qx, qy, qz]
            q2: (4,) quaternion [qw, qx, qy, qz]

        Returns:
            q_result: (4,) quaternion product
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    def compute_metrics(self, prediction: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute trajectory error metrics (ATE, RPE, RMSE).

        Args:
            prediction: Dict with 'abs_positions' and 'abs_quaternions'

        Returns:
            Dict with metrics: ate_rmse, ate_mean, ate_std, rpe_trans_rmse, rpe_rot_rmse, etc.
        """
        gt_pos = self.gt_poses["positions"]
        est_pos = prediction["abs_positions"]

        # Align estimate to ground truth using rigid SE3 transformation
        est_aligned, R, t = self._align_se3(est_pos, gt_pos)

        # Absolute Trajectory Error
        ate_errors = est_aligned - gt_pos
        ate_norms = np.linalg.norm(ate_errors, axis=1)

        ate_rmse = np.sqrt(np.mean(ate_norms**2))
        ate_mean = np.mean(ate_norms)
        ate_std = np.std(ate_norms)
        ate_median = np.median(ate_norms)
        ate_max = np.max(ate_norms)
        ate_min = np.min(ate_norms)

        # Per-axis RMSE
        rmse_x = np.sqrt(np.mean(ate_errors[:, 0] ** 2))
        rmse_y = np.sqrt(np.mean(ate_errors[:, 1] ** 2))
        rmse_z = np.sqrt(np.mean(ate_errors[:, 2] ** 2))

        # Relative Pose Error (frame-to-frame)
        rpe_trans_errors = []
        rpe_rot_errors = []

        gt_quat = self.gt_poses["quaternions"]
        est_quat = prediction["abs_quaternions"]

        for i in range(len(gt_pos) - 1):
            # Ground truth relative motion
            gt_dt = gt_pos[i + 1] - gt_pos[i]
            gt_dq = self._quaternion_multiply(
                self._quaternion_inverse(gt_quat[i]), gt_quat[i + 1]
            )

            # Estimated relative motion (from aligned trajectory)
            est_dt_aligned = est_aligned[i + 1] - est_aligned[i]
            est_dq = self._quaternion_multiply(
                self._quaternion_inverse(est_quat[i]), est_quat[i + 1]
            )

            # Translation error
            trans_error = np.linalg.norm(est_dt_aligned - gt_dt)
            rpe_trans_errors.append(trans_error)

            # Rotation error (angle between quaternions)
            rot_error = self._quaternion_angle_diff(gt_dq, est_dq)
            rpe_rot_errors.append(rot_error)

        rpe_trans_errors = np.array(rpe_trans_errors)
        rpe_rot_errors = np.array(rpe_rot_errors)

        rpe_trans_rmse = np.sqrt(np.mean(rpe_trans_errors**2))
        rpe_trans_mean = np.mean(rpe_trans_errors)
        rpe_trans_std = np.std(rpe_trans_errors)

        rpe_rot_rmse = np.sqrt(np.mean(rpe_rot_errors**2))
        rpe_rot_mean = np.mean(rpe_rot_errors)
        rpe_rot_std = np.std(rpe_rot_errors)

        return {
            "ate_rmse": float(ate_rmse),
            "ate_mean": float(ate_mean),
            "ate_std": float(ate_std),
            "ate_median": float(ate_median),
            "ate_max": float(ate_max),
            "ate_min": float(ate_min),
            "rmse_x": float(rmse_x),
            "rmse_y": float(rmse_y),
            "rmse_z": float(rmse_z),
            "rpe_trans_rmse": float(rpe_trans_rmse),
            "rpe_trans_mean": float(rpe_trans_mean),
            "rpe_trans_std": float(rpe_trans_std),
            "rpe_rot_rmse": float(rpe_rot_rmse),
            "rpe_rot_mean": float(rpe_rot_mean),
            "rpe_rot_std": float(rpe_rot_std),
            "num_poses": len(gt_pos),
        }

    @staticmethod
    def _align_se3(
        est: np.ndarray, gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rigidly align estimate to ground truth using SE3 (rotation + translation, no scale).

        Args:
            est: (N, 3) estimated positions
            gt: (N, 3) ground truth positions

        Returns:
            est_aligned: (N, 3) aligned positions
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        if len(est) == 0 or len(gt) == 0:
            return est, np.eye(3), np.zeros(3)

        # Center both point clouds
        est_mean = np.mean(est, axis=0)
        gt_mean = np.mean(gt, axis=0)

        est_centered = est - est_mean
        gt_centered = gt - gt_mean

        # Compute rotation using SVD
        H = est_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = gt_mean - R @ est_mean

        # Apply transformation
        est_aligned = (R @ est.T).T + t

        return est_aligned, R, t

    @staticmethod
    def _quaternion_inverse(q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quaternion_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute angle difference between two quaternions in radians.

        Args:
            q1: (4,) quaternion [qw, qx, qy, qz]
            q2: (4,) quaternion [qw, qx, qy, qz]

        Returns:
            angle: Angle difference in radians
        """
        # Normalize quaternions
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        q2 = q2 / (np.linalg.norm(q2) + 1e-12)

        # Compute dot product
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, 0.0, 1.0)

        # Angle is 2 * arccos(dot)
        angle = 2.0 * np.arccos(dot)

        return float(angle)

    def visualize(
        self,
        prediction: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        output_dir: Path,
        show: bool = False,
    ):
        """
        Generate trajectory plots comparing estimate vs ground truth.

        Args:
            prediction: Dict with 'abs_positions'
            metrics: Dict with computed metrics
            output_dir: Directory to save plots
            show: Whether to display plots interactively
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        gt_pos = self.gt_poses["positions"]
        est_pos = prediction["abs_positions"]

        # Align for visualization
        est_aligned, _, _ = self._align_se3(est_pos, gt_pos)

        # Compute errors
        errors = est_aligned - gt_pos
        error_norms = np.linalg.norm(errors, axis=1)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 5))

        # 1. Trajectory plot (X-Y view)
        ax1 = fig.add_subplot(131)
        ax1.plot(
            gt_pos[:, 0],
            gt_pos[:, 1],
            "o-",
            label="Ground Truth",
            color="tab:pink",
            linewidth=2.0,
            markersize=3,
        )
        ax1.plot(
            est_aligned[:, 0],
            est_aligned[:, 1],
            "o-",
            label="Estimate",
            color="tab:blue",
            linewidth=1.7,
            markersize=2,
        )
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("Trajectory (X-Y View)")
        ax1.axis("equal")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Error norm over time
        ax2 = fig.add_subplot(132)
        time_steps = np.arange(len(error_norms))
        ax2.plot(time_steps, error_norms, "k-", linewidth=1.0, label="|e|")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Position Error [m]")
        ax2.set_title("Position Error Norm vs Frame")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. ATE histogram
        ax3 = fig.add_subplot(133)
        ax3.hist(error_norms, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
        ax3.axvline(
            metrics["ate_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {metrics['ate_mean']:.3f}m",
        )
        ax3.axvline(
            metrics["ate_median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {metrics['ate_median']:.3f}m",
        )
        ax3.set_xlabel("Position Error [m]")
        ax3.set_ylabel("Frequency")
        ax3.set_title("ATE Distribution")
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.legend()

        # Overall title with metrics
        fig.suptitle(
            f'{self.sequence_id} | ATE RMSE: {metrics["ate_rmse"]:.3f}m | '
            f'RMSE [x={metrics["rmse_x"]:.3f}, y={metrics["rmse_y"]:.3f}, z={metrics["rmse_z"]:.3f}]',
            fontsize=12,
            fontweight="bold",
        )

        fig.tight_layout()

        # Save figure
        plot_path = output_dir / f"{self.sequence_id}_trajectory.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"  Saved plot: {plot_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_outputs(
        self,
        prediction: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        output_dir: Path,
    ):
        """
        Save prediction outputs in multiple formats.

        Args:
            prediction: Dict with predictions
            metrics: Dict with computed metrics
            output_dir: Directory to save outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_path = output_dir / f"{self.sequence_id}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "sequence_id": self.sequence_id,
                    "model_type": self.model_type.name,
                    "metadata": self.metadata,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"  Saved metrics: {metrics_path}")

        # Save trajectory as CSV
        csv_path = output_dir / f"{self.sequence_id}_trajectory.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame",
                    "gt_tx",
                    "gt_ty",
                    "gt_tz",
                    "gt_qw",
                    "gt_qx",
                    "gt_qy",
                    "gt_qz",
                    "est_tx",
                    "est_ty",
                    "est_tz",
                    "est_qw",
                    "est_qx",
                    "est_qy",
                    "est_qz",
                ]
            )

            for i in range(len(prediction["abs_positions"])):
                frame = self.gt_poses["frames"][i]
                gt_pos = self.gt_poses["positions"][i]
                gt_quat = self.gt_poses["quaternions"][i]
                est_pos = prediction["abs_positions"][i]
                est_quat = prediction["abs_quaternions"][i]

                writer.writerow(
                    [
                        frame,
                        gt_pos[0],
                        gt_pos[1],
                        gt_pos[2],
                        gt_quat[0],
                        gt_quat[1],
                        gt_quat[2],
                        gt_quat[3],
                        est_pos[0],
                        est_pos[1],
                        est_pos[2],
                        est_quat[0],
                        est_quat[1],
                        est_quat[2],
                        est_quat[3],
                    ]
                )
        print(f"  Saved CSV: {csv_path}")

        # Save in TUM format (for EVO compatibility)
        tum_gt_path = output_dir / f"{self.sequence_id}_tum_gt.txt"
        tum_est_path = output_dir / f"{self.sequence_id}_tum_est.txt"

        # TUM format: timestamp tx ty tz qx qy qz qw (note: qw is LAST in TUM)
        with open(tum_gt_path, "w") as f:
            for i in range(len(self.gt_poses["positions"])):
                frame = self.gt_poses["frames"][i]
                pos = self.gt_poses["positions"][i]
                quat = self.gt_poses["quaternions"][i]  # [qw, qx, qy, qz]
                # TUM uses qx qy qz qw order
                f.write(
                    f"{frame} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {quat[0]:.6f}\n"
                )

        with open(tum_est_path, "w") as f:
            for i in range(len(prediction["abs_positions"])):
                frame = self.gt_poses["frames"][i]
                pos = prediction["abs_positions"][i]
                quat = prediction["abs_quaternions"][i]  # [qw, qx, qy, qz]
                # TUM uses qx qy qz qw order
                f.write(
                    f"{frame} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {quat[0]:.6f}\n"
                )

        print(f"  Saved TUM GT: {tum_gt_path}")
        print(f"  Saved TUM EST: {tum_est_path}")


def load_model(
    checkpoint_path: Path, model_type: ModelTypes, device: str
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model to load
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint first so we can inspect it before building the model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle torch.compile() prefix (_orig_mod.) if present
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }

    # Create model architecture
    if model_type == ModelTypes.VISUAL:
        from Models import VisualModel

        model = VisualModel()
    elif model_type == ModelTypes.INERTIAL:
        from Models import InertialModel

        model = InertialModel()
    elif model_type == ModelTypes.VISUAL_INERTIAL:
        from Models import VisualInertialModel

        # Infer lstm_hidden_size from the checkpoint's LSTM weight shape so the
        # architecture matches what was saved, regardless of training defaults.
        lstm_key = next((k for k in state_dict if "lstm.weight_hh_l0" in k), None)
        lstm_hidden_size = int(state_dict[lstm_key].shape[1]) if lstm_key else 256
        model = VisualInertialModel(
            feature_size=256, hidden_size=512, lstm_hidden_size=lstm_hidden_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"with validation loss {checkpoint['val_loss']:.4f}"
    )

    return model


def get_test_sequences(
    data_dir: Path, split: str = "test", sequence_id: Optional[str] = None
) -> List[Path]:
    """
    Get list of sequence directories to evaluate.

    Args:
        data_dir: Root data directory (should contain split subdirectories or be a split itself)
        split: Which split to use ('train', 'val', or 'test')
        sequence_id: Optional specific sequence ID to evaluate

    Returns:
        List of sequence paths
    """
    # Try to find the split directory
    if (data_dir / split).exists():
        # data_dir is the Generated root, use the split subdirectory
        eval_dir = data_dir / split
    elif data_dir.name in ["train", "val", "test"]:
        # data_dir is already a split directory
        eval_dir = data_dir
    else:
        # Assume data_dir itself contains sequences
        eval_dir = data_dir

    if sequence_id:
        # Evaluate specific sequence
        seq_path = eval_dir / sequence_id
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence not found: {seq_path}")
        return [seq_path]
    else:
        # Evaluate all sequences in directory
        sequences = sorted(
            [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("seq_")]
        )
        if not sequences:
            raise ValueError(f"No sequences found in {eval_dir}")
        return sequences


def test_single_sequence(
    model: torch.nn.Module,
    model_type: ModelTypes,
    sequence_path: Path,
    image_height: int,
    image_width: int,
    device: str,
    output_dir: Path,
    show_plots: bool = False,
) -> Dict[str, float]:
    """
    Test model on a single sequence.

    Returns:
        Dict with computed metrics
    """
    print(f"\nEvaluating {sequence_path.name}...")

    # Create evaluator
    evaluator = SequenceEvaluator(
        model=model,
        model_type=model_type,
        sequence_path=sequence_path,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )

    # Run inference
    print(f"  Running inference on {evaluator.num_frames} frames...")
    prediction = evaluator.run_inference()

    # Compute metrics
    print(f"  Computing metrics...")
    metrics = evaluator.compute_metrics(prediction)

    # Create output directory for this sequence
    seq_output_dir = output_dir / sequence_path.name

    # Generate visualizations
    print(f"  Generating visualizations...")
    evaluator.visualize(prediction, metrics, seq_output_dir, show=show_plots)

    # Save outputs
    print(f"  Saving outputs...")
    evaluator.save_outputs(prediction, metrics, seq_output_dir)

    # Print summary
    print(f"\n  Results for {sequence_path.name}:")
    print(f"    ATE RMSE:  {metrics['ate_rmse']:.4f} m")
    print(f"    ATE Mean:  {metrics['ate_mean']:.4f} m")
    print(
        f"    RMSE X/Y/Z: {metrics['rmse_x']:.4f} / {metrics['rmse_y']:.4f} / {metrics['rmse_z']:.4f} m"
    )
    print(f"    RPE Trans: {metrics['rpe_trans_rmse']:.4f} m")
    print(
        f"    RPE Rot:   {metrics['rpe_rot_rmse']:.4f} rad ({np.rad2deg(metrics['rpe_rot_rmse']):.2f}°)"
    )

    return metrics


def test_batch(
    model: torch.nn.Module,
    model_type: ModelTypes,
    sequences: List[Path],
    image_height: int,
    image_width: int,
    device: str,
    output_dir: Path,
    show_plots: bool = False,
):
    """
    Test model on multiple sequences and generate summary.

    Args:
        model: Trained model
        model_type: Type of model
        sequences: List of sequence paths
        image_height: Image height
        image_width: Image width
        device: Device to run on
        output_dir: Output directory
        show_plots: Whether to show plots
    """
    all_metrics = []

    for seq_path in sequences:
        try:
            metrics = test_single_sequence(
                model=model,
                model_type=model_type,
                sequence_path=seq_path,
                image_height=image_height,
                image_width=image_width,
                device=device,
                output_dir=output_dir,
                show_plots=show_plots,
            )
            metrics["sequence_id"] = seq_path.name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nERROR evaluating {seq_path.name}: {e}")
            continue

    # Generate summary
    if all_metrics:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SEQUENCES")
        print("=" * 80)

        # Create summary table
        print(
            f"\n{'Sequence':<15} {'ATE RMSE':<12} {'ATE Mean':<12} {'RPE Trans':<12} {'RPE Rot (deg)':<15}"
        )
        print("-" * 80)

        for m in all_metrics:
            print(
                f"{m['sequence_id']:<15} "
                f"{m['ate_rmse']:<12.4f} "
                f"{m['ate_mean']:<12.4f} "
                f"{m['rpe_trans_rmse']:<12.4f} "
                f"{np.rad2deg(m['rpe_rot_rmse']):<15.2f}"
            )

        # Compute aggregate statistics
        ate_rmse_all = [m["ate_rmse"] for m in all_metrics]
        ate_mean_all = [m["ate_mean"] for m in all_metrics]
        rpe_trans_all = [m["rpe_trans_rmse"] for m in all_metrics]
        rpe_rot_all = [m["rpe_rot_rmse"] for m in all_metrics]

        print("-" * 80)
        print(
            f"{'Mean':<15} "
            f"{np.mean(ate_rmse_all):<12.4f} "
            f"{np.mean(ate_mean_all):<12.4f} "
            f"{np.mean(rpe_trans_all):<12.4f} "
            f"{np.rad2deg(np.mean(rpe_rot_all)):<15.2f}"
        )
        print(
            f"{'Std':<15} "
            f"{np.std(ate_rmse_all):<12.4f} "
            f"{np.std(ate_mean_all):<12.4f} "
            f"{np.std(rpe_trans_all):<12.4f} "
            f"{np.rad2deg(np.std(rpe_rot_all)):<15.2f}"
        )
        print("=" * 80)

        # Save summary JSON
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "model_type": model_type.name,
                    "num_sequences": len(all_metrics),
                    "sequences": all_metrics,
                    "aggregate": {
                        "ate_rmse_mean": float(np.mean(ate_rmse_all)),
                        "ate_rmse_std": float(np.std(ate_rmse_all)),
                        "ate_mean_mean": float(np.mean(ate_mean_all)),
                        "rpe_trans_rmse_mean": float(np.mean(rpe_trans_all)),
                        "rpe_rot_rmse_mean": float(np.mean(rpe_rot_all)),
                    },
                },
                f,
                indent=2,
            )
        print(f"\nSaved summary: {summary_path}")


def parse_args():
    parser = ArgumentParser(description="Test DL-based odometry models")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (contains train/val/test subdirectories or is a split directory)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate on (default: test)",
    )

    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Specific sequence ID to evaluate (e.g., seq_000041). If not provided, evaluates all sequences in split.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Default: ../Output/Testing/{MODEL_TYPE}/",
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=240,
        help="Image height used during training (default: 240)",
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="Image width used during training (default: 320)",
    )

    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument("-v", action="store_true", help="Test Visual Model")
    type_group.add_argument("-i", action="store_true", help="Test Inertial Model")
    type_group.add_argument(
        "-vi", action="store_true", help="Test Visual-Inertial Model"
    )

    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu/mps). Default: auto-detect",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine model type
    if args.v:
        model_type = ModelTypes.VISUAL
    elif args.i:
        model_type = ModelTypes.INERTIAL
    else:
        model_type = ModelTypes.VISUAL_INERTIAL

    # Determine device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = current_dir.parent / "Output" / "Testing" / model_type.name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading {model_type.name} model from {args.checkpoint}...")
    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        model_type=model_type,
        device=device,
    )

    # Get sequences to evaluate
    sequences = get_test_sequences(Path(args.data_dir), args.split, args.sequence_id)
    print(f"\nFound {len(sequences)} sequence(s) to evaluate on '{args.split}' split")

    # Run testing
    if len(sequences) == 1:
        # Single sequence
        test_single_sequence(
            model=model,
            model_type=model_type,
            sequence_path=sequences[0],
            image_height=args.image_height,
            image_width=args.image_width,
            device=device,
            output_dir=output_dir,
            show_plots=args.show_plots,
        )
    else:
        # Batch testing
        test_batch(
            model=model,
            model_type=model_type,
            sequences=sequences,
            image_height=args.image_height,
            image_width=args.image_width,
            device=device,
            output_dir=output_dir,
            show_plots=args.show_plots,
        )

    print(f"\nTesting complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
