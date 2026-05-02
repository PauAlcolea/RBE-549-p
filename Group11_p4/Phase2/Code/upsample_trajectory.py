#!/usr/bin/env python3
"""
Upsample existing 100 Hz trajectory poses to 1000 Hz for IMU generation.

This script interpolates pose data using cubic splines for position and SLERP
for quaternion orientation. The result aligns perfectly at every 10th sample
with the original 100 Hz data.

Usage:
    python upsample_trajectory.py --poses-csv path/to/poses.csv [--validate]
    python upsample_trajectory.py --sequence-dir path/to/seq_000001 [--validate]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


def read_poses_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read poses.csv and return frames, positions, and quaternions.

    Returns:
        frames: (N,) array of frame numbers
        positions: (N, 3) array of [x, y, z]
        quaternions: (N, 4) array of [qw, qx, qy, qz]
    """
    frames = []
    positions = []
    quaternions = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
        if not all(col in reader.fieldnames for col in required):
            raise ValueError(f"Missing required columns in {csv_path}")

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

    return (
        np.array(frames, dtype=np.int64),
        np.array(positions, dtype=np.float64),
        np.array(quaternions, dtype=np.float64),
    )


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit length."""
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return q / norms


def ensure_quaternion_continuity(q: np.ndarray) -> np.ndarray:
    """
    Flip quaternion signs to ensure continuity along the trajectory.
    
    Quaternions q and -q represent the same rotation, but we need continuity
    for interpolation. This function ensures each quaternion has the same sign
    convention as its predecessor (dot product > 0).
    """
    q = normalize_quaternions(q)
    q_continuous = q.copy()
    
    for i in range(1, len(q)):
        if np.dot(q_continuous[i - 1], q_continuous[i]) < 0:
            q_continuous[i] = -q_continuous[i]
    
    return q_continuous


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q0: Starting quaternion [qw, qx, qy, qz]
        q1: Ending quaternion [qw, qx, qy, qz]
        t: Interpolation parameter in [0, 1]
    
    Returns:
        Interpolated quaternion
    """
    dot = np.dot(q0, q1)
    
    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # Clamp dot product to valid range
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # Standard SLERP formula
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    
    return w0 * q0 + w1 * q1


def interpolate_quaternions_slerp(
    q_orig: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    """
    Interpolate quaternions using SLERP.
    
    Args:
        q_orig: (N, 4) original quaternions
        t_orig: (N,) original time indices
        t_new: (M,) new time indices to interpolate at
    
    Returns:
        (M, 4) interpolated quaternions
    """
    q_continuous = ensure_quaternion_continuity(q_orig)
    q_interp = np.zeros((len(t_new), 4), dtype=np.float64)
    
    for i, t in enumerate(t_new):
        # Find bracketing indices
        idx_high = np.searchsorted(t_orig, t)
        
        if idx_high == 0:
            q_interp[i] = q_continuous[0]
        elif idx_high >= len(t_orig):
            q_interp[i] = q_continuous[-1]
        else:
            idx_low = idx_high - 1
            t0, t1 = t_orig[idx_low], t_orig[idx_high]
            
            # Interpolation parameter
            if t1 - t0 < 1e-12:
                local_t = 0.0
            else:
                local_t = (t - t0) / (t1 - t0)
            
            q_interp[i] = slerp(q_continuous[idx_low], q_continuous[idx_high], local_t)
    
    return normalize_quaternions(q_interp)


def upsample_poses(
    frames_orig: np.ndarray,
    positions_orig: np.ndarray,
    quaternions_orig: np.ndarray,
    original_hz: float = 100.0,
    target_hz: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Upsample poses from original_hz to target_hz.
    
    Args:
        frames_orig: Original frame numbers
        positions_orig: (N, 3) positions
        quaternions_orig: (N, 4) quaternions
        original_hz: Original sampling rate
        target_hz: Target sampling rate
    
    Returns:
        frames_new: New frame numbers (1-indexed)
        positions_new: Upsampled positions
        quaternions_new: Upsampled quaternions
    """
    if target_hz % original_hz != 0:
        raise ValueError(
            f"target_hz ({target_hz}) must be a multiple of original_hz ({original_hz})"
        )
    
    upsample_factor = int(target_hz / original_hz)
    n_orig = len(frames_orig)
    
    # Time indices for original and upsampled data
    t_orig = np.arange(n_orig, dtype=np.float64)
    n_new = (n_orig - 1) * upsample_factor + 1
    t_new = np.linspace(0, n_orig - 1, n_new)
    
    # Interpolate positions using cubic spline
    cs_x = CubicSpline(t_orig, positions_orig[:, 0])
    cs_y = CubicSpline(t_orig, positions_orig[:, 1])
    cs_z = CubicSpline(t_orig, positions_orig[:, 2])
    
    positions_new = np.column_stack([cs_x(t_new), cs_y(t_new), cs_z(t_new)])
    
    # Interpolate quaternions using SLERP
    quaternions_new = interpolate_quaternions_slerp(quaternions_orig, t_orig, t_new)
    
    # Generate new frame numbers (1-indexed)
    frames_new = np.arange(1, n_new + 1, dtype=np.int64)
    
    return frames_new, positions_new, quaternions_new


def validate_upsampling(
    positions_orig: np.ndarray,
    quaternions_orig: np.ndarray,
    positions_new: np.ndarray,
    quaternions_new: np.ndarray,
    upsample_factor: int,
) -> Tuple[float, float]:
    """
    Validate that every upsample_factor-th sample matches the original.
    
    Returns:
        max_position_error: Maximum position error in meters
        max_quaternion_error: Maximum quaternion error (1 - |dot product|)
    """
    downsampled_positions = positions_new[::upsample_factor]
    downsampled_quaternions = quaternions_new[::upsample_factor]
    
    # Check array lengths match
    min_len = min(len(positions_orig), len(downsampled_positions))
    
    # Position error (Euclidean distance)
    pos_errors = np.linalg.norm(
        positions_orig[:min_len] - downsampled_positions[:min_len], axis=-1
    )
    max_pos_error = np.max(pos_errors)
    
    # Quaternion error (1 - |dot product|, where 0 = perfect match)
    quat_errors = []
    for i in range(min_len):
        q_orig_norm = quaternions_orig[i] / np.linalg.norm(quaternions_orig[i])
        q_down_norm = downsampled_quaternions[i] / np.linalg.norm(
            downsampled_quaternions[i]
        )
        dot = abs(np.dot(q_orig_norm, q_down_norm))
        quat_errors.append(1.0 - min(dot, 1.0))
    
    max_quat_error = max(quat_errors)
    
    return max_pos_error, max_quat_error


def write_poses_csv(
    output_path: Path,
    frames: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    """Write upsampled poses to CSV."""
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])
        
        for i in range(len(frames)):
            writer.writerow(
                [
                    int(frames[i]),
                    f"{positions[i, 0]:.10f}",
                    f"{positions[i, 1]:.10f}",
                    f"{positions[i, 2]:.10f}",
                    f"{quaternions[i, 0]:.10f}",
                    f"{quaternions[i, 1]:.10f}",
                    f"{quaternions[i, 2]:.10f}",
                    f"{quaternions[i, 3]:.10f}",
                ]
            )


def process_sequence(
    poses_csv_path: Path,
    original_hz: float = 100.0,
    target_hz: float = 1000.0,
    validate: bool = True,
) -> None:
    """
    Upsample a single sequence's poses.csv to poses_1000hz.csv.
    
    Args:
        poses_csv_path: Path to input poses.csv
        original_hz: Original sampling rate
        target_hz: Target sampling rate
        validate: Whether to validate interpolation accuracy
    """
    print(f"[Upsampling] {poses_csv_path}")
    
    # Read original poses
    frames_orig, positions_orig, quaternions_orig = read_poses_csv(poses_csv_path)
    n_orig = len(frames_orig)
    
    # Upsample
    frames_new, positions_new, quaternions_new = upsample_poses(
        frames_orig, positions_orig, quaternions_orig, original_hz, target_hz
    )
    
    # Validate if requested
    if validate:
        upsample_factor = int(target_hz / original_hz)
        max_pos_err, max_quat_err = validate_upsampling(
            positions_orig,
            quaternions_orig,
            positions_new,
            quaternions_new,
            upsample_factor,
        )
        
        # Thresholds
        pos_threshold = 1e-6  # meters
        quat_threshold = 1e-4  # quaternion error
        
        if max_pos_err > pos_threshold or max_quat_err > quat_threshold:
            print(
                f"  [WARNING] Validation failed: "
                f"pos_err={max_pos_err:.2e} m (threshold={pos_threshold:.2e}), "
                f"quat_err={max_quat_err:.2e} (threshold={quat_threshold:.2e})"
            )
        else:
            print(
                f"  [VALIDATED] "
                f"pos_err={max_pos_err:.2e} m, quat_err={max_quat_err:.2e}"
            )
    
    # Write output
    output_path = poses_csv_path.parent / "poses_1000hz.csv"
    write_poses_csv(output_path, frames_new, positions_new, quaternions_new)
    
    print(
        f"  [OK] {n_orig} frames @ {original_hz} Hz → "
        f"{len(frames_new)} frames @ {target_hz} Hz"
    )
    print(f"  [OUTPUT] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsample 100 Hz trajectory poses to 1000 Hz for IMU generation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--poses-csv", type=Path, help="Path to poses.csv file to upsample"
    )
    group.add_argument(
        "--sequence-dir",
        type=Path,
        help="Path to sequence directory containing poses.csv",
    )
    
    parser.add_argument(
        "--original-hz", type=float, default=100.0, help="Original sampling rate (Hz)"
    )
    parser.add_argument(
        "--target-hz", type=float, default=1000.0, help="Target sampling rate (Hz)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate interpolation accuracy",
    )
    parser.add_argument(
        "--no-validate", dest="validate", action="store_false", help="Skip validation"
    )
    
    args = parser.parse_args()
    
    # Resolve poses.csv path
    if args.poses_csv:
        poses_csv_path = args.poses_csv
    else:
        poses_csv_path = args.sequence_dir / "poses.csv"
    
    if not poses_csv_path.exists():
        print(f"[ERROR] Poses file not found: {poses_csv_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        process_sequence(
            poses_csv_path, args.original_hz, args.target_hz, args.validate
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
