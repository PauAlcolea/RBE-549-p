#!/usr/bin/env python3
"""
Batch generate visual-inertial dataset: upsample trajectories and add IMU data.

This script orchestrates the full VI dataset generation pipeline:
1. Upsample existing 100 Hz poses to 1000 Hz (upsample_trajectory.py)
2. Generate IMU measurements from 1000 Hz poses (generate_imu.py)

Processes all sequences in specified splits (train/val/test) under Data/Generated.

Usage:
    python generate_vi_dataset.py --data-root Data/Generated --splits train val test
    python generate_vi_dataset.py --data-root Data/Generated --splits train --resume
    python generate_vi_dataset.py --data-root Data/Generated --all-splits --noise-profile high
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Set


def find_sequences(data_root: Path, splits: List[str]) -> List[Path]:
    """
    Find all sequence directories in specified splits.
    
    Args:
        data_root: Path to Generated/ directory
        splits: List of split names (train, val, test)
    
    Returns:
        List of sequence directory paths
    """
    sequences = []
    
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[WARNING] Split directory not found: {split_dir}")
            continue
        
        # Find all subdirectories that look like sequences (contain poses.csv)
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            poses_csv = seq_dir / "poses.csv"
            if poses_csv.exists():
                sequences.append(seq_dir)
    
    return sequences


def check_sequence_status(seq_dir: Path, target_hz: float = 1000.0) -> dict:
    """
    Check which files exist for a sequence.
    
    Returns:
        dict with boolean flags: has_poses, has_poses_upsampled, has_imu
    """
    poses_csv = seq_dir / "poses.csv"
    poses_1000hz_csv = seq_dir / "poses_1000hz.csv"
    
    # IMU files with hz suffix
    hz_suffix = f"_{int(target_hz)}hz" if target_hz != 100 else ""
    imu_gt = seq_dir / f"imu_gt{hz_suffix}.csv"
    imu_noisy = seq_dir / f"{seq_dir.name}_imu{hz_suffix}.csv"
    
    return {
        "has_poses": poses_csv.exists(),
        "has_poses_upsampled": poses_1000hz_csv.exists(),
        "has_imu": imu_gt.exists() and imu_noisy.exists(),
    }


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def rotation_label_stats(seq_dir: Path, small_rot_thresh_rad: float = 1e-4) -> dict:
    """
    Compute rotation-label quality stats from consecutive pose quaternions.

    Returns:
        dict with keys: valid, num_pairs, pct_small_rot, median_rot_deg, max_rot_deg
    """
    poses_csv = seq_dir / "poses.csv"
    if not poses_csv.exists():
        return {
            "valid": False,
            "num_pairs": 0,
            "pct_small_rot": 1.0,
            "median_rot_deg": 0.0,
            "max_rot_deg": 0.0,
        }

    quats: list[tuple[float, float, float, float]] = []
    with poses_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
            )
            quats.append(_quat_normalize(q))

    if len(quats) < 2:
        return {
            "valid": False,
            "num_pairs": 0,
            "pct_small_rot": 1.0,
            "median_rot_deg": 0.0,
            "max_rot_deg": 0.0,
        }

    rot_angles_rad: list[float] = []
    small_count = 0
    for i in range(len(quats) - 1):
        q0 = quats[i]
        q1 = quats[i + 1]
        dot = abs(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3])
        dot = max(-1.0, min(1.0, dot))
        angle = 2.0 * math.acos(dot)
        rot_angles_rad.append(angle)
        if angle < small_rot_thresh_rad:
            small_count += 1

    rot_angles_rad.sort()
    mid = len(rot_angles_rad) // 2
    if len(rot_angles_rad) % 2 == 0:
        median_rot = 0.5 * (rot_angles_rad[mid - 1] + rot_angles_rad[mid])
    else:
        median_rot = rot_angles_rad[mid]

    return {
        "valid": True,
        "num_pairs": len(rot_angles_rad),
        "pct_small_rot": small_count / len(rot_angles_rad),
        "median_rot_deg": math.degrees(median_rot),
        "max_rot_deg": math.degrees(max(rot_angles_rad)),
    }


def run_upsample_trajectory(
    seq_dir: Path,
    original_hz: float = 100.0,
    target_hz: float = 1000.0,
    validate: bool = True,
) -> bool:
    """
    Run upsample_trajectory.py on a sequence.
    
    Returns:
        True if successful, False otherwise
    """
    script_dir = Path(__file__).parent
    upsample_script = script_dir / "upsample_trajectory.py"
    
    if not upsample_script.exists():
        print(f"[ERROR] upsample_trajectory.py not found at {upsample_script}")
        return False
    
    cmd = [
        sys.executable,
        str(upsample_script),
        "--sequence-dir",
        str(seq_dir),
        "--original-hz",
        str(original_hz),
        "--target-hz",
        str(target_hz),
    ]
    
    if validate:
        cmd.append("--validate")
    else:
        cmd.append("--no-validate")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Print output for visibility
        if result.stdout:
            print(result.stdout.rstrip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Upsampling failed for {seq_dir.name}")
        if e.stderr:
            print(e.stderr)
        return False


def run_generate_imu(
    data_root: Path,
    hz: float = 1000.0,
    noise_profile: str = "mid",
    noise_seed: int | None = None,
) -> bool:
    """
    Run generate_imu.py on all sequences in data_root.
    
    Returns:
        True if successful, False otherwise
    """
    script_dir = Path(__file__).parent
    imu_script = script_dir / "generate_imu.py"
    
    if not imu_script.exists():
        print(f"[ERROR] generate_imu.py not found at {imu_script}")
        return False
    
    cmd = [
        sys.executable,
        str(imu_script),
        "--data-root",
        str(data_root),
        "--hz",
        str(hz),
        "--noise-profile",
        noise_profile,
    ]
    
    if noise_seed is not None:
        cmd.extend(["--noise-seed", str(noise_seed)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Print output for visibility
        if result.stdout:
            print(result.stdout.rstrip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] IMU generation failed")
        if e.stderr:
            print(e.stderr)
        return False


def process_sequences(
    sequences: List[Path],
    original_hz: float = 100.0,
    target_hz: float = 1000.0,
    noise_profile: str = "mid",
    noise_seed: int | None = None,
    resume: bool = False,
    validate: bool = True,
) -> None:
    """
    Process all sequences: upsample + generate IMU.
    
    Args:
        sequences: List of sequence directories
        original_hz: Original sampling rate
        target_hz: Target sampling rate
        noise_profile: IMU noise profile (low/mid/high)
        noise_seed: Random seed for noise generation
        resume: Skip sequences that already have all outputs
        validate: Validate upsampling accuracy
    """
    total = len(sequences)
    skipped = 0
    failed_upsample = []
    successful_upsample = []
    degenerate_rotation = []
    
    print(f"\n[VI Dataset Generation]")
    print(f"  Sequences: {total}")
    print(f"  Original: {original_hz} Hz → Target: {target_hz} Hz")
    print(f"  Noise: {noise_profile}")
    print(f"  Resume: {resume}")
    print()
    
    # Phase 1: Upsample trajectories
    print("=" * 70)
    print("PHASE 1: Upsampling Trajectories")
    print("=" * 70)
    
    for i, seq_dir in enumerate(sequences, 1):
        print(f"\n[{i}/{total}] {seq_dir.name}")
        
        status = check_sequence_status(seq_dir, target_hz)
        
        if not status["has_poses"]:
            print(f"  [SKIP] No poses.csv found")
            skipped += 1
            continue
        
        if resume and status["has_poses_upsampled"]:
            print(f"  [SKIP] poses_1000hz.csv already exists (resume mode)")
            successful_upsample.append(seq_dir)
            skipped += 1
            continue

        rot_stats = rotation_label_stats(seq_dir)
        if rot_stats["valid"]:
            print(
                "  [ROT] "
                f"median={rot_stats['median_rot_deg']:.6f} deg, "
                f"max={rot_stats['max_rot_deg']:.6f} deg, "
                f"small<{1e-4:.0e}rad={100.0 * rot_stats['pct_small_rot']:.1f}%"
            )
            if rot_stats["pct_small_rot"] >= 0.98:
                print(
                    "  [WARNING] Degenerate rotation supervision detected "
                    "(>=98% near-zero consecutive rotations)."
                )
                degenerate_rotation.append(seq_dir)
        
        # Run upsampling
        success = run_upsample_trajectory(seq_dir, original_hz, target_hz, validate)
        
        if success:
            successful_upsample.append(seq_dir)
        else:
            failed_upsample.append(seq_dir)
    
    print()
    print("=" * 70)
    print(f"Phase 1 Complete: {len(successful_upsample)} upsampled, {len(failed_upsample)} failed, {skipped} skipped")
    print("=" * 70)
    
    if failed_upsample:
        print("\nFailed upsampling:")
        for seq_dir in failed_upsample:
            print(f"  - {seq_dir.name}")
        print()

    if degenerate_rotation:
        print("[WARNING] Sequences with degenerate rotation labels:")
        for seq_dir in degenerate_rotation:
            print(f"  - {seq_dir.name}")
        print(
            "[WARNING] Regenerate poses upstream with non-constant heading "
            "(e.g., tangent heading and/or non-zero heading spin)."
        )
        print()
    
    if not successful_upsample:
        print("\n[WARNING] No sequences were successfully upsampled. Skipping IMU generation.")
        return
    
    # Phase 2: Generate IMU data
    print("\n" + "=" * 70)
    print("PHASE 2: Generating IMU Data")
    print("=" * 70)
    print()
    
    # Run IMU generation once on the data root (it finds all poses_1000hz.csv files)
    data_root = sequences[0].parent.parent
    success = run_generate_imu(data_root, target_hz, noise_profile, noise_seed)
    
    if success:
        print("\n" + "=" * 70)
        print("VI Dataset Generation Complete!")
        print("=" * 70)
    else:
        print("\n[ERROR] IMU generation failed")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate visual-inertial dataset from existing sequences"
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to Generated/ directory containing train/val/test splits",
    )
    
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        help="Process specific splits (e.g., --splits train val)",
    )
    split_group.add_argument(
        "--all-splits",
        action="store_true",
        help="Process all splits (train, val, test)",
    )
    
    parser.add_argument(
        "--original-hz",
        type=float,
        default=100.0,
        help="Original sampling rate (default: 100 Hz)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=1000.0,
        help="Target IMU sampling rate (default: 1000 Hz)",
    )
    parser.add_argument(
        "--noise-profile",
        type=str,
        default="mid",
        choices=["low", "mid", "high"],
        help="IMU noise profile (default: mid)",
    )
    parser.add_argument(
        "--noise-seed", type=int, default=None, help="Random seed for noise generation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sequences that already have upsampled poses and IMU data",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of upsampling accuracy",
    )
    
    args = parser.parse_args()
    
    # Validate data root
    if not args.data_root.exists():
        print(f"[ERROR] Data root not found: {args.data_root}")
        sys.exit(1)
    
    # Determine splits to process
    if args.all_splits:
        splits = ["train", "val", "test"]
    else:
        splits = args.splits
    
    # Find sequences
    sequences = find_sequences(args.data_root, splits)
    
    if not sequences:
        print(f"[ERROR] No sequences found in splits: {splits}")
        sys.exit(1)
    
    # Process sequences
    process_sequences(
        sequences,
        original_hz=args.original_hz,
        target_hz=args.target_hz,
        noise_profile=args.noise_profile,
        noise_seed=args.noise_seed,
        resume=args.resume,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
