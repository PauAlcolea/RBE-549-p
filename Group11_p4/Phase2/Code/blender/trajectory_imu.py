#!/usr/bin/env python3
"""
Standalone trajectory/pose generator for IMU pipelines.

This script intentionally avoids Blender/rendering. It generates smooth trajectory
poses directly and writes IMU-only files in a sequence folder.

Output columns:
    frame, tx, ty, tz, qw, qx, qy, qz
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class CommonCfg:
    height: float
    laps: int
    speed: float
    sim_hz: int


@dataclass
class ShapeCfg:
    shape: str
    side: float = 2.0
    radius: float = 1.0
    width: float = 3.0
    length: float = 2.0


IMG_WIDTH = 640
IMG_HEIGHT = 480
FOCAL_MM = 50.0
SENSOR_WIDTH_MM = 36.0
ALLOWED_SPLITS = {"train", "val", "test"}


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion for roll=pitch=0 and yaw around +Z."""
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def polyline_length(points: Sequence[Tuple[float, float, float]]) -> float:
    s = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        s += math.sqrt(dx * dx + dy * dy + dz * dz)
    return s


def resample_polyline_constant_speed(
    points: Sequence[Tuple[float, float, float]],
    speed: float,
    hz: int,
) -> List[Tuple[float, float, float]]:
    if len(points) < 2:
        return list(points)

    lengths = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        lengths.append(lengths[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))

    total_len = lengths[-1]
    if total_len < 1e-12:
        return [points[0], points[-1]]

    total_time = total_len / speed
    n_steps = max(2, int(round(total_time * hz)))
    target_s = [total_len * (k / n_steps) for k in range(n_steps + 1)]

    out: List[Tuple[float, float, float]] = []
    seg = 0
    for s in target_s:
        while seg < len(lengths) - 2 and lengths[seg + 1] < s:
            seg += 1
        s0 = lengths[seg]
        s1 = lengths[seg + 1]
        p0 = points[seg]
        p1 = points[seg + 1]
        if s1 - s0 < 1e-12:
            out.append(p0)
            continue
        t = (s - s0) / (s1 - s0)
        out.append(
            (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            )
        )
    return out


def sample_circle(common: CommonCfg, cfg: ShapeCfg) -> List[Tuple[float, float, float]]:
    total_len = 2.0 * math.pi * cfg.radius * common.laps
    total_time = total_len / common.speed
    n_steps = max(2, int(round(total_time * common.sim_hz)))
    # omega = common.speed / cfg.radius

    pts = []
    for k in range(n_steps + 1):
        # t = k / common.sim_hz
        # th = omega * t
        # Parameterize by mornalized index so that the final sample closes exactly
        th = 2.0 * math.pi * common.laps * (k / n_steps)
        pts.append((cfg.radius * math.cos(th), cfg.radius * math.sin(th), common.height))
    return pts


def sample_figure8(common: CommonCfg, cfg: ShapeCfg) -> List[Tuple[float, float, float]]:
    # Parametric figure-8 then arc-length resample for near-constant speed.
    a = cfg.width / 2.0
    b = cfg.length / 2.0
    dense_per_lap = 3000
    n = dense_per_lap * common.laps

    dense: List[Tuple[float, float, float]] = []
    for i in range(n + 1):
        u = i / n
        t = 2.0 * math.pi * common.laps * u
        x = a * math.sin(t)
        y = b * math.sin(t) * math.cos(t)
        dense.append((x, y, common.height))

    return resample_polyline_constant_speed(dense, common.speed, common.sim_hz)


def sample_square(common: CommonCfg, cfg: ShapeCfg) -> List[Tuple[float, float, float]]:
    h = cfg.side / 2.0
    corners = [(-h, -h, common.height), (h, -h, common.height), (h, h, common.height), (-h, h, common.height)]

    loop: List[Tuple[float, float, float]] = []
    for _ in range(common.laps):
        loop.extend(corners)
    loop.append(corners[0])

    return resample_polyline_constant_speed(loop, common.speed, common.sim_hz)


def sample_triangle(common: CommonCfg, cfg: ShapeCfg) -> List[Tuple[float, float, float]]:
    """Sample an equilateral triangle trajectory."""
    h = cfg.side / 2.0
    r = cfg.side * math.sin(math.radians(60))  # radius of circumscribed circle
    corners = [
        (-h, -r / 3, common.height),  # bottom-left
        (0, 2 * r / 3, common.height),  # top
        (h, -r / 3, common.height),  # bottom-right
    ]

    loop: List[Tuple[float, float, float]] = []
    for _ in range(common.laps):
        loop.extend(corners)
    loop.append(corners[0])

    return resample_polyline_constant_speed(loop, common.speed, common.sim_hz)


def compute_tangent_yaws(
    positions: Sequence[Tuple[float, float, float]],
    yaw_offset_rad: float,
) -> List[float]:
    yaws: List[float] = []
    prev = yaw_offset_rad
    n = len(positions)
    for i in range(n):
        if n == 1:
            dx, dy = 0.0, 0.0
        elif i == 0:
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
        elif i == n - 1:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx = positions[i + 1][0] - positions[i - 1][0]
            dy = positions[i + 1][1] - positions[i - 1][1]

        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            yaw = prev
        else:
            yaw = math.atan2(dy, dx) + yaw_offset_rad
        yaws.append(yaw)
        prev = yaw
    return yaws


def generate_positions(common: CommonCfg, cfg: ShapeCfg) -> List[Tuple[float, float, float]]:
    if cfg.shape == "circle":
        return sample_circle(common, cfg)
    if cfg.shape == "figure8":
        return sample_figure8(common, cfg)
    if cfg.shape == "square":
        return sample_square(common, cfg)
    if cfg.shape == "triangle":
        return sample_triangle(common, cfg)
    raise ValueError(f"Unsupported shape '{cfg.shape}'")


def write_poses_csv(
    out_csv: Path,
    positions: Sequence[Tuple[float, float, float]],
    yaws: Sequence[float],
) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])
        for i, p in enumerate(positions, start=1):
            q = yaw_to_quat(yaws[i - 1])
            w.writerow(
                [
                    i,
                    f"{p[0]:.10f}",
                    f"{p[1]:.10f}",
                    f"{p[2]:.10f}",
                    f"{q[0]:.10f}",
                    f"{q[1]:.10f}",
                    f"{q[2]:.10f}",
                    f"{q[3]:.10f}",
                ]
            )


def save_trajectory_plot(
    out_png: Path,
    positions: Sequence[Tuple[float, float, float]],
    seq_id: str,
    shape: str,
) -> None:
    if not positions:
        return

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=2.0, color="#1f77b4", label="trajectory")

    ax.set_title(f"3D Trajectory - {seq_id} ({shape})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_trajectory_summary(
    path: Path,
    common: CommonCfg,
    cfg: ShapeCfg,
    n_frames: int,
    seq_id: str,
    split: str,
    texture: str,
    seed: str,
) -> None:
    meta = {
        "shape": cfg.shape,
        "height": common.height,
        "laps": common.laps,
        "speed": common.speed,
        "sim_hz": common.sim_hz,
        "side": cfg.side,
        "radius": cfg.radius,
        "width": cfg.width,
        "length": cfg.length,
        "split": split,
        "texture": texture,
        "sequence_id": seq_id,
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "frames": n_frames,
        "duration_s": n_frames / common.sim_hz,
    }

    with path.open("w") as f:
        f.write("=== Drone Trajectory Summary ===\n\n")
        f.write(f"  Shape            : {cfg.shape}\n")
        if cfg.shape == "square":
            f.write(f"  Square side      : {cfg.side} m\n")
        elif cfg.shape == "triangle":
            f.write(f"  Triangle side    : {cfg.side} m\n")
        elif cfg.shape == "figure8":
            f.write(f"  Figure8 width    : {cfg.width} m\n")
            f.write(f"  Figure8 length   : {cfg.length} m\n")
        elif cfg.shape == "circle":
            f.write(f"  Circle radius    : {cfg.radius} m\n")
        f.write(f"  Camera height    : {common.height} m\n")
        f.write(f"  Number of laps   : {common.laps}\n")
        f.write(f"  Drone speed      : {common.speed} m/s\n")
        f.write(f"  Sim rate (SIM_HZ): {common.sim_hz} Hz\n")
        f.write(f"  Total frames     : {n_frames}\n")
        f.write(f"  Total duration   : {n_frames / common.sim_hz:.2f} s\n\n")
        f.write("Effective parameters (JSON):\n")
        f.write(json.dumps(meta, indent=2) + "\n")


def resolve_output_dir(data_root: Path, split: str, seq_id: str) -> Path:
    split_dir = data_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out = split_dir / seq_id
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {out}")
        return out
    out.mkdir(parents=False, exist_ok=False)
    return out


def write_camera_k(path: Path) -> None:
    fx = (FOCAL_MM / SENSOR_WIDTH_MM) * IMG_WIDTH
    fy = fx
    cx = IMG_WIDTH / 2.0
    cy = IMG_HEIGHT / 2.0

    k = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]

    with path.open("w") as f:
        f.write(f"# Intrinsic matrix K  (image size: {IMG_WIDTH} x {IMG_HEIGHT})\n")
        f.write("# fx  0   cx\n")
        f.write("# 0   fy  cy\n")
        f.write("# 0   0   1\n\n")
        for row in k:
            f.write("  ".join(f"{v:12.6f}" for v in row) + "\n")
        f.write(f"\nfx = {fx:.6f}\n")
        f.write(f"fy = {fy:.6f}\n")
        f.write(f"cx = {cx:.6f}\n")
        f.write(f"cy = {cy:.6f}\n")


def write_sequence_metadata(output_dir: Path, metadata: Dict) -> None:
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def append_dataset_manifest(base_output_dir: Path, row: Dict[str, str]) -> None:
    manifest_path = base_output_dir / "index.csv"
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    fieldnames = [
        "sequence_id",
        "split",
        "shape",
        "texture",
        "height_m",
        "speed_mps",
        "laps",
        "sim_hz",
        "num_frames",
        "duration_s",
        "seed",
        "rel_path",
        "timestamp_utc",
    ]

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IMU trajectory sequences with the same dataset structure as "
            "visual generation (except frames/ rendering)."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "Data" / "Generated")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--seq-id", type=str, required=True, help="Sequence folder name, e.g. seq_000123")
    parser.add_argument("--texture", type=str, default="playrug", help="Texture tag for metadata/manifest consistency")
    parser.add_argument("--seed", type=str, default="", help="Optional seed tag for metadata/manifest consistency")

    parser.add_argument("--shape", type=str, default="square", choices=["square", "figure8", "circle", "triangle"])
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--sim-hz", type=int, default=100)

    parser.add_argument("--side", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--length", type=float, default=2.0)

    parser.add_argument("--heading-mode", type=str, default="tangent", choices=["tangent", "fixed"])
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Yaw offset in degrees")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.height <= 0.0:
        raise ValueError("--height must be > 0")
    if args.laps < 1:
        raise ValueError("--laps must be >= 1")
    if args.speed <= 0.0:
        raise ValueError("--speed must be > 0")
    if args.sim_hz <= 0:
        raise ValueError("--sim-hz must be > 0")

    common = CommonCfg(height=args.height, laps=args.laps, speed=args.speed, sim_hz=args.sim_hz)
    cfg = ShapeCfg(shape=args.shape, side=args.side, radius=args.radius, width=args.width, length=args.length)

    if cfg.shape == "square" and cfg.side <= 0.0:
        raise ValueError("--side must be > 0 for square")
    if cfg.shape == "triangle" and cfg.side <= 0.0:
        raise ValueError("--side must be > 0 for triangle")
    if cfg.shape == "circle" and cfg.radius <= 0.0:
        raise ValueError("--radius must be > 0 for circle")
    if cfg.shape == "figure8" and (cfg.width <= 0.0 or cfg.length <= 0.0):
        raise ValueError("--width and --length must be > 0 for figure8")

    split = str(args.split).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"--split must be one of {sorted(ALLOWED_SPLITS)}")

    out_dir = resolve_output_dir(args.data_root, split, args.seq_id)

    positions = generate_positions(common, cfg)
    yaw_offset_rad = math.radians(args.yaw_deg)
    if args.heading_mode == "tangent":
        yaws = compute_tangent_yaws(positions, yaw_offset_rad)
    else:
        yaws = [yaw_offset_rad] * len(positions)

    write_poses_csv(out_dir / "poses.csv", positions, yaws)
    save_trajectory_plot(out_dir / "trajectory_3d.png", positions, args.seq_id, cfg.shape)
    write_camera_k(out_dir / "camera_K.txt")
    write_trajectory_summary(
        out_dir / "trajectory.txt",
        common,
        cfg,
        len(positions),
        args.seq_id,
        split=split,
        texture=args.texture,
        seed=str(args.seed).strip(),
    )

    rel_path = str((Path(split) / args.seq_id).as_posix())
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    metadata = {
        "sequence_id": args.seq_id,
        "split": split,
        "shape": cfg.shape,
        "texture": args.texture,
        "height_m": common.height,
        "speed_mps": common.speed,
        "laps": common.laps,
        "sim_hz": common.sim_hz,
        "num_frames": len(positions),
        "duration_s": len(positions) / common.sim_hz,
        "seed": str(args.seed).strip(),
        "blend_file": "trajectory_imu.py",
        "output_dir": str(out_dir.resolve()),
        "relative_output_dir": rel_path,
        "trajectory_parameters": {
            "shape": cfg.shape,
            "height": common.height,
            "laps": common.laps,
            "speed": common.speed,
            "sim_hz": common.sim_hz,
            "side": cfg.side,
            "radius": cfg.radius,
            "width": cfg.width,
            "length": cfg.length,
            "texture": args.texture,
            "split": split,
            "sequence_id": args.seq_id,
            "seed": str(args.seed).strip(),
        },
        "generated_utc": generated_utc,
    }
    write_sequence_metadata(out_dir, metadata)
    append_dataset_manifest(
        args.data_root,
        {
            "sequence_id": args.seq_id,
            "split": split,
            "shape": cfg.shape,
            "texture": args.texture,
            "height_m": f"{common.height:.6f}",
            "speed_mps": f"{common.speed:.6f}",
            "laps": str(common.laps),
            "sim_hz": str(common.sim_hz),
            "num_frames": str(len(positions)),
            "duration_s": f"{len(positions) / common.sim_hz:.6f}",
            "seed": str(args.seed).strip(),
            "rel_path": rel_path,
            "timestamp_utc": generated_utc,
        },
    )

    print(f"[OK] Generated sequence at: {out_dir}")
    print(f"[OK] poses.csv frames: {len(positions)}")
    print(f"[OK] metadata.json + index.csv updated")


if __name__ == "__main__":
    main()
