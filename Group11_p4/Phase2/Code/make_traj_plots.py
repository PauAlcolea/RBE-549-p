import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMMON_CFG = {
    "height": 1.5,
    "laps": 1,
}

DEFAULT_SHAPE_PARAMS = {
    "square": {"side": 2.0},
    "figure8": {"width": 3.0, "length": 2.0, "samples_per_lap": 400},
    "circle": {"radius": 1.0, "samples_per_lap": 360},
    "triangle": {"side": 2.0, "samples_per_lap": 360},
}


def _repeat_closed_loop(base_loop, n_laps):
    """Repeat a closed loop polyline for n_laps without duplicate joins."""
    if n_laps < 1:
        return np.empty((0, 3), dtype=float)
    if base_loop.size == 0:
        return np.empty((0, 3), dtype=float)

    laps = [base_loop]
    for _ in range(1, n_laps):
        laps.append(base_loop[1:])
    return np.vstack(laps)


def build_square_waypoints(side, height, n_laps):
    """Build a centered square closed loop at a fixed altitude."""
    h = side / 2.0
    corners = np.array(
        [
            (-h, -h, height),
            (h, -h, height),
            (h, h, height),
            (-h, h, height),
        ],
        dtype=float,
    )
    base_loop = np.vstack((corners, corners[0]))
    return _repeat_closed_loop(base_loop, n_laps)


def build_figure8_waypoints(width, length, height, n_laps, samples_per_lap):
    """Build a centered figure-8 using x=a*sin(t), y=b*sin(t)*cos(t)."""
    samples_per_lap = max(8, int(samples_per_lap))
    a = width / 2.0
    b = length / 2.0
    t = np.linspace(0.0, 2.0 * math.pi, samples_per_lap, endpoint=False)

    x = a * np.sin(t)
    y = b * np.sin(t) * np.cos(t)
    z = np.full_like(x, height)
    base_loop = np.column_stack((x, y, z))
    base_loop = np.vstack((base_loop, base_loop[0]))
    return _repeat_closed_loop(base_loop, n_laps)


def build_circle_waypoints(radius, height, n_laps, samples_per_lap):
    """Build a centered circular closed loop sampled uniformly in angle."""
    samples_per_lap = max(8, int(samples_per_lap))
    t = np.linspace(0.0, 2.0 * math.pi, samples_per_lap, endpoint=False)

    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.full_like(x, height)
    base_loop = np.column_stack((x, y, z))
    base_loop = np.vstack((base_loop, base_loop[0]))
    return _repeat_closed_loop(base_loop, n_laps)


def build_triangle_waypoints(side, height, n_laps, samples_per_lap):
    """Build a centered equilateral triangle with dense edge sampling."""
    samples_per_lap = max(9, int(samples_per_lap))
    samples_per_edge = max(1, samples_per_lap // 3)

    h = side / 2.0
    r = side * math.sin(math.radians(60.0))
    corners = np.array(
        [
            (-h, -r / 3.0, height),
            (0.0, 2.0 * r / 3.0, height),
            (h, -r / 3.0, height),
        ],
        dtype=float,
    )

    base_points = []
    for i in range(3):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 3]
        ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        edge_points = (1.0 - ts[:, None]) * start_corner + ts[:, None] * end_corner
        base_points.append(edge_points)

    base_loop = np.vstack(base_points)
    base_loop = np.vstack((base_loop, corners[0]))
    return _repeat_closed_loop(base_loop, n_laps)


def build_all_trajectories(common_cfg=None, shape_params=None):
    """Build trajectory arrays for all supported shapes."""
    common_cfg = common_cfg or DEFAULT_COMMON_CFG
    shape_params = shape_params or DEFAULT_SHAPE_PARAMS

    return {
        "square": build_square_waypoints(
            shape_params["square"]["side"],
            common_cfg["height"],
            common_cfg["laps"],
        ),
        "figure8": build_figure8_waypoints(
            shape_params["figure8"]["width"],
            shape_params["figure8"]["length"],
            common_cfg["height"],
            common_cfg["laps"],
            shape_params["figure8"]["samples_per_lap"],
        ),
        "circle": build_circle_waypoints(
            shape_params["circle"]["radius"],
            common_cfg["height"],
            common_cfg["laps"],
            shape_params["circle"]["samples_per_lap"],
        ),
        "triangle": build_triangle_waypoints(
            shape_params["triangle"]["side"],
            common_cfg["height"],
            common_cfg["laps"],
            shape_params["triangle"]["samples_per_lap"],
        ),
    }


def _set_equal_xy_aspect(ax, trajectory):
    """Keep axes proportionate so shape geometry is not visually distorted."""
    mins = trajectory.min(axis=0)
    maxs = trajectory.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    ax.set_box_aspect((span[0], span[1], span[2]))


def _hide_z_axis_labels(ax):
    """Hide Z axis title and tick labels to reduce clutter in flat trajectories."""
    ax.set_zlabel("")
    ax.set_zticks([])


def plot_trajectory_3d(trajectory, title="Trajectory", save_path=None, dpi=300):
    """Plot a single 3D trajectory and optionally save it."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape (N, 3)")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b-", linewidth=2, label="Path")
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color="green", s=90, label="Start", marker="o")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color="red", s=90, label="End", marker="s")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=2, y=0.95)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _set_equal_xy_aspect(ax, trajectory)
    _hide_z_axis_labels(ax)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_all_trajectories(trajectories_dict, save_path=None, dpi=300):
    """Plot all trajectories on a 2x2 grid and optionally save it."""
    fig = plt.figure(figsize=(14, 12))

    for idx, (shape_name, trajectory) in enumerate(trajectories_dict.items(), 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b-", linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color="green", s=80, marker="o")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color="red", s=80, marker="s")

        # ax.set_title(shape_name.capitalize(), fontsize=12, fontweight="bold", pad=2, y=0.95)
        ax.grid(True, alpha=0.3)
        _set_equal_xy_aspect(ax, trajectory)
        _hide_z_axis_labels(ax)

    plt.tight_layout(h_pad=0.2)


    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D trajectory plots for all supported drone paths.")
    parser.add_argument("--save-dir", type=Path, default=Path("plots"), help="Directory to save generated plots.")
    parser.add_argument("--height", type=float, default=DEFAULT_COMMON_CFG["height"], help="Trajectory altitude in meters.")
    parser.add_argument("--laps", type=int, default=DEFAULT_COMMON_CFG["laps"], help="Number of closed-loop laps.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional shared samples-per-lap override for figure8/circle/triangle.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display and only save plots.")
    return parser.parse_args()


def main():
    args = _parse_args()

    common_cfg = {"height": args.height, "laps": max(1, args.laps)}
    shape_params = {
        shape: dict(params) for shape, params in DEFAULT_SHAPE_PARAMS.items()
    }
    if args.samples is not None:
        shared_samples = max(8, args.samples)
        shape_params["figure8"]["samples_per_lap"] = shared_samples
        shape_params["circle"]["samples_per_lap"] = shared_samples
        shape_params["triangle"]["samples_per_lap"] = max(9, shared_samples)

    trajectories = build_all_trajectories(common_cfg, shape_params)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_all_trajectories(
        trajectories,
        save_path=args.save_dir / "all_trajectories.png",
        dpi=args.dpi,
    )

    for shape_name, trajectory in trajectories.items():
        plot_trajectory_3d(
            trajectory,
            title=f"{shape_name.capitalize()} trajectory",
            save_path=args.save_dir / f"{shape_name}_trajectory.png",
            dpi=args.dpi,
        )

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()