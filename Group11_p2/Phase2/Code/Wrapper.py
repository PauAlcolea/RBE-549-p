import argparse
import torch
import numpy as np
from pathlib import Path
import imageio

from Train import train
from Dataset import NeRFDataset
from NeRFModel import NeRFmodel


def render(model, rays_origin, rays_direction, device, chunk_size=4096):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    num_rays = rays_origin.shape[0]
    pred_chunks = []

    model.eval()
    with torch.no_grad():
        for start in range(0, num_rays, chunk_size):
            end = min(start + chunk_size, num_rays)
            ro_chunk = rays_origin[start:end].to(device)
            rd_chunk = rays_direction[start:end].to(device)
            _, pred_rgb_fine = model(ro_chunk, rd_chunk)
            pred_chunks.append(pred_rgb_fine.detach().cpu())

    return torch.cat(pred_chunks, dim=0)


def _look_at_pose(camera_position, target, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = camera_position - target
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position
    return torch.from_numpy(pose)


def _estimate_focus_point(camera_positions, forward_dirs):
    """
    Estimate scene focus point as the least-squares closest point to all camera rays.
    """
    I = np.eye(3, dtype=np.float32)
    A = np.zeros((3, 3), dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)

    for p, d in zip(camera_positions, forward_dirs):
        d = d / (np.linalg.norm(d) + 1e-8)
        M = I - np.outer(d, d)
        A += M
        b += M @ p

    center, *_ = np.linalg.lstsq(A, b, rcond=None)
    return center.astype(np.float32)


def _resolve_orbit_axis(cam_positions):
    stds = cam_positions.std(axis=0)
    idx = int(np.argmin(stds))
    axis = np.zeros(3, dtype=np.float32)
    axis[idx] = 1.0
    return axis, idx


def generate_orbit_poses(poses, num_frames):
    poses_np = poses.cpu().numpy()
    cam_positions = poses_np[:, :3, 3]

    # In this convention, camera forward is -Z axis of camera-to-world matrix.
    forward_dirs = -poses_np[:, :3, 2]
    center = _estimate_focus_point(cam_positions, forward_dirs)

    # Force a rigid circular trajectory around exactly one axis.
    orbit_axis, axis_idx = _resolve_orbit_axis(cam_positions)
    offsets = cam_positions - center[None, :]

    # Radius is measured in the two coordinates orthogonal to the orbit axis.
    plane_idx = [i for i in range(3) if i != axis_idx]
    radius = np.median(np.linalg.norm(offsets[:, plane_idx], axis=1))
    if radius < 1e-6:
        radius = 4.0

    # Keep the axis coordinate fixed for all frames to avoid tilt/axis drift.
    axis_level = np.median(cam_positions[:, axis_idx])

    # Preserve the starting viewpoint direction by initializing the orbit
    # from the first camera's projection in the orbit plane.
    start_vec = cam_positions[0] - center
    start_angle = np.arctan2(start_vec[plane_idx[1]], start_vec[plane_idx[0]])

    frames = []
    angles = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False) + start_angle
    for theta in angles:
        cam_pos = center.copy().astype(np.float32)
        cam_pos[axis_idx] = axis_level
        cam_pos[plane_idx[0]] = center[plane_idx[0]] + radius * np.cos(theta)
        cam_pos[plane_idx[1]] = center[plane_idx[1]] + radius * np.sin(theta)
        frames.append(
            _look_at_pose(cam_pos, center.astype(np.float32), world_up=orbit_axis)
        )

    return torch.stack(frames, dim=0)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="lego",
        choices=["lego", "ship"],
        help="dataset to train on: lego or ship",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="whether to run test (default is Train)",
    )
    parser.add_argument(
        "--down",
        type=int,
        default=1,
        help="how much you want to downscale the images so training takes less time",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="number of frames to render for the gif",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="frames per second for the output gif",
    )

    args = parser.parse_args()
    return args


def main(args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    top_data_dir = Path(__file__).parent.parent / "Data" / "nerf_synthetic"
    dataset_dir = top_data_dir / args.dataset
    code_dir = Path(__file__).parent

    if args.test:
        model = NeRFmodel().to(device)

        checkpoint_path = code_dir / "checkpoints" / args.dataset / "best_model.pth"
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        test_dataset = NeRFDataset(
            dataset_dir / "test",
            batch_size=4096,
            device=device,
            downscale=4,
        )

        orbit_poses = generate_orbit_poses(
            test_dataset.poses,
            args.frames,
        )

        gif_dir = code_dir / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"{args.dataset}.gif"

        frames = []
        for pose in orbit_poses:
            pose = pose.to(test_dataset.ray_directions.device)
            R, t = pose[:3, :3], pose[:3, 3]

            ray_directions_world = test_dataset.ray_directions @ R.T
            ray_origins_world = t.view(1, 1, 3).expand_as(ray_directions_world)

            rgb_flat = render(
                model,
                ray_origins_world.reshape(-1, 3),
                ray_directions_world.reshape(-1, 3),
                device,
            )

            frame = rgb_flat.view(test_dataset.h, test_dataset.w, 3).numpy()
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).astype(np.uint8)
            frames.append(frame)

        imageio.mimsave(gif_path, frames, fps=args.fps)
        print(f"Saved gif to: {gif_path}")
    else:
        train(
            train_data_dir=dataset_dir / "train",
            val_data_dir=dataset_dir / "val",
            device=device,
            downscale=args.down,
            dataset_name=args.dataset,
        )


if __name__ == "__main__":
    main(parseArgs())
