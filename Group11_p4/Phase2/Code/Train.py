#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")       # allows this to run headless 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

from Datasets import VisualDataset, InertialDataset, VisualInertialDataset
from Models import VisualModel, InertialModel, VisualInertialModel

current_dir = Path(__file__).parent
output_dir = current_dir.parent / "Output" / "Training"


class ModelTypes(Enum):
    VISUAL = 1
    INERTIAL = 2
    VISUAL_INERTIAL = 3


class Pipeline:
    def __init__(
        self,
        model_type,
        train_dir,
        val_dir,
        sequence_length,
        lstm_hidden,
        image_height,
        image_width,
    ):
        self.model_type = model_type

        if model_type == ModelTypes.VISUAL:
            self.train_dataset = VisualDataset(
                train_dir,
                mode="sequences",
                sequence_length=sequence_length,
                image_height=image_height,
                image_width=image_width,
            )
            self.val_dataset = VisualDataset(
                val_dir,
                mode="sequences",
                sequence_length=sequence_length,
                image_height=image_height,
                image_width=image_width,
            )
            self.model = VisualModel(
                lstm_hidden_size=lstm_hidden,
                image_height=image_height,
                image_width=image_width,
            )
        elif model_type == ModelTypes.INERTIAL:
            self.train_dataset = InertialDataset(train_dir)
            self.val_dataset = InertialDataset(val_dir)
            self.model = InertialModel()
        elif model_type == ModelTypes.VISUAL_INERTIAL:
            self.train_dataset = VisualInertialDataset(train_dir)
            self.val_dataset = VisualInertialDataset(val_dir)
            self.model = VisualInertialModel()


def _quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotmat_np(q):
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _relative_poses_to_world_positions(rel_poses):
    """
    Convert relative local-frame poses [dx,dy,dz,qw,qx,qy,qz] into world XYZ path.
    Returns (T, 3) positions, starting at the origin.
    """
    n = rel_poses.shape[0]
    positions = np.zeros((n + 1, 3), dtype=np.float64)
    q_world = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i in range(n):
        d_local = rel_poses[i, :3]
        q_rel = rel_poses[i, 3:]
        q_rel = q_rel / max(np.linalg.norm(q_rel), 1e-12)

        r_world = _quat_to_rotmat_np(q_world)
        d_world = r_world @ d_local
        positions[i + 1] = positions[i] + d_world
        q_world = _quat_mul_np(q_world, q_rel)
        q_world = q_world / max(np.linalg.norm(q_world), 1e-12)

    return positions


def _save_val_trajectory_plot(model, val_loader, device, epoch, val_loss, writer):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            pred = model.forward(batch)
            gt = batch["target_rel_poses"]

            pred_np = pred[0].detach().cpu().numpy()
            gt_np = gt[0].detach().cpu().numpy()
            seq_id = batch.get("sequence_id", ["unknown"])[0]

            pred_xyz = _relative_poses_to_world_positions(pred_np)
            gt_xyz = _relative_poses_to_world_positions(gt_np)

            fig = plt.figure(figsize=(12, 10))

            # 3D trajectory
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
            ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground Truth", linewidth=2.0)
            ax3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="Prediction", linewidth=2.0)
            ax3d.set_title("3D")
            ax3d.set_xlabel("X [m]")
            ax3d.set_ylabel("Y [m]")
            ax3d.set_zlabel("Z [m]")
            ax3d.legend(loc="best")
            ax3d.grid(True)

            # XY plane
            ax_xy = fig.add_subplot(2, 2, 2)
            ax_xy.plot(gt_xyz[:, 0], gt_xyz[:, 1], linewidth=2.0, label="Ground Truth")
            ax_xy.plot(pred_xyz[:, 0], pred_xyz[:, 1], linewidth=2.0, label="Prediction")
            ax_xy.set_title("XY Plane")
            ax_xy.set_xlabel("X [m]")
            ax_xy.set_ylabel("Y [m]")
            ax_xy.axis("equal")
            ax_xy.grid(True)
            ax_xy.legend(loc="best")

            # YZ plane
            ax_yz = fig.add_subplot(2, 2, 3)
            ax_yz.plot(gt_xyz[:, 1], gt_xyz[:, 2], linewidth=2.0, label="Ground Truth")
            ax_yz.plot(pred_xyz[:, 1], pred_xyz[:, 2], linewidth=2.0, label="Prediction")
            ax_yz.set_title("YZ Plane")
            ax_yz.set_xlabel("Y [m]")
            ax_yz.set_ylabel("Z [m]")
            ax_yz.axis("equal")
            ax_yz.grid(True)
            ax_yz.legend(loc="best")

            # XZ plane
            ax_xz = fig.add_subplot(2, 2, 4)
            ax_xz.plot(gt_xyz[:, 0], gt_xyz[:, 2], linewidth=2.0, label="Ground Truth")
            ax_xz.plot(pred_xyz[:, 0], pred_xyz[:, 2], linewidth=2.0, label="Prediction")
            ax_xz.set_title("XZ Plane")
            ax_xz.set_xlabel("X [m]")
            ax_xz.set_ylabel("Z [m]")
            ax_xz.axis("equal")
            ax_xz.grid(True)
            ax_xz.legend(loc="best")

            fig.suptitle(
                f"Val Trajectory Comparison - Epoch {epoch} - {seq_id} - Val Loss: {val_loss:.6f}",
                fontsize=14,
            )
            fig.tight_layout()

            # Log to TensorBoard
            writer.add_figure(f"Validation_Trajectory/{seq_id}", fig, epoch)
            plt.close(fig)
            break


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train(
    train_data_dir,
    val_data_dir,
    num_epochs,
    batch_size,
    lr,
    log_dir=Path(output_dir) / "logs",
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    num_workers=2,
    checkpoint_dir=Path(output_dir) / "checkpoints",
    model=ModelTypes.VISUAL,
    sequence_length=10,
    lstm_hidden=1000,
    image_height=360,
    image_width=480,
    plot_every=5,
):
    # specify log and checkpoint directories by model type to avoid conflicts
    log_dir = log_dir / model.name
    checkpoint_dir = checkpoint_dir / model.name

    # clear contents of current model type's log_dir
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    # make a folder for checkpoints in case it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    pipeline = Pipeline(
        model,
        train_data_dir,
        val_data_dir,
        sequence_length=sequence_length,
        lstm_hidden=lstm_hidden,
        image_height=image_height,
        image_width=image_width,
    )
    train_dataset = pipeline.train_dataset
    val_dataset = pipeline.val_dataset

    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = torch.compile(pipeline.model.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # mixed precision scaler
    scaler = GradScaler(enabled=(torch.cuda.is_available() and "cuda" in str(device)))

    start_epoch = 0
    best_val_loss = float("inf")
    global_step = start_epoch * max(1, len(train_loader))
    for epoch in range(start_epoch, num_epochs):
        #### train ####
        model.train()

        train_loss = 0.0
        num_train_samples = len(train_dataset)
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            batch = _move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=scaler.is_enabled()):
                loss = model.compute_loss(batch)

            # Back propagation and optimizer step with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_batch_size = batch["target_rel_poses"].shape[0]
            train_loss += loss.item() * current_batch_size

            # Logging
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1

        train_loss /= max(1, num_train_samples)

        #### validation ####
        model.eval()
        val_loss = 0.0
        num_val_samples = len(val_dataset)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}"):
                batch = _move_batch_to_device(batch, device)

                loss = model.compute_loss(batch)
                current_batch_size = batch["target_rel_poses"].shape[0]
                val_loss += loss.item() * current_batch_size

        val_loss /= max(1, num_val_samples)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        if plot_every > 0 and (epoch % plot_every == 0 or epoch == num_epochs - 1):
            _save_val_trajectory_plot(
                model, val_loader, device, epoch, val_loss, writer
            )

        # logging
        writer.add_scalar("Loss/Val", val_loss, epoch)
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

    writer.close()


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=360,
        help="Image height for training (original: 480)",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=480,
        help="Image width for training (original: 640)",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=5,
        help="Save val GT-vs-predicted 3D trajectory plot every N epochs (<=0 disables)",
    )

    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument("-v", action="store_true", help="Use Visual Model")
    type_group.add_argument("-i", action="store_true", help="Use Inertial Model")
    type_group.add_argument(
        "-vi", action="store_true", help="Use Visual-Inertial Model"
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    model_type = (
        ModelTypes.VISUAL
        if args.v
        else (ModelTypes.INERTIAL if args.i else ModelTypes.VISUAL_INERTIAL)
    )

    train(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model=model_type,
        image_height=args.image_height,
        image_width=args.image_width,
        plot_every=args.plot_every,
    )


if __name__ == "__main__":
    main()
