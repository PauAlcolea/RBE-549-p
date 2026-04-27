#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        self, model_type, train_dir, val_dir, sequence_length, lstm_hidden
    ):
        self.model_type = model_type

        if model_type == ModelTypes.VISUAL:
            self.train_dataset = VisualDataset(
                train_dir,
                mode="sequences",
                sequence_length=sequence_length,
            )
            self.val_dataset = VisualDataset(
                val_dir,
                mode="sequences",
                sequence_length=sequence_length,
            )
            self.model = VisualModel(
                lstm_hidden_size=lstm_hidden,
            )
        elif model_type == ModelTypes.INERTIAL:
            self.train_dataset = InertialDataset(train_dir)
            self.val_dataset = InertialDataset(val_dir)
            self.model = InertialModel()
        elif model_type == ModelTypes.VISUAL_INERTIAL:
            self.train_dataset = VisualInertialDataset(train_dir)
            self.val_dataset = VisualInertialDataset(val_dir)
            self.model = VisualInertialModel()


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
    num_workers=4,
    checkpoint_dir=Path(output_dir) / "checkpoints",
    model=ModelTypes.VISUAL,
    sequence_length=10,
    lstm_hidden=1000,
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

    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    pipeline = Pipeline(
        model,
        train_data_dir,
        val_data_dir,
        sequence_length=sequence_length,
        lstm_hidden=lstm_hidden,
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = pipeline.model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

            current_batch_size = batch["target_rel_pose"].shape[0]
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
                current_batch_size = batch["target_rel_pose"].shape[0]
                val_loss += loss.item() * current_batch_size

        val_loss /= max(1, num_val_samples)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # TODO: something that plots the predicted vs ground truth trajectory for tensorboard

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
        num_workers=args.num_workers,
        model=model_type,
    )


if __name__ == "__main__":
    main()
