#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
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
    def __init__(self, model_type, train_dir, val_dir):
        self.model_type = model_type

        if model_type == ModelTypes.VISUAL:
            self.train_dataset = VisualDataset(train_dir)
            self.val_dataset = VisualDataset(val_dir)
            self.model = VisualModel()
        elif model_type == ModelTypes.INERTIAL:
            self.train_dataset = InertialDataset(train_dir)
            self.val_dataset = InertialDataset(val_dir)
            self.model = InertialModel()
        elif model_type == ModelTypes.VISUAL_INERTIAL:
            self.train_dataset = VisualInertialDataset(train_dir)
            self.val_dataset = VisualInertialDataset(val_dir)
            self.model = VisualInertialModel()


def train(
    train_data_dir,
    val_data_dir,
    num_epochs,
    batch_size,
    lr,
    log_dir=Path(output_dir) / "logs",
    device="cuda",
    checkpoint_dir=Path(output_dir) / "checkpoints",
    model=ModelTypes.VISUAL,
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

    pipeline = Pipeline(model, train_data_dir, val_data_dir)
    train_dataset = pipeline.train_dataset
    val_dataset = pipeline.val_dataset
    model = pipeline.model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")

    global_step = start_epoch * max(1, math.ceil(len(train_dataset) / batch_size))

    for epoch in range(start_epoch, num_epochs):
        #### train ####
        model.train()

        train_loss = 0.0
        num_train_samples = len(train_dataset)
        num_train_iters = max(1, math.ceil(num_train_samples / batch_size))

        for _ in tqdm(range(num_train_iters), desc=f"Train {epoch}"):
            batch = train_dataset.get_batch(batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Logging
            train_loss += loss.item()
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1

        train_loss /= num_train_samples

        #### validation ####
        model.eval()
        val_loss = 0.0
        num_val_samples = len(val_dataset)
        num_val_iters = max(1, math.ceil(num_val_samples / batch_size))

        with torch.no_grad():
            for _ in tqdm(range(num_val_iters), desc=f"Val {epoch}"):
                batch = val_dataset.get_batch(batch_size)
                batch = {k: v.to(device) for k, v in batch.items()}

                loss = model.compute_loss(batch)
                val_loss += loss.item() * batch_size

        val_loss /= num_val_samples
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
    )


if __name__ == "__main__":
    main()
