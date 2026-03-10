#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from Dataset import NeRFDataset
from NeRFModel import NeRFmodel


def train(
    train_data_dir,
    val_data_dir,
    num_epochs=50,
    batch_size=4096,
    lr=5e-4,
    log_dir="logs",
    device="cuda",
    checkpoint_dir="checkpoints",
):

    # make a folder for checkpoints in case it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    # clear contents of log_dir to begin with a clean slate
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    # datasets
    train_dataset = NeRFDataset(train_data_dir, batch_size, device=device)
    val_dataset = NeRFDataset(val_data_dir, batch_size, device=device)

    # model
    model = NeRFmodel(embed_pos_L=10, embed_direction_L=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        #### train ####
        model.train()

        # Tally of total loss zeroed
        # Reporting information
        train_loss = 0.0
        num_train_samples = len(train_dataset)
        num_train_iters = max(1, math.ceil(num_train_samples / batch_size))

        # Progress Bar per epoch
        for _ in tqdm(range(num_train_iters), desc=f"Train {epoch}"):
            # sample a batch of rays and corresponding RGB values from the training dataset
            ray_batch, rgb_batch = train_dataset.get_batch()

            # forward pass through the model to get predicted RGB values
            pred_rgb_coarse, pred_rgb_fine = model(ray_batch)

            # loss calculation
            loss = model.compute_loss(pred_rgb_coarse, pred_rgb_fine, rgb_batch)

            # Back propagation and gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ray_batch.shape[0]
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            global_step += ray_batch.shape[0]

        train_loss /= num_train_iters

        #### validation ####
        model.eval()
        val_loss = 0.0

        # do not compute the gradients for validation, this is only to see how the model is doing at this point
        with torch.no_grad():
            num_val_samples = len(val_dataset)
            num_val_iters = max(1, math.ceil(num_val_samples / batch_size))

            for _ in tqdm(range(num_val_iters), desc=f"Val {epoch}"):
                ray_batch, rgb_batch = val_dataset.get_batch()
                pred_rgb_coarse, pred_rgb_fine = model(ray_batch)
                loss = model.compute_loss(pred_rgb_coarse, pred_rgb_fine, rgb_batch)
                val_loss += loss.item() * ray_batch.shape[0]

        val_loss /= num_val_iters
        best_val_loss = min(best_val_loss, val_loss)

        #### logging ####
        writer.add_scalars(
            "loss/epoch",
            {"train": train_loss, "val": val_loss},
            epoch,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train: {train_loss:.4f} | "
            f"val: {val_loss:.4f} | "
        )

        # TODO: save model checkpoint

    writer.close()


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="lego",
        choices=["lego", "ship"],
        help="dataset to train on: lego or ship",
    )
    args = parser.parse_args()

    top_data_dir = Path(__file__).parent.parent / "Data" / "nerf_synthetic"
    dataset_dir = top_data_dir / args.dataset

    train(
        train_data_dir=dataset_dir / "train",
        val_data_dir=dataset_dir / "val",
        device=device,
    )


if __name__ == "__main__":
    main()
