#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

from Dataset import NORMALIZING_FACTOR, GenerateBatch, HomographyDataset
from Network.Network import SupervisedHomographyModel, UnsupervisedHomographyModel


def train(
    train_data_dir,
    train_label_file,
    val_data_dir,
    val_label_file,
    num_epochs=300,
    batch_size=128,
    lr=0.0001,
    log_dir="logs",
    device="cuda",
    patience=25,
    checkpoint_dir="checkpoints",
    supervised=True,
    resume_checkpoint=None,
):

    # make a folder for checkpoints in case it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # datasets
    train_dataset = HomographyDataset(train_data_dir, train_label_file)
    val_dataset = HomographyDataset(val_data_dir, val_label_file)

    # Specify the Model, the optimizer, the scheduler and the loss function
    # the scheduler decreases the learning rate if the model is not improving
    model = (
        SupervisedHomographyModel().to(device)
        if supervised
        else UnsupervisedHomographyModel().to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # load from checkpoint if specified
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            print(f"Loading checkpoint from {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(
                f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}"
            )
        else:
            print(
                f"Warning: Checkpoint {resume_checkpoint} not found. Starting from scratch."
            )

    # clear contents of log_dir to begin with a clean slate
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    global_step = start_epoch * max(1, math.ceil(len(train_dataset) / batch_size))

    # go epoch by epoch, this is each "cycle" of the training
    for epoch in range(start_epoch, num_epochs):
        #### train ####
        model.train()

        # Tally of total loss zeroed
        # Reporting information
        train_loss = 0.0
        num_train_samples = len(train_dataset)
        num_train_iters = max(1, math.ceil(num_train_samples / batch_size))

        # Progress Bar per epoch
        for _ in tqdm(range(num_train_iters), desc=f"Train {epoch}"):
            # Generate a batch. Three tensors of dimensions **assumed**  patch_a and patch_b = (B, h, w c), gt_delta = (B, 8)
            patch_a, patch_b, gt_delta = GenerateBatch(train_dataset, batch_size)

            patch_a = patch_a.to(device)
            patch_b = patch_b.to(device)
            gt_delta = gt_delta.to(device)

            # Forward pass
            # this predicted delta is normalized
            pred_delta = model(patch_a, patch_b)

            # loss calculation
            if supervised:
                loss, corner_err = SupervisedHomographyModel.compute_loss(
                    pred_delta, gt_delta, NORMALIZING_FACTOR
                )
            else:
                # Unsupervised loss: photometric error after warping
                loss, warped_a = UnsupervisedHomographyModel.compute_loss(
                    pred_delta, patch_a, patch_b, NORMALIZING_FACTOR
                )

            # Back propagation and gradient
            optimizer.zero_grad()
            loss.backward()

            # clip gradient for stability
            if not supervised:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Logging
            train_loss += loss.item()
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            global_step += 1

        train_loss /= num_train_iters

        #### validation ####
        model.eval()
        val_loss = 0.0
        val_corner_err = 0.0

        # do not compute the gradients for validation, this is only to see how the model is doing at this point
        with torch.no_grad():
            num_val_samples = len(val_dataset)
            num_val_iters = max(1, math.ceil(num_val_samples / batch_size))

            for i in tqdm(range(num_val_iters), desc=f"Val {epoch}"):
                start_idx = i * batch_size
                patch_a, patch_b, gt_delta = val_dataset.get_batch_from_index(
                    start_idx, batch_size
                )

                patch_a = patch_a.to(device)
                patch_b = patch_b.to(device)
                gt_delta = gt_delta.to(device)

                # forward pass
                pred_delta = model(patch_a, patch_b)

                # loss calculation depending on if its supervised or unsupervised model
                if supervised:
                    loss, corner_err = SupervisedHomographyModel.compute_loss(
                        pred_delta, gt_delta, NORMALIZING_FACTOR
                    )
                    val_corner_err += corner_err.item()
                    val_loss += loss.item()
                else:
                    # Unsupervised loss: photometric error after warping
                    loss, warped_a = UnsupervisedHomographyModel.compute_loss(
                        pred_delta, patch_a, patch_b, NORMALIZING_FACTOR
                    )
                    val_loss += loss.item()

        # If corner didn't improve, scheduler decreases the learning rate
        val_loss /= num_val_iters
        val_corner_err /= num_val_iters

        # in unsupervised learning, you are not counting with the actual labels
        if supervised:
            lr_scheduler.step(val_corner_err)
        else:
            lr_scheduler.step(val_loss)

        #### logging ####
        writer.add_scalars(
            "loss/epoch",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        if supervised:
            writer.add_scalar("val/corner_err", val_corner_err, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train: {train_loss:.4f} | "
            f"val: {val_loss:.4f} | "
            f"corner_err: {val_corner_err:.4f}"
        )

        if epoch % 5 == 0:
            writer.add_images("val/patch_a", patch_a, epoch)
            if not supervised:
                writer.add_images("val/warped_a", warped_a, epoch)
            writer.add_images("val/patch_b", patch_b, epoch)

        #### checkpoint ####
        # save best model
        if epoch > 25:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    os.path.join(checkpoint_dir, "best_model.pt"),
                )

                print(f"✓ New best model saved (val_loss={val_loss:.4f})")

            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{patience} epochs")

        #### early stopping ####
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    writer.close()


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    gen_data_dir = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Data/Generated"
    )
    train_data_dir = f"{gen_data_dir}/Train"
    val_data_dir = f"{gen_data_dir}/Val"

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--ModelType",
        type=str,
        default="supervised",
        choices=["supervised", "unsupervised"],
        help="Type of training: supervised or unsupervised",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    train(
        train_data_dir=train_data_dir,
        train_label_file=f"{train_data_dir}/labels.txt",
        val_data_dir=val_data_dir,
        val_label_file=f"{val_data_dir}/labels.txt",
        device=device,
        supervised=True if args.ModelType == "supervised" else False,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
