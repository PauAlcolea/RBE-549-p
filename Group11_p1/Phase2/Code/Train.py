#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Dataset import NORMALIZING_FACTOR, GenerateBatch, HomographyDataset
from Network.Network import SupervisedHomographyModel


def train(
    train_data_dir,
    train_label_file,
    val_data_dir,
    val_label_file,
    num_epochs=100,
    batch_size=128,
    lr=0.003,
    log_dir="logs",
    device="cuda",
    patience=10,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # dataset
    train_dataset = HomographyDataset(train_data_dir, train_label_file)
    val_dataset = HomographyDataset(val_data_dir, val_label_file)

    # model
    model = SupervisedHomographyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = torch.nn.SmoothL1Loss()
    corner_loss_weight = 0.1

    # clear contents of log_dir
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    writer = SummaryWriter(log_dir)

    global_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        #### train ####
        model.train()
        train_loss = 0.0

        num_train_samples = len(train_dataset)
        num_train_iters = max(1, math.ceil(num_train_samples / batch_size))

        for _ in tqdm(range(num_train_iters), desc=f"Train {epoch}"):
            patch_a, patch_b, gt_delta = GenerateBatch(train_dataset, batch_size)

            patch_a = patch_a.to(device)
            patch_b = patch_b.to(device)
            gt_delta = gt_delta.to(device)

            pred_delta = model(patch_a, patch_b)
            loss = loss_fn(pred_delta, gt_delta)

            # incorporate corner error into loss
            pred_delta = pred_delta * NORMALIZING_FACTOR
            gt_delta = gt_delta * NORMALIZING_FACTOR
            loss_corner = (pred_delta - gt_delta).view(-1, 4, 2).norm(dim=2).mean()
            loss += corner_loss_weight * loss_corner

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            global_step += 1

        train_loss /= num_train_iters

        #### validation ####
        model.eval()
        val_loss = 0.0
        val_corner_err = 0.0

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

                pred_delta = model(patch_a, patch_b)
                loss = loss_fn(pred_delta, gt_delta)

                pred_delta = pred_delta * NORMALIZING_FACTOR
                gt_delta = gt_delta * NORMALIZING_FACTOR

                # EPE is L2 err between predicted and ground truth corners
                corner_err = (1/4) * torch.norm(
                    (pred_delta - gt_delta).view(-1, 4, 2), dim=2
                ).sum(dim=1).mean()
                val_corner_err += corner_err.item()

                # incorporate corner error into loss
                loss_corner = (pred_delta - gt_delta).view(-1, 4, 2).norm(dim=2).mean()
                loss += corner_loss_weight * loss_corner
                val_loss += loss.item()

        val_loss /= num_val_iters
        val_corner_err /= num_val_iters
        lr_scheduler.step(val_corner_err)

        #### logging ####
        writer.add_scalars(
            "loss/epoch",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        writer.add_scalar("val/corner_err", val_corner_err, epoch)

        print(
            f"Epoch {epoch:03d} | " f"train: {train_loss:.4f} | " f"val: {val_loss:.4f} | " f"corner_err: {val_corner_err:.4f}"
        )

        if epoch % 5 == 0:
            writer.add_images("val/patch_a", patch_a, epoch)
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
            print(f"Early stopping triggered after {epochs_no_improve} epochs.")
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

    train(
        train_data_dir=train_data_dir,
        train_label_file=f"{train_data_dir}/labels.txt",
        val_data_dir=val_data_dir,
        val_label_file=f"{val_data_dir}/labels.txt",
        device=device,
    )


if __name__ == "__main__":
    main()
