#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Dataset import HomographyDataset
from Network.Network import SupervisedHomographyModel


def train(
    train_data_dir,
    train_label_file,
    val_data_dir,
    val_label_file,
    num_epochs=50,
    batch_size=32,
    lr=1e-4,
    log_dir="logs",
    device="cuda",
    patience=10,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # dataset and dataloader
    train_dataset = HomographyDataset(train_data_dir, train_label_file)
    val_dataset = HomographyDataset(val_data_dir, val_label_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # model
    model = SupervisedHomographyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_fn = torch.nn.SmoothL1Loss()

    # clear contents of log_dir
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))

    writer = SummaryWriter(log_dir)

    global_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        #### train ####
        model.train()
        train_loss = 0.0

        for patch_a, patch_b, gt_delta in tqdm(train_loader, desc=f"Train {epoch}"):
            patch_a = patch_a.to(device)
            patch_b = patch_b.to(device)
            gt_delta = gt_delta.to(device)

            pred_delta = model(patch_a, patch_b)
            loss = loss_fn(pred_delta, gt_delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            global_step += 1

        train_loss /= len(train_loader)
        lr_scheduler.step()

        #### validation ####
        model.eval()
        val_loss = 0.0
        val_corner_err = 0.0

        with torch.no_grad():
            for patch_a, patch_b, gt_delta in tqdm(val_loader, desc=f"Val {epoch}"):
                patch_a = patch_a.to(device)
                patch_b = patch_b.to(device)
                gt_delta = gt_delta.to(device)

                pred_delta = model(patch_a, patch_b)
                loss = loss_fn(pred_delta, gt_delta)

                # EPE is L2 err between predicted and ground truth corners
                corner_err = torch.norm(pred_delta - gt_delta, p=2, dim=1).mean()
                val_corner_err += corner_err.item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_corner_err /= len(val_loader)

        #### logging ####
        writer.add_scalars(
            "loss/epoch",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        writer.add_scalar("val/corner_err", val_corner_err, epoch)

        print(
            f"Epoch {epoch:03d} | " f"train: {train_loss:.4f} | " f"val: {val_loss:.4f}"
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
