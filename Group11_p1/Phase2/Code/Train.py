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
import kornia


def train_supervised(
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
                corner_err = (1 / 4) * torch.norm(
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
            f"Epoch {epoch:03d} | "
            f"train: {train_loss:.4f} | "
            f"val: {val_loss:.4f} | "
            f"corner_err: {val_corner_err:.4f}"
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

""" 
Tensor Direct Linear Transform fully differentiable
@param batch_shifts is a tensor of shape torch.Size([batch_size, 8]), 
@param batch_patch_shape is (B, C, H, W)
"""
def dlt(batch_shifts: torch.Tensor, batch_patch_shape: torch.Tensor):
    #patch shape from patch a
    # print("batch patch shape:", batch_patch_shape)
    B = batch_shifts.shape[0]
    h = batch_patch_shape[2]
    w = batch_patch_shape[3]
    
    #make sure that the source points are in the same format as the batch shifts (B, 4, 2)
    src_points = torch.tensor([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], device=batch_shifts.device, dtype=batch_shifts.dtype)
    src_points = src_points.unsqueeze(0).repeat(B, 1, 1) 
    
    #separate batch into different points [[x1',y1'],[x2',y2'],[x3',y3'],[x4',y4']]
    # then add them to the source points to get the destination points
    dst_points = src_points + batch_shifts.view(B, 4, 2)

    # separate into each component, which will be tensors of shape (B, 4)
    x = src_points[:,:,0]
    y = src_points[:,:,1]
    xp = dst_points[:,:,0]
    yp = dst_points[:,:,1]
    
    #zeros and ones for the A matrix
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # make the two rows of A and then join them
    # each batch B will have 4 rows (4 because x, y, xp, yp, ... each have shape (B, 4), and each row will have the 8 elements)
    # A is of shape (B, 8, 8)
    # https://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf
    ###################### UNSURE ABOUT THIS ^^^, SOMEWHAT CONFLICTING WITH THE SLIDES FROM LECTURE ######################
    A1 = torch.stack([x, y, ones, zeros, zeros, zeros, -x*xp, -y*x], dim=2)
    A2 = torch.stack([zeros, zeros, zeros, x, y, ones, -x*y, -y*yp], dim=2)
    A = torch.cat([A1, A2], dim=1)
    b = torch.cat([xp, yp], dim=1)

    # Solve for h
    # h will be of shape (B, 8), it's missing the one at the end
    # h = torch.inverse(torch.transpose(A) @ A) @ torch.transpose(A) @ b
    h = torch.linalg.solve(A, b)

    # Add the last value
    h = torch.cat([h, torch.ones(B, 1, device=batch_shifts.device)], dim=1)
    H = h.view(B, 3, 3)
    return H


def train_unsupervised(train_data_dir, train_label_file, val_data_dir, val_label_file, num_epochs=300, batch_size=128, 
                       lr=0.003,log_dir="logs", device="cuda", patience=20, checkpoint_dir="checkpoints", supervised=True):
    
    # make a folder for checkpoints in case it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # datasets
    train_dataset = HomographyDataset(train_data_dir, train_label_file)
    val_dataset = HomographyDataset(val_data_dir, val_label_file)

    # Specify the Model, the optimizer, the scheduler and the loss function
    # the scheduler decreases the learning rate if the model is not improving
    model = SupervisedHomographyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = torch.nn.SmoothL1Loss()
    corner_loss_weight = 0.1

    # clear contents of log_dir to begin with a clean slate
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    global_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # go epoch by epoch, this is each "cycle" of the training
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
            # Generate a batch. Three tensors of dimensions **assumed**  patch_a and patch_b = (B, h, w c), gt_delta = (B, 8)
            patch_a, patch_b, gt_delta = GenerateBatch(train_dataset, batch_size)

            patch_a = patch_a.to(device)
            patch_b = patch_b.to(device)
            gt_delta = gt_delta.to(device)

            # Forward pass
            pred_delta = model(patch_a, patch_b)
            
            ## this is where the unsupervised changes will happen
            
            # loss calculation
            if supervised:
                loss = loss_fn(pred_delta, gt_delta)
                # incorporate corner error into loss
                pred_delta = pred_delta * NORMALIZING_FACTOR
                gt_delta = gt_delta * NORMALIZING_FACTOR
                loss_corner = (pred_delta - gt_delta).view(-1, 4, 2).norm(dim=2).mean()
                loss += corner_loss_weight * loss_corner
            else:                 
                #take the 4 Preddicted Shifts and throw them into the Direct Linear Transform (DLT)
                # (make sure that it remains differentiable)
                # the output of that is a predicted homography
                # H_batch is of shape (batch_size, 3, 3)
                H_batch = dlt(pred_delta, batch_patch_shape=patch_a.shape)

                # take the homography and throw it into the Spatial Transform Network
                # This should output a warped image of B based on the 
                # Use inverse mapping
                height = patch_a.shape[2]
                width = patch_a.shape[3]

                # spatial transform network makes a grid of the same size as patch b
                # for every pixel in that tensor, it uses the Homography from dlt to look back into what point in patch a it would have to pull from
                # this is most likely a float, so interpolation is necessary
                # this is all taken care of my kornia
                warped_a = kornia.geometry.transform.warp_perspective(patch_a, 
                                                                      torch.inverse(H_batch), 
                                                                      dsize=(height, width),
                                                                      mode="bilinear",
                                                                      padding_mode="zeros",
                                                                      align_corners=True,)
                
                # Photometric loss
                loss = torch.mean(torch.abs(warped_a - patch_b))

            
            # Back propagation and gradient
            optimizer.zero_grad()
            loss.backward()
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
                    loss = loss_fn(pred_delta, gt_delta)
                    pred_delta = pred_delta * NORMALIZING_FACTOR
                    gt_delta = gt_delta * NORMALIZING_FACTOR

                    # EPE is L2 err between predicted and ground truth corners
                    # this is the metric, what the scheduler watches
                    corner_err = (1 / 4) * torch.norm(
                        (pred_delta - gt_delta).view(-1, 4, 2), dim=2
                    ).sum(dim=1).mean()
                    val_corner_err += corner_err.item()

                    # incorporate corner error into loss
                    loss_corner = (pred_delta - gt_delta).view(-1, 4, 2).norm(dim=2).mean()
                    loss += corner_loss_weight * loss_corner
                    val_loss += loss.item()
                else:
                    #dlt
                    H_batch = dlt(pred_delta, batch_patch_shape=patch_a.shape)

                    # take the homography and throw it into the Spatial Transform Network
                    # This should output a warped image of B based on the 
                    # Use inverse mapping
                    height_nograd = patch_a.shape[2]
                    width_nograd = patch_a.shape[3]

                    # stn
                    warped_a = kornia.geometry.transform.warp_perspective(patch_a, 
                                                                        torch.inverse(H_batch), 
                                                                        dsize=(height_nograd, width_nograd),
                                                                        mode="bilinear",
                                                                        padding_mode="zeros",
                                                                        align_corners=True,)
                    
                    # Photometric loss
                    loss = torch.mean(torch.abs(warped_a - patch_b))
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
        "-t", "--type", type=str, default="s", choices=["s", "u"], help="Type of training: s for supervised or u for unsupervised"
    )
    args = parser.parse_args()
    if args.type == "s":
        train_supervised(
            train_data_dir=train_data_dir,
            train_label_file=f"{train_data_dir}/labels.txt",
            val_data_dir=val_data_dir,
            val_label_file=f"{val_data_dir}/labels.txt",
            device=device,
        )
    else:
        train_unsupervised(
            train_data_dir=train_data_dir,
            train_label_file=f"{train_data_dir}/labels.txt",
            val_data_dir=val_data_dir,
            val_label_file=f"{val_data_dir}/labels.txt",
            device=device,
            supervised=False
        )


if __name__ == "__main__":
    main()
