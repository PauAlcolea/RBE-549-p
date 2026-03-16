#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from Dataset import NeRFDataset
from NeRFModel import NeRFmodel


def _render_view(model, dataset, img_idx=0, device="cuda", chunk_size=8192):
    """
    render a single view for validation
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # get all rays and ground truth RGB values for the given image
        ray_origins, ray_directions, rgb_gt, h, w = dataset.get_image_rays(img_idx)
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)

        num_rays = ray_origins.shape[0]
        # pass rays through the model in smaller chunks to avoid memory errors
        pred_chunks = []
        for start in range(0, num_rays, chunk_size):
            end = min(start + chunk_size, num_rays)
            ro_chunk = ray_origins[start:end]
            rd_chunk = ray_directions[start:end]

            # FIXME: using fine model for rendering???
            _, pred_rgb_fine = model(ro_chunk, rd_chunk)
            pred_chunks.append(pred_rgb_fine.detach().cpu())

        pred_rgb_flat = torch.cat(pred_chunks, dim=0)

        # reshape to image and clamp to [0, 1]
        pred_img = pred_rgb_flat.view(h, w, 3).permute(2, 0, 1)
        pred_img = torch.clamp(pred_img, 0.0, 1.0)

        gt_img = rgb_gt.view(h, w, 3).permute(2, 0, 1).cpu()
        gt_img = torch.clamp(gt_img, 0.0, 1.0)

    if was_training:
        model.train()

    return pred_img, gt_img


def train(
    train_data_dir,
    val_data_dir,
    num_iters=100000,
    batch_size=4096,
    lr=5e-4,
    log_dir="logs",
    device="cuda",
    checkpoint_dir="checkpoints",
    val_every=1000,
    render_every=5000,
    downscale=1,
    dataset_name=None,
    max_val_iters=50,
):

    # specify log and checkpoint directories by dataset name to avoid clashes
    if dataset_name is not None:
        log_dir = os.path.join(log_dir, str(dataset_name))
        checkpoint_dir = os.path.join(checkpoint_dir, str(dataset_name))

    # make a folder for checkpoints in case it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    # clear contents of current dataset's log_dir
    if dataset_name is not None and os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    # Prepare tensorboard logger
    writer = SummaryWriter(log_dir)

    # datasets
    train_dataset = NeRFDataset(
        train_data_dir, batch_size, device=device, downscale=downscale
    )
    val_dataset = NeRFDataset(
        val_data_dir, batch_size, device=device, downscale=downscale
    )

    # model
    model = NeRFmodel(embed_pos_L=10, embed_direction_L=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )

    # mixed precision scaler
    scaler = GradScaler(enabled=(torch.cuda.is_available() and "cuda" in str(device)))

    global_step = 0
    best_val_loss = float("inf")
    for iter in tqdm(range(num_iters), desc=f"Train"):
        #### train ####
        model.train()

        # sample a batch of rays and corresponding RGB values from the training dataset
        ray_origin_batch, ray_direction_batch, rgb_batch = train_dataset.get_batch()

        optimizer.zero_grad()
        # forward + loss under autocast for mixed precision
        with autocast('cuda', enabled=scaler.is_enabled()):
            pred_rgb_coarse, pred_rgb_fine = model(
                ray_origin_batch, ray_direction_batch
            )
            loss = model.compute_loss(pred_rgb_coarse, pred_rgb_fine, rgb_batch)

        # Back propagation and optimizer step with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update learning rate scheduler
        lr_scheduler.step()

        writer.add_scalar("train/loss_iter", loss.item(), global_step)
        global_step += ray_origin_batch.shape[0]

        #### validation ####
        if (iter + 1) % val_every == 0 or (iter + 1) == num_iters:
            model.eval()
            val_loss = 0.0

            # do not compute the gradients for validation, this is only to see how the model is doing at this point
            with torch.no_grad():
                num_val_samples = len(val_dataset)
                num_val_iters = min(max_val_iters, math.ceil(num_val_samples / batch_size))

                for _ in tqdm(range(num_val_iters), desc=f"Val {iter+1}"):
                    ray_origin_batch, ray_direction_batch, rgb_batch = (
                        val_dataset.get_batch()
                    )
                    pred_rgb_coarse, pred_rgb_fine = model(
                        ray_origin_batch, ray_direction_batch
                    )
                    loss = model.compute_loss(pred_rgb_coarse, pred_rgb_fine, rgb_batch)
                    val_loss += loss.item() * ray_origin_batch.shape[0]

            val_loss /= num_val_iters
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth")
                )

            # render sample view for tensorboard
            if (iter + 1) % render_every == 0 or (iter + 1) == num_iters:
                with torch.no_grad():
                    pred_img, gt_img = _render_view(
                        model,
                        val_dataset,
                        device=device,
                        chunk_size=batch_size,
                    )
                writer.add_image("val/render_pred", pred_img, global_step)
                writer.add_image("val/render_gt", gt_img, global_step)

            #### logging ####
            writer.add_scalars(
                "loss/iter",
                {"val": val_loss},
                iter + 1,
            )

            print(f"Iter {iter+1:07d} | " f"val: {val_loss:.4f} | ")

    writer.close()
