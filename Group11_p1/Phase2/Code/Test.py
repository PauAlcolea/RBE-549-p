#!/usr/bin/env python

import sys

sys.dont_write_bytecode = True

import os
import time
import torch
import numpy as np
from argparse import ArgumentParser

from Dataset import NORMALIZING_FACTOR, HomographyDataset
from Network.Network import SupervisedHomographyModel, UnsupervisedHomographyModel


def calculate_epe(predictions, ground_truth):
    """
    Calculate Endpoint Error (L2 distance) between predictions and ground truth

    Args:
        predictions: predicted corner shifts (N, 8) or (N, 4, 2)
        ground_truth: ground truth corner shifts (N, 8) or (N, 4, 2)

    Returns:
        epe: average L2 distance per corner point
    """
    # Reshape to (N, 4, 2) if needed
    if predictions.ndim == 2 and predictions.shape[1] == 8:
        predictions = predictions.reshape(-1, 4, 2)
    if ground_truth.ndim == 2 and ground_truth.shape[1] == 8:
        ground_truth = ground_truth.reshape(-1, 4, 2)

    # Calculate L2 distance for each corner
    diff = predictions - ground_truth
    distances = np.linalg.norm(diff, axis=2)  # (N, 4)

    # Average over all corners and all samples
    epe = np.mean(distances)
    return epe


def evaluate_model(model, dataset, device, batch_size=64):
    """
    Evaluate model on entire dataset and calculate average EPE

    Args:
        model: trained model
        dataset: HomographyDataset instance
        device: device to run inference on
        batch_size: batch size for inference

    Returns:
        avg_epe: average EPE over the dataset
        avg_inference_time: average forward pass time per sample (in ms)
    """
    model.eval()

    all_predictions = []
    all_ground_truth = []

    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Running inference on {num_samples} samples...")

    # Warm-up run to initialize CUDA graphs
    if device == "cuda":
        print("Warming up CUDA...")
        with torch.no_grad():
            dummy_a = torch.randn(1, 1, 128, 128).to(device)
            dummy_b = torch.randn(1, 1, 128, 128).to(device)
            _ = model(dummy_a, dummy_b)
            torch.cuda.synchronize()

    total_inference_time = 0.0
    total_samples_timed = 0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # Load batch
            batch_patches_a = []
            batch_patches_b = []
            batch_shifts = []

            for idx in range(start_idx, end_idx):
                patch_a, patch_b, shifts = dataset.get_sample(idx)
                batch_patches_a.append(patch_a)
                batch_patches_b.append(patch_b)
                batch_shifts.append(shifts)

            # Stack into tensors
            patch_a = torch.stack(batch_patches_a).to(device)
            patch_b = torch.stack(batch_patches_b).to(device)
            gt_shifts = torch.stack(batch_shifts).numpy()

            # Time the forward pass
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            predictions = model(patch_a, patch_b)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Accumulate timing
            total_inference_time += end_time - start_time
            total_samples_timed += current_batch_size

            # Convert predictions to numpy
            predictions_np = predictions.cpu().numpy()

            # Denormalize (predictions and ground truth are in normalized space)
            predictions_denorm = predictions_np * NORMALIZING_FACTOR
            gt_shifts_denorm = gt_shifts * NORMALIZING_FACTOR

            all_predictions.append(predictions_denorm)
            all_ground_truth.append(gt_shifts_denorm)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{num_samples} samples...")

    # Concatenate all predictions and ground truth
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Calculate EPE
    avg_epe = calculate_epe(all_predictions, all_ground_truth)

    # Calculate average inference time per sample in milliseconds
    avg_inference_time = (total_inference_time / total_samples_timed) * 1000

    return avg_epe, avg_inference_time


def main():
    parser = ArgumentParser(description="Evaluate homography model and compute EPE")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="Val",
        choices=["Train", "Val", "Test"],
        help="Dataset to evaluate on (Train, Val, or Test)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "-t",
        "--ModelType",
        type=str,
        default="supervised",
        choices=["supervised", "unsupervised"],
        help="Type of model architecture",
    )

    args = parser.parse_args()

    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Setup paths
    gen_data_dir = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Data/Generated"
    )
    data_dir = os.path.join(gen_data_dir, args.dataset)
    if args.dataset == "Test":
        data_dir = os.path.join(gen_data_dir, "Test/Phase2")
    label_file = os.path.join(data_dir, "labels.txt")

    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        return

    # Load dataset
    print(f"Loading {args.dataset} dataset from {data_dir}...")
    dataset = HomographyDataset(data_dir, label_file)
    print(f"Dataset size: {len(dataset)} samples")

    # Load model
    if args.ModelType == "supervised":
        model = SupervisedHomographyModel()
        path = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/supervised.pt"
    else:
        model = UnsupervisedHomographyModel()
        path = (
            os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/unsupervised.pt"
        )
    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

    # Evaluate
    avg_epe, avg_time = evaluate_model(
        model, dataset, device, batch_size=args.batch_size
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS - {args.dataset} Dataset")
    print("=" * 60)
    print(f"Average EPE (Endpoint Error): {avg_epe:.4f} pixels")
    print(f"Average inference time per sample: {avg_time:.4f} ms")
    print(f"Throughput: {1000.0/avg_time:.2f} samples/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
