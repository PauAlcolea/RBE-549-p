import argparse
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

from Train import train


def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    pass


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="lego",
        choices=["lego", "ship"],
        help="dataset to train on: lego or ship",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="whether to run test (default is Train)",
    )
    parser.add_argument(
        "--down",
        type=int,
        default=1,
        help="how much you want to downscale the images so training takes less time",
    )

    args = parser.parse_args()
    return args


def main(args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    top_data_dir = Path(__file__).parent.parent / "Data" / "nerf_synthetic"
    dataset_dir = top_data_dir / args.dataset

    if args.test:
        raise NotImplementedError("TODO: implement test mode")
    else:
        train(
            train_data_dir=dataset_dir / "train",
            val_data_dir=dataset_dir / "val",
            device=device,
            downscale=args.down
        )


if __name__ == "__main__":
    main(parseArgs())
