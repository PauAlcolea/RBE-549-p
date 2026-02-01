#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from argparse import ArgumentParser

from Network.Network import SupervisedHomographyModel


def load_images(dir):
    images = []
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((filename, img))
    return images


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    data_top_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/Phase1/Data/"
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="Train/CustomSet1",
        help="directory containing test images; relative to Phase1/Data, i.e. 'Train/Set1'")
    args = parser.parse_args()
    test_data_dir = os.path.join(data_top_dir, args.dir)

    images = load_images(test_data_dir)

    # use a few patches from each pair of images with trained model to estimate pairwise homography
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/best_model.pt"
    model = SupervisedHomographyModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    


if __name__ == "__main__":
    main()
