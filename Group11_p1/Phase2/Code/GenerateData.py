import numpy as np
import cv2
import os
import random
from argparse import ArgumentParser
import sys


def read_images(directory, num_images):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
                if len(images) >= num_images:
                    break
    return images


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "-n", "--num_images", type=int, default=100, help="number of images to generate"
    )
    ap.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["Train", "Val"],
        default="Train",
        help="type of dataset to generate",
    )
    args = ap.parse_args()
    num_images = args.num_images
    dataset_type = args.type

    # set up directories
    data_top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Data"
    input_dir = os.path.join(data_top_dir, dataset_type)
    output_dir = os.path.join(data_top_dir, "Generated", dataset_type)

    # read images
    images = read_images(input_dir, num_images)
    print(len(images), "images read from", input_dir)


if __name__ == "__main__":
    main()
