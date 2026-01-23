#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:
import numpy as np
import cv2
import argparse
import os
from matplotlib import pyplot as plt
from skimage.feature import corner_peaks


def locate_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    C_img = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    return C_img


def ANMS(C_img, N_best):
    # find local maxima
    coordinates = corner_peaks(C_img, min_distance=5, threshold_rel=0.01)
    # sort corners by corner strength (descending)
    values = C_img[coordinates[:, 0], coordinates[:, 1]]
    sorted_indices = np.argsort(-values)
    sorted_coords = coordinates[sorted_indices]
    # find minimum distance from each corner i to a stronger corner j
    N = len(sorted_coords)
    radii = np.full(N, np.inf)
    for i in range(1, N):
        yi, xi = sorted_coords[i]
        for j in range(i):  # iterate over corners stronger than i
            yj, xj = sorted_coords[j]
            dist = (xi - xj) ** 2 + (yi - yj) ** 2
            radii[i] = min(radii[i], dist)

    # select N_best corners with largest radii
    best_indices = np.argsort(-radii)[:N_best]
    return sorted_coords[best_indices]


def encode_feature_points(image, corners):
    descriptors = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    PATCH_RADIUS = 20
    for y, x in corners:
        # extract 40x40 patch around corner
        if (
            x - PATCH_RADIUS < 0
            or x + PATCH_RADIUS >= W
            or y - PATCH_RADIUS < 0
            or y + PATCH_RADIUS >= H
        ):
            continue
        patch = gray[
            y - PATCH_RADIUS : y + PATCH_RADIUS, x - PATCH_RADIUS : x + PATCH_RADIUS
        ]
        patch = cv2.GaussianBlur(patch, (5, 5), sigmaX=0)
        # downsample, normalize, and flatten
        small_patch = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
        std = max(np.std(small_patch), 1e-5)
        norm_patch = (small_patch - np.mean(small_patch)) / std
        descriptor = norm_patch.flatten()
        descriptors.append(descriptor)
    return np.array(descriptors)


def main():
    # get path to current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default=f"{parent_dir}/Data/Train/Set1",
        help="Directory of input images",
    )

    args = ap.parse_args()
    input_dir = args.dir

    """
    Read a set of images for Panorama stitching
    """
    images = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_dir, filename))
            if img is not None:
                images.append(img)
    print(f"Read {len(images)} images from {input_dir}")
    image1 = images[0]
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    raw_corners = locate_corners(image1)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    anms_corners = ANMS(raw_corners, 100)
    # for y, x in anms_corners:
    #     cv2.circle(image1, (x, y), 3, (0, 0, 255), -1)
    # cv2.imshow("ANMS Corners", image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    fd = encode_feature_points(image1, anms_corners)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """

    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """

    return


if __name__ == "__main__":
    main()
