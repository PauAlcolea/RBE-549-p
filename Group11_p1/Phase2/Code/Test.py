#!/usr/bin/env python

import sys

sys.dont_write_bytecode = True

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Phase1.Code.Wrapper import (
    locate_corners,
    ANMS,
    encode_feature_points,
    match_features,
    _normalize_points,
    _form_A_matrix,
    form_panorama,
)
from Network.Network import SupervisedHomographyModel
from Dataset import NORMALIZING_FACTOR


def load_images(dir):
    images = []
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir, filename))
            if img is not None:
                images.append(img)
    return images


def get_matching_features(images):
    raw_corners = [locate_corners(image) for image in images]
    anms_corners = [ANMS(corners) for corners in raw_corners]
    fds, kps = zip(
        *[
            encode_feature_points(image, corners)
            for image, corners in zip(images, anms_corners)
        ]
    )
    pairwise_matches = []
    for i in range(len(images) - 1):
        # match_features returns index pairs (idx_in_img_i, idx_in_img_i+1)
        idx_matches = match_features(fds[i], fds[i + 1])

        # convert index pairs to coordinate pairs ((x1, y1), (x2, y2))
        kp1 = kps[i]
        kp2 = kps[i + 1]
        coord_matches = [(kp1[i1], kp2[i2]) for i1, i2 in idx_matches]

        pairwise_matches.append(coord_matches)
    return pairwise_matches


def extract_patches(image1, image2, matches, patch_size=128, max_patches=12, min_dist=None):
    """
    select spatially spread patches around matched features
    """

    if len(matches) == 0:
        return [], [], []

    half_size = patch_size // 2
    if min_dist is None:
        min_dist = half_size

    # list candidate centers where a full patch fits in both images
    candidates = []
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    for (x1, y1), (x2, y2) in matches:
        x1_i, y1_i = int(x1), int(y1)
        x2_i, y2_i = int(x2), int(y2)

        if (
            x1_i - half_size >= 0
            and y1_i - half_size >= 0
            and x1_i + half_size <= w1
            and y1_i + half_size <= h1
            and x2_i - half_size >= 0
            and y2_i - half_size >= 0
            and x2_i + half_size <= w2
            and y2_i + half_size <= h2
        ):
            candidates.append((x1_i, y1_i, x2_i, y2_i))

    if not candidates:
        return [], [], []

    # select spatially spread candidates
    selected = []
    for cand in candidates:
        x1_i, y1_i, _, _ = cand
        if not selected:
            selected.append(cand)
        else:
            # compute distance to previously selected centers; keep if far enough away
            dists = [np.hypot(x1_i - sx1, y1_i - sy1) for (sx1, sy1, _, _) in selected]
            if min(dists) >= min_dist:
                selected.append(cand)
        if len(selected) >= max_patches:
            break

    # extract patches around selected centers
    patches1 = []
    patches2 = []
    patches1_coords = []
    for x1_i, y1_i, x2_i, y2_i in selected:
        p1_x1, p1_y1 = x1_i - half_size, y1_i - half_size
        p1_x2, p1_y2 = x1_i + half_size, y1_i + half_size
        patch1 = image1[p1_y1:p1_y2, p1_x1:p1_x2]
        patch2 = image2[
            y2_i - half_size : y2_i + half_size,
            x2_i - half_size : x2_i + half_size,
        ]
        patches1.append(patch1)
        patches2.append(patch2)
        patches1_coords.append(
            ((p1_x1, p1_y1), (p1_x2, p1_y1), (p1_x2, p1_y2), (p1_x1, p1_y2))
        )

    return patches1, patches2, patches1_coords


def patches_to_tensor(patches):
    """
    patches: list of (H, W, 3) numpy arrays
    returns: torch tensor (B, 1, H, W)
    """
    tensors = []
    for p in patches:
        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        p = torch.from_numpy(p).unsqueeze(0).float() / 255.0
        tensors.append(p)
    return torch.stack(tensors)


def _compute_homography(pairs):
    pairs = np.asarray(pairs, dtype=np.float32)

    if pairs.ndim == 4 and pairs.shape[2:] == (2, 2):
        pairs = pairs.reshape(-1, 2, 2)

    # normalize points
    points1_norm, T1 = _normalize_points(pairs[:, 0])
    points2_norm, T2 = _normalize_points(pairs[:, 1])
    norm_pairs = np.stack([points1_norm, points2_norm], axis=1)

    # form A matrix and solve A*h=0 using SVD
    A = _form_A_matrix(norm_pairs)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # last row of Vt is null space of A, i.e. solution h
    H = h.reshape((3, 3)) / np.max(np.abs(h))
    # denormalize H
    H = np.linalg.inv(T2) @ H @ T1
    return H


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    data_top_dir = (
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        + "/Phase1/Data/"
    )
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="Train/CustomSet1",
        help="directory containing test images; relative to Phase1/Data, i.e. 'Train/Set1'",
    )
    args = parser.parse_args()
    test_data_dir = os.path.join(data_top_dir, args.dir)

    # initialize model
    model_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/best_model.pt"
    )
    model = SupervisedHomographyModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # extract patches from images via feature matching
    images = load_images(test_data_dir)
    pairwise_matches = get_matching_features(images)
    pairwise_H = []
    for i in range(len(images) - 1):
        # extract patches around matched features
        patches_a, patches_b, patch_a_coords = extract_patches(
            images[i], images[i + 1], pairwise_matches[i]
        )

        # predict deltas with trained model
        with torch.no_grad():
            delta_preds = (
                model(
                    patches_to_tensor(patches_a).to(device),
                    patches_to_tensor(patches_b).to(device),
                )
                .cpu()
                .numpy()
            )

        delta_preds = delta_preds * NORMALIZING_FACTOR

        patch_a_coords_np = np.array(patch_a_coords, dtype=np.float32)
        deltas = delta_preds.reshape(-1, 4, 2)
        patch_b_coords_est = patch_a_coords_np + deltas

        # build all corner correspondence pairs across all patches
        src_pts = patch_a_coords_np.reshape(-1, 2)
        dst_pts = patch_b_coords_est.reshape(-1, 2)
        corner_pairs = np.stack([src_pts, dst_pts], axis=1)

        H_i = _compute_homography(corner_pairs)
        pairwise_H.append(H_i)
    panorama = form_panorama(images, pairwise_H=pairwise_H, graph_mode=False)
    cv2.imwrite("mypano.png", panorama)


if __name__ == "__main__":
    main()
