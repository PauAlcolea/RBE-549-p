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


def extract_patches(image1, image2, matches, patch_size=128):
    patches1 = []
    patches2 = []
    patches1_coords = []
    half_size = patch_size // 2
    for (x1, y1), (x2, y2) in matches:
        if (
            x1 - half_size >= 0
            and y1 - half_size >= 0
            and x1 + half_size <= image1.shape[1]
            and y1 + half_size <= image1.shape[0]
            and x2 - half_size >= 0
            and y2 - half_size >= 0
            and x2 + half_size <= image2.shape[1]
            and y2 + half_size <= image2.shape[0]
        ):
            p1_x1, p1_y1 = x1 - half_size, y1 - half_size
            p1_x2, p1_y2 = x1 + half_size, y1 + half_size
            patch1 = image1[p1_y1:p1_y2, p1_x1:p1_x2]
            patch2 = image2[
                y2 - half_size : y2 + half_size, x2 - half_size : x2 + half_size
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
    # convert to NumPy array first
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


def RANSAC_homography(
    pairs,
    n_iterations=1000,
    inlier_thresh=10,
    stop_thresh=0.85,
):
    best_inlier_pairs = []

    for _ in range(n_iterations):
        pair_idx = np.random.choice(len(pairs), replace=False)
        random_pairs = pairs[pair_idx]
        # skip iteration if selected points are degenerate
        if np.linalg.matrix_rank(_form_A_matrix(random_pairs)) < 8:
            continue

        # compute homography from the four pairs; skip if SVD fails
        try:
            H = _compute_homography(random_pairs)
        except np.linalg.LinAlgError:
            continue

        # use H to map all points from image1 to image2, counting inliers
        inlier_pairs = []
        for p in pairs:
            dists = []
            for (x1, y1), (x2, y2) in p:
                p1 = np.array([x1, y1, 1]).reshape((3, 1))
                p2_est = H @ p1
                p2_est = (p2_est[:2] / max(p2_est[2], 1e-8)).flatten()
                dists.append(np.linalg.norm(p2_est - np.array([x2, y2]), ord=1))
            if np.mean(dists) < inlier_thresh:
                inlier_pairs.append(p)

        if len(inlier_pairs) > len(best_inlier_pairs):
            best_inlier_pairs = inlier_pairs

        if len(inlier_pairs) > stop_thresh * len(pairs):
            print(f"Early stopping RANSAC with {len(inlier_pairs)} inliers")
            break

    try:
        H_final = _compute_homography(best_inlier_pairs)
    except np.linalg.LinAlgError:
        H_final = None
        best_inlier_pairs = []
    return H_final


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
        patches_a, patches_b, patch_coords = extract_patches(
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

        patch_pairs = []
        for coords, d in zip(patch_coords, delta_preds):
            dst = np.array(coords) + d.reshape(4, 2)
            patch_pairs.append(list(zip(coords, dst)))

        # compute homography with RANSAC
        H_i = RANSAC_homography(patch_pairs)
        pairwise_H.append(H_i)
    panorama = form_panorama(images, pairwise_H=pairwise_H, graph_mode=False)
    cv2.imwrite("mypano.png", panorama)


if __name__ == "__main__":
    main()
