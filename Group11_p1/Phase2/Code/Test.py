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
from GenerateData import _get_random_patch


def load_images(dir):
    images = []
    for filename in sorted(os.listdir(dir)):
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

def RANSAC_homography(
    matching_pairs,
    n_iterations=1000,
    inlier_thresh=5,
    stop_thresh=0.85,
):
    # convert to numpy array of shape (N, 2, 2): [[x1,y1],[x2,y2]]
    pairs_arr = np.asarray(matching_pairs, dtype=np.float32)
    if pairs_arr.size == 0 or pairs_arr.shape[0] < 4:
        return None

    best_inlier_pairs = []
    N = pairs_arr.shape[0]
    for _ in range(n_iterations):
        idx = np.random.choice(N, size=4, replace=False)
        sample = pairs_arr[idx]  # shape (4, 2, 2)

        # skip iteration if selected points are degenerate
        if np.linalg.matrix_rank(_form_A_matrix(sample)) < 8:
            continue

        # compute homography from the four pairs; skip if SVD fails
        try:
            H = _compute_homography(sample)
        except np.linalg.LinAlgError:
            continue

        # use H to map all source points to target and count inliers
        inlier_pairs = []
        for s in pairs_arr:
            (x1, y1), (x2, y2) = s
            p1 = np.array([x1, y1, 1.0]).reshape((3, 1))
            p2_est = H @ p1
            p2_est = (p2_est[:2] / max(p2_est[2], 1e-8)).flatten()
            dist = np.linalg.norm(p2_est - np.array([x2, y2]), ord=1)
            if dist < inlier_thresh:
                inlier_pairs.append(s)

        if len(inlier_pairs) > len(best_inlier_pairs):
            best_inlier_pairs = inlier_pairs

        if len(inlier_pairs) > stop_thresh * N:
            print(f"Early stopping RANSAC with {len(inlier_pairs)} inliers")
            break

    try:
        if len(best_inlier_pairs) < 4:
            return None
        H_final = _compute_homography(np.asarray(best_inlier_pairs, dtype=np.float32))
    except np.linalg.LinAlgError:
        H_final = None
        best_inlier_pairs = []
    return H_final


def extract_patches(images, pairwise_H, num_patches=30):
    """
    get num_patches patch pairs between image i warped to image i+1 frame and image i+1
    """
    all_pair_patches = []
    h, w = images[0].shape[:2]
    for i in range(len(images) - 1):
        H = pairwise_H[i]
        pair_patches = []
        # warp image i to image i+1 frame
        warped_img = cv2.warpPerspective(images[i], H, dsize=(w, h))
        attempts = 0
        while len(pair_patches) < num_patches and attempts < num_patches * 5:
            attempts += 1
            # get patch_a from image i+1
            patch_a, patch_coords = _get_random_patch(images[i + 1])
            if patch_a is None:
                continue
            y1, y2 = patch_coords[0][1], patch_coords[2][1]
            x1, x2 = patch_coords[0][0], patch_coords[2][0]
            patch_b = warped_img[y1:y2, x1:x2]
            if patch_b.shape[:2] != patch_a.shape[:2]:
                continue
            # skip patches where warped image is mostly black (outside source)
            gray_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2GRAY) if patch_b.ndim == 3 else patch_b
            if np.mean(gray_b > 0) < 0.9:
                continue
            pair_patches.append((patch_a, patch_b, patch_coords))
        all_pair_patches.append(pair_patches)
    return all_pair_patches


def patches_to_tensor(patches, device):
    """
    patches: list of (H, W, 3) numpy arrays
    returns: torch tensor (B, 1, H, W)
    """
    tensors = []
    for p in patches:
        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        p = torch.from_numpy(p).unsqueeze(0).float() / 255.0
        tensors.append(p.to(device))
    return torch.stack(tensors)


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser = ArgumentParser(
        description="Test homography model by stitching images in Phase#/Data/DIR"
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Read Phase 1 or Phase 2 Data/ directory",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="Train/Set1",
        help="directory containing images to stitch; relative to Phase#/Data, i.e. 'Train/Set1' or 'Test/unity_hall'",
    )
    args = parser.parse_args()
    data_top_dir = (
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        + f"/Phase{args.phase}/Data/"
    )
    test_data_dir = os.path.join(data_top_dir, args.dir)
    if not os.path.isdir(test_data_dir):
        print(f"Error: directory {test_data_dir} does not exist.")
        return

    # initialize model
    model_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/supervised.pt"
    )
    model = SupervisedHomographyModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # extract patches from images via traditional method
    images = load_images(test_data_dir)
    pairwise_matches = get_matching_features(images)
    pairwise_H_ransac = [RANSAC_homography(m) for m in pairwise_matches]
    all_pair_patches = extract_patches(images, pairwise_H_ransac, num_patches=30)

    # for each image pair, predict residual correction and compose with RANSAC H
    H_pred = []
    with torch.no_grad():
        for idx, pair_patches in enumerate(all_pair_patches):
            all_corr = []
            for patch_a, patch_b, patch_coords in pair_patches:
                xa = patches_to_tensor([patch_a], device=device)
                xb = patches_to_tensor([patch_b], device=device)
                pred_delta = model(xa, xb).detach().cpu().numpy().reshape(4, 2)
                patch_coords = np.asarray(patch_coords, dtype=np.float32)
                pred_coords = patch_coords + pred_delta * NORMALIZING_FACTOR
                corr = np.stack([patch_coords, pred_coords], axis=1)
                all_corr.append(corr)
            all_corr = np.concatenate(all_corr, axis=0)
            print(f"Pair {idx}: fitting correction from {len(all_corr)} correspondences")
            H_correction = _compute_homography(all_corr)
            H_final = H_correction @ pairwise_H_ransac[idx]
            H_pred.append(H_final)

    panorama = form_panorama(images, pairwise_H=H_pred, graph_mode=False)
    cv2.imwrite("mypano.png", panorama)


if __name__ == "__main__":
    main()
