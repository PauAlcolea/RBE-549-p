#!/usr/bin/env python

import numpy as np
from pathlib import Path
from typing import List
from EstimateFundamentalMatrix import estimateFundamentalMat
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix


def load_pair_matches(
    current_image_id: int,
    other_image_id: int,
) -> np.ndarray:
    """
    load point correspondences between current image and other image.

    for the current image i, each row in the matching file has the format:

        n_matches  R G B  u_i  v_i  (image_id  u  v)*

    save correspondence ((u_i, v_i), (u_j, v_j)) for each row where
    one of the (image_id, u, v) entries matches other_image_id
    """

    phase1_dir = Path(__file__).resolve().parents[1]
    data_dir = phase1_dir / "Data"
    matching_file = (data_dir / f"matching{current_image_id}.txt").resolve()
    if not matching_file.exists():
        raise FileNotFoundError(f"Could not find matching file: {matching_file}")

    correspondences: List[np.ndarray] = []

    with matching_file.open("r") as f:
        # ignore first line
        first_line = f.readline()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            n_matches = int(parts[0])
            # count, R, G, B, u_curr, v_curr, then 3 * (n_matches - 1) entries
            if len(parts) < 6:
                continue

            u_curr = float(parts[4])
            v_curr = float(parts[5])

            # remaining tokens contain (image_id, u, v) entries for other images
            id_point_pairs = parts[6:]
            remaining_tokens = 3 * max(n_matches - 1, 0)
            if len(id_point_pairs) < remaining_tokens:
                continue

            for k in range(max(n_matches - 1, 0)):
                base = 3 * k
                img_id = int(id_point_pairs[base])
                u_other = float(id_point_pairs[base + 1])
                v_other = float(id_point_pairs[base + 2])

                if img_id == other_image_id:
                    correspondences.append(
                        np.array([[u_curr, v_curr], [u_other, v_other]], dtype=float)
                    )

    if not correspondences:
        return np.empty((0, 2, 2), dtype=float)

    return np.stack(correspondences, axis=0)


def load_intrinsics() -> np.ndarray:
    """
    load intrinsics matrix K from calibration.txt
    """
    phase1_dir = Path(__file__).resolve().parents[1]
    calibration_path = phase1_dir / "Data" / "calibration.txt"

    with calibration_path.open("r") as f:
        k_list = []
        for line in f:
            parts = line.strip().split()
            k_list.extend([float(x) for x in parts])
        K = np.array(k_list).reshape((3, 3))
        return K


def sfm_pipeline(
    current_image_id: int,
    other_image_id: int,
    n_samples: int = 8,
    n_ransac_iters: int = 1000,
    inlier_thresh: float = 1.0,
) -> None:
    correspondences = load_pair_matches(current_image_id, other_image_id)

    if correspondences.shape[0] < n_samples:
        raise ValueError(
            f"Not enough correspondences between image {current_image_id} and {other_image_id}: "
            f"found {correspondences.shape[0]}, need at least {n_samples}"
        )

    inliers = getInliersRANSAC(correspondences, n_ransac_iters, inlier_thresh)

    if inliers.size >= 8:
        F = estimateFundamentalMat(inliers)
        print(
            f"Estimated Fundamental matrix F (RANSAC inliers) "
            f"for images {current_image_id}-{other_image_id}:"
        )
        print(F)
        print(
            f"Number of RANSAC inliers: {inliers.shape[0]} / {correspondences.shape[0]}"
        )

    # evaluate simple epipolar constraint error |x2^T F x1| on all matches for this pair
    # errors = []
    # for (x1, y1), (x2, y2) in correspondences:
    #     p1 = np.array([x1, y1, 1.0])
    #     p2 = np.array([x2, y2, 1.0])
    #     err = float(abs(p2.T @ F @ p1))
    #     errors.append(err)

    # errors = np.array(errors)
    # print("Epipolar constraint |x2^T F x1| statistics over all matches:")
    # print(f"  count = {errors.size}")
    # print(f"  mean  = {errors.mean():.6f}")
    # print(f"  median= {np.median(errors):.6f}")
    # print(f"  max   = {errors.max():.6f}")

    K = load_intrinsics()
    E = estimateEssentialMatrix(K, F)
    print(
        f"Estimated Essential matrix E (from F) "
        f"for images {current_image_id}-{other_image_id}:"
    )
    print(E)


def main() -> None:
    # TODO - add command line arguments for image ids, RANSAC parameters, etc.
    # TODO: visualize inliers and epipolar lines on the images

    sfm_pipeline(
        current_image_id=1,
        other_image_id=2,
    )


if __name__ == "__main__":
    main()
