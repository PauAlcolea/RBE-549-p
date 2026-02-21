#!/usr/bin/env python

import numpy as np
from pathlib import Path
from typing import List
from EstimateFundamentalMatrix import estimateFundamentalMat
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix
from ExtractCameraPose import extractCameraPose
from NonlinearTriangulation import nonlinearTriangulation
from DisambiguateCameraPose import disambiguatePose


def load_pair_matches(
    current_image_id: int,
    other_image_id: int,
) -> np.ndarray:
    """
    load point correspondences between current image and other image where
    each row in the current image's matching file has the format:

    n_matches  R G B  u_i  v_i  (image_id  u  v)*

    return all correspondences where other_image_id is found in (image_id, u, v) entry
    :param current_image_id: id of current image (1-based)
    :param other_image_id: id of other image to find matches with (1-based)
    :return: (N, 2, 2) array of correspondences, where each row is [[u_i, v_i], [u_j, v_j]]
    """

    phase1_dir = Path(__file__).resolve().parents[1]
    data_dir = phase1_dir / "Data"
    matching_file = (data_dir / f"matching{current_image_id}.txt").resolve()
    if not matching_file.exists():
        raise FileNotFoundError(f"Could not find matching file: {matching_file}")

    correspondences: List[np.ndarray] = []

    with matching_file.open("r") as f:
        # ignore first line
        _ = f.readline()
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

    if len(k_list) != 9:
        raise ValueError(
            f"Expected 9 intrinsics values in {calibration_path}, found {len(k_list)}"
        )

    K = np.array(k_list).reshape((3, 3))
    return K


def print_outputs(
    current_image_id: int,
    other_image_id: int,
    F: np.ndarray,
    inliers: np.ndarray,
    E: np.ndarray,
    num_correspondences: int,
    poses: List[np.ndarray] = None,
    world_points_refined: np.ndarray = None,
) -> None:
    """
    helper for printing outputs of pipeline
    """
    print(
        f"Estimated Fundamental matrix F (RANSAC inliers) "
        f"for images {current_image_id}-{other_image_id}:"
    )
    print(F)
    print(f"Number of RANSAC inliers: {inliers.shape[0]} / {num_correspondences}")
    print(
        f"Estimated Essential matrix E (from F) "
        f"for images {current_image_id}-{other_image_id}:"
    )
    print(E)
    print(
        f"Estimated camera poses (from E) for images {current_image_id}-{other_image_id}:"
    )
    for i, pose in enumerate(poses):
        print(f"Pose {i + 1}:")
        print(pose)
    print(
        f"Refined 3D points (from nonlinear triangulation) for images {current_image_id}-{other_image_id}:"
    )
    print(world_points_refined)


def sfm_pipeline(
    current_image_id: int,
    other_image_id: int,
    n_samples: int = 8,
    n_ransac_iters: int = 1000,
    inlier_thresh: float = 1.0,
) -> None:
    # load correspondences for current image and other image
    correspondences = load_pair_matches(current_image_id, other_image_id)

    if correspondences.shape[0] < n_samples:
        raise ValueError(
            f"Not enough correspondences between image {current_image_id} and {other_image_id}: "
            f"found {correspondences.shape[0]}, need at least {n_samples}"
        )

    # estimate fundamental matrix F using inliers from RANSAC
    F = np.zeros((3, 3))
    inliers = getInliersRANSAC(correspondences, n_ransac_iters, inlier_thresh)
    if inliers.size >= 8:
        F = estimateFundamentalMat(inliers)

    # estimate essential matrix E from F
    K = load_intrinsics()
    E = estimateEssentialMatrix(K, F)

    # estimate 4 possible camera poses from E
    poses = extractCameraPose(E, K)

    # first camera pose [I | 0]
    pose1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # disambiguate the second camera pose using correspondences
    pose2, world_point_est = disambiguatePose(poses, K, inliers)

    # TODO: visulize triangulated points

    # construct 3x4 pose matrices P1, P2 = K [R | t] for nonlinear refinement
    P1 = K @ pose1
    P2 = K @ pose2

    # TODO: nonlinear triangulation to refine 3D points
    world_points_refined = nonlinearTriangulation(inliers, P1, P2, world_point_est)

    # print outputs
    print_outputs(
        current_image_id,
        other_image_id,
        F,
        inliers,
        E,
        correspondences.shape[0],
        poses,
        world_points_refined,
    )

    return


def main() -> None:
    # TODO - add command line arguments for image ids, RANSAC parameters, etc.
    # TODO: visualize inliers and epipolar lines on the images

    sfm_pipeline(
        current_image_id=1,
        other_image_id=2,
    )

    ######## NOTES ##########

    # when calling estimateFundamentalMat(correspondances), the correspondances should be an np.array of shape (8, 2, 2)
    # it should look like this:
    # np.array([[[x1, y1], [x1', y1']], [[x2, y2], [x2', y2']], [[x3, y3], [x3', y3']] , ... , ... [[x8, y8], [x8', y8']] ])

    # the camera poses are already in the form [R|t], so that can be passed directly as one matrix to the linear triangulation function and things

    # Questions:
    # for the linear triangluation, do i need to make the projective matrices transposes? what is the original equation x1 = PX?
    # What really is the camera pose from ExtractCameraPose.py? is it already the Projection Matrix?
    # There's a chance that the the pose is extracted wrong, maybe it should be a 4x4, not a 3x4, look at the notes,
    return


if __name__ == "__main__":
    main()
