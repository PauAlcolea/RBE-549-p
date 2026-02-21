#!/usr/bin/env python

import numpy as np
from pathlib import Path
from typing import List
from argparse import ArgumentParser
from EstimateFundamentalMatrix import estimateFundamentalMat
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix
from ExtractCameraPose import extractCameraPose
from NonlinearTriangulation import nonlinearTriangulation, project_point
from DisambiguateCameraPose import disambiguatePose
from Visualization import (
    plot_correspondences,
    plot_triangulation,
    plot_reprojection,
    plot_epipolar_lines,
)


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


def filter_triangulation_outliers(
    inliers: np.ndarray,
    world_points_est: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    inlier_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    filter out triangulated points with high reprojection error or negative depth

    :param inliers: (N, 2, 2) array of correspondences for inlier matches
    :param world_points_est: (N, 3) array of triangulated 3D points from linear triangulation
    :param P1: 3x4 camera matrix for view 1
    :param P2: 3x4 camera matrix for view 2
    :param inlier_thresh: reprojection error threshold for filtering
    :return: filtered inliers and corresponding world points
    """
    rms_thresh = 2.0 * inlier_thresh
    mask_list: list[bool] = []
    # keep points with low reprojection error and positive depth in both views
    for corr, X in zip(inliers, world_points_est):
        x1_uv, x2_uv = corr
        uv1_hat = project_point(P1, X)
        uv2_hat = project_point(P2, X)

        err1 = np.linalg.norm(x1_uv - uv1_hat)
        err2 = np.linalg.norm(x2_uv - uv2_hat)
        rms = np.sqrt(0.5 * (err1**2 + err2**2))

        X_h = np.hstack([X, 1.0])
        _, _, z1 = P1 @ X_h
        _, _, z2 = P2 @ X_h

        ok = (z1 > 0.0) and (z2 > 0.0) and (rms <= rms_thresh)
        mask_list.append(ok)

    mask = np.array(mask_list, dtype=bool)
    if not np.any(mask):
        inliers_filtered = inliers
        world_points_est_filtered = world_points_est
    else:
        inliers_filtered = inliers[mask]
        world_points_est_filtered = world_points_est[mask]

    return inliers_filtered, world_points_est_filtered


def print_outputs(
    current_image_id: int,
    other_image_id: int,
    F: np.ndarray,
    inliers: np.ndarray,
    E: np.ndarray,
    correspondences: np.ndarray,
    poses: List[np.ndarray] = None,
    world_points_est: np.ndarray = None,
    world_points_refined: np.ndarray = None,
    camera_centers: np.ndarray | None = None,
    plot_flags: set[str] | None = None,
) -> None:
    """
    helper for printing and visualizing pipeline outputs
    """
    print(
        f"Estimated Fundamental matrix F (RANSAC inliers) "
        f"for images {current_image_id}-{other_image_id}:"
    )
    print(F)
    print(f"Number of RANSAC inliers: {inliers.shape[0]} / {correspondences.shape[0]}")
    print(
        f"Estimated Essential matrix E (from F) "
        f"for images {current_image_id}-{other_image_id}:"
    )
    print(E)
    print(
        f"Estimated camera poses (from E) for images {current_image_id}-{other_image_id}:"
    )
    if poses is not None:
        for i, pose in enumerate(poses):
            print(f"Pose {i + 1}:")
            print(pose)
    print(
        f"Refined 3D points (from nonlinear triangulation) for images {current_image_id}-{other_image_id}:"
    )
    print(world_points_refined)

    if plot_flags is not None:
        # visualize matches and inliers
        if "i" in plot_flags:
            plot_correspondences(
                current_image_id,
                other_image_id,
                correspondences,
                inliers=inliers,
                title=f"Correspondences between images {current_image_id} and {other_image_id}",
            )

        # visualize triangulated points
        if "t" in plot_flags:
            plot_triangulation(
                world_points_est,
                world_points_refined,
                camera_centers=camera_centers,
                title=f"Triangulated points for images {current_image_id} and {other_image_id}",
                set_labels=["Linear Triangulation", "Nonlinear Triangulation"],
            )

        # visualize reprojection
        if "r" in plot_flags:
            plot_reprojection(
                current_image_id,
                other_image_id,
                inliers,
                poses[0],
                poses[1],
                world_points=world_points_est,
                title=f"Linear Reprojection of 3D points for images {current_image_id} and {other_image_id}",
            )
            plot_reprojection(
                current_image_id,
                other_image_id,
                inliers,
                poses[0],
                poses[1],
                world_points=world_points_refined,
                title=f"Nonlinear Reprojection of 3D points for images {current_image_id} and {other_image_id}",
            )

        # visualize epipolar lines
        if "e" in plot_flags:
            plot_epipolar_lines(
                current_image_id,
                other_image_id,
                F,
                inliers,
                title=f"Epipolar lines for images {current_image_id} and {other_image_id}",
            )


def sfm_pipeline(
    current_image_id: int,
    other_image_id: int,
    n_samples: int = 8,
    n_ransac_iters: int = 1000,
    inlier_thresh: float = 3.0,
    plot_flags: set[str] | None = None,
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
    pose2, world_points_est = disambiguatePose(
        poses, K, inliers, bool(plot_flags) and "p" in plot_flags
    )

    # construct 3x4 pose matrices P1, P2 = K [R | t] for nonlinear refinement
    P1 = K @ pose1
    P2 = K @ pose2

    # remove extreme outliers from triangulated points before nonlinear refinement
    inliers, world_points_est = filter_triangulation_outliers(
        inliers, world_points_est, P1, P2, inlier_thresh
    )

    # nonlinear triangulation to refine 3D points
    world_points_refined = nonlinearTriangulation(inliers, P1, P2, world_points_est)

    # camera centers in world coordinates C1 = (0,0,0), C2 = -R^T t
    C1 = np.zeros(3)
    R2 = pose2[:, :3]
    t2 = pose2[:, 3]
    C2 = -R2.T @ t2
    camera_centers = np.vstack([C1, C2])

    # print, visualize outputs
    print_outputs(
        current_image_id,
        other_image_id,
        F,
        inliers,
        E,
        correspondences,
        [P1, P2],
        world_points_est,
        world_points_refined,
        camera_centers,
        plot_flags,
    )

    return


def main() -> None:
    # TODO - add command line arguments for image ids, RANSAC parameters, etc.

    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--plot",
        nargs="*",
        choices=["i", "t", "r", "p", "e"],
        default=None,
        metavar="FLAG",
        help=(
            "Which plots to show: i=inliers, t=triangulation, r=reprojection, p=possible poses, e=epipolar lines. "
            "If -p is given with no flags, all are shown."
        ),
    )

    args = parser.parse_args()

    # Interpret plotting flags: None => no plotting; [] => all plots.
    if args.plot is None:
        plot_flags = None
    elif len(args.plot) == 0:
        plot_flags = {"i", "t", "r", "p", "e"}
    else:
        plot_flags = set(args.plot)

    sfm_pipeline(
        current_image_id=1,
        other_image_id=2,
        plot_flags=plot_flags,
    )

    ######## NOTES ##########

    # the camera poses are already in the form [R|t], so that can be passed directly as one matrix to the linear triangulation function and things

    # Questions:
    # for the linear triangulation, do i need to make the projective matrices transposes? what is the original equation x1 = PX?
    # What really is the camera pose from ExtractCameraPose.py? is it already the Projection Matrix?
    return


if __name__ == "__main__":
    main()
