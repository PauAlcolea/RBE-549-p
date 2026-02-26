#!/usr/bin/env python

import numpy as np
from pathlib import Path
from typing import List
from argparse import ArgumentParser
from EstimateFundamentalMatrix import estimateFundamentalMat
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix
from ExtractCameraPose import extractCameraPose
from LinearTriangulation import linearTriangulation
from NonlinearTriangulation import nonlinearTriangulation, project_point
from DisambiguateCameraPose import disambiguatePose
from PnPRANSAC import pnpRANSAC
from NonlinearPnP import nonlinearPnP
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
    inlier_thresh: float = 5.0,
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


def build_pnp_correspondences(
    base_inliers: np.ndarray,
    world_points: np.ndarray,
    base_image_id: int,
    new_image_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    helper to build 2D-3D correspondences for PnP

    :param base_inliers: (M, 2, 2) array of inlier correspondences for the
    initial pair of images, where each row is [[u_base, v_base], [u_other, v_other]]
    :param world_points: (M, 3) array of 3D points corresponding to the inliers
    :param base_image_id: id of the base image (1-based)
    :param new_image_id: id of the new image to find matches with (1-based)

    :return (xs, Xs, xs_base): tuple where
    :return xs: (N, 2) array of 2D points in the new image
    :return Xs: (N, 3) array of corresponding 3D world points
    :return corr_base_new: (N, 2, 2) array of all pixel-space correspondences between base and new image
    """

    # load pixel-space correspondences between base image and the new image
    corr_base_new = load_pair_matches(base_image_id, new_image_id)
    if corr_base_new.size == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 2), dtype=float),
        )

    base_points = base_inliers[:, 0, :]  # inlier points in base image

    xs_list: list[np.ndarray] = []  # 2D points in new image
    Xs_list: list[np.ndarray] = []  # corresponding 3D world points
    base_list: list[np.ndarray] = []  # corresponding 2D points in base image

    for uv_base, uv_new in corr_base_new:
        # of correspondences between base and new image, keep those that match pre-computed inliers
        diffs = np.linalg.norm(base_points - uv_base, axis=1)
        idx = np.where(diffs < 5e-3)[0]
        if idx.size == 0:
            continue

        k = int(idx[0])
        xs_list.append(uv_new)
        Xs_list.append(world_points[k])
        base_list.append(base_points[k])

    if not xs_list:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 2), dtype=float),
        )

    xs = np.asarray(xs_list, dtype=float)
    Xs = np.asarray(Xs_list, dtype=float)
    return xs, Xs, corr_base_new


def _pose_from_Rt(R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    helper to construct 3x4 pose matrix P = K [R | t] from rotation R and translation t
    """
    return K @ np.hstack([R, t.reshape(3, 1)])


def _triangulate_for_new_view(
    pose1: np.ndarray,
    R_pnp_refined: np.ndarray,
    t_pnp_refined: np.ndarray,
    corr_base_new: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    helper to use new refined PnP pose to triangulate additional points for single new view

    :param pose1: 3x4 pose matrix [R|t] for initial view
    :param R_pnp_refined: refined rotation from nonlinear PnP for new view
    :param t_pnp_refined: refined translation from nonlinear PnP for new view
    :param corr_base_new: (N, 2, 2) array of pixel-space correspondences between base image and new image, where each row is [[u_base, v_base], [u_new, v_new]]
    :param K: camera intrinsics matrix

    :return: (M, 2, 2) array of inlier correspondences between base and new image, and (M, 3) array of corresponding refined 3D points
    """
    X_est_list = []
    for uv_base, uv_new in corr_base_new:
        X_est = linearTriangulation(
            K,
            pose1,
            np.hstack([R_pnp_refined, t_pnp_refined.reshape(3, 1)]),
            uv_base,
            uv_new,
        )
        X_est_list.append(X_est)
    X_est = np.stack(X_est_list, axis=0)
    P1 = _pose_from_Rt(np.eye(3), np.zeros(3), K)
    P_pnp = _pose_from_Rt(R_pnp_refined, t_pnp_refined, K)
    inliers_new, X_est = filter_triangulation_outliers(
        corr_base_new, X_est, P1, P_pnp, inlier_thresh=4.0
    )
    X_refined = nonlinearTriangulation(inliers_new, P1, P_pnp, X_est)
    return inliers_new, X_refined


def triangulate_additional_points(
    all_poses_Rt: dict[int, np.ndarray],
    new_image_id: int,
    current_image_id: int,
    corr_base_new: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    triangulate additional points for new view with refined PnP pose, using all previous views,
    i.e. call `_triangulate_for_new_view` for the new view and each previous view

    :param all_poses_Rt: dictionary mapping image ids to their 3x4 pose matrices [R|t] in world coordinates
    :param new_image_id: id of the new image to find matches with (1-based)
    :param current_image_id: id of the current image (1-based) that was used for initial triangulation
    :param corr_base_new: (N, 2, 2) array of pixel-space correspondences between base image and new image, where each row is [[u_base, v_base], [u_new, v_new]]
    :param K: camera intrinsics

    :return: list of inlier correspondences and corresponding refined 3D points for each previous view
    """

    all_new_inliers, all_new_Xs = [], []
    for base_img_id in sorted(all_poses_Rt.keys()):
        if base_img_id >= new_image_id:
            continue

        correspondences = (
            corr_base_new
            if base_img_id == current_image_id
            else load_pair_matches(base_img_id, new_image_id)
        )

        inliers_new, X_refined = _triangulate_for_new_view(
            all_poses_Rt[base_img_id],
            all_poses_Rt[new_image_id][:, :3],
            all_poses_Rt[new_image_id][:, 3],
            correspondences,
            K,
        )

        all_new_inliers.append(inliers_new)
        all_new_Xs.append(X_refined)
    return np.concatenate(all_new_inliers, axis=0), np.concatenate(all_new_Xs, axis=0)


def print_outputs(
    current_image_id: int,
    other_image_id: int,
    extra_image_ids: list[int] | None,
    F: np.ndarray,
    inliers: np.ndarray,
    E: np.ndarray,
    correspondences: np.ndarray,
    poses: List[np.ndarray] = None,
    poses_pnp: List[np.ndarray] = None,
    poses_pnp_refined: List[np.ndarray] = None,
    world_points_est: np.ndarray = None,
    world_points_refined: np.ndarray = None,
    world_points_pnp: List[np.ndarray] = None,
    image_points_pnp: List[np.ndarray] = None,
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
            # TODO: also visualize correspondences for additional views??
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
                camera_centers=camera_centers[
                    :2
                ],  # only show camera centers for initial pair
                title=f"Triangulated points for images {current_image_id} and {other_image_id}",
                set_labels=["Linear Triangulation", "Nonlinear Triangulation"],
            )
            plot_triangulation(
                *world_points_pnp,
                camera_centers=camera_centers,
                title=f"PnP points for all views",
            )

        # visualize reprojection
        if "r" in plot_flags:
            for i, pose in enumerate(poses, start=1):
                plot_reprojection(
                    i,
                    inliers[:, 0, :],
                    pose,
                    world_points=world_points_est,
                    title=f"Linear Reprojection of 3D points for image {i}",
                )
                plot_reprojection(
                    i,
                    inliers[:, 0, :],
                    pose,
                    world_points=world_points_refined,
                    title=f"Nonlinear Reprojection of 3D points for image {i}",
                )
            for j, (pose, img_id) in enumerate(zip(poses_pnp, extra_image_ids)):
                plot_reprojection(
                    img_id,
                    image_points_pnp[j][:, 1, :],
                    pose,
                    world_points=world_points_pnp[j],
                    title=f"Linear PnP Reprojection of 3D points for image {img_id}",
                )
            for k, (pose, img_id) in enumerate(zip(poses_pnp_refined, extra_image_ids)):
                plot_reprojection(
                    img_id,
                    image_points_pnp[k][:, 1, :],
                    pose,
                    world_points=world_points_pnp[k],
                    title=f"Nonlinear PnP Reprojection of 3D points for image {img_id}",
                )

        # visualize epipolar lines
        if "e" in plot_flags:
            for i in range(1, len(poses + poses_pnp)):
                plot_epipolar_lines(
                    i,
                    i + 1,
                    F,
                    inliers,
                    title=f"Epipolar lines for images {i} and {i+1}",
                )


def sfm_pipeline(
    current_image_id: int,
    other_image_id: int,
    extra_image_ids: list[int] | None = None,
    plot_flags: set[str] | None = None,
) -> None:
    """
    main wrapper function for SfM pipeline

    :param current_image_id: id of current image (1-based)
    :param other_image_id: id of other image to find matches with (1-based)
    :param extra_image_ids: list of additional image ids to estimate poses for using PnP-RANSAC
    :param plot_flags: optional set of flags to control which plots to show
    """

    # load correspondences for current image and other image
    correspondences = load_pair_matches(current_image_id, other_image_id)

    if correspondences.shape[0] < 8:
        raise ValueError(
            f"Not enough correspondences between image {current_image_id} and {other_image_id}: "
            f"found {correspondences.shape[0]}, need at least 8"
        )

    # estimate fundamental matrix F using inliers from RANSAC
    F = np.zeros((3, 3))
    inliers = getInliersRANSAC(correspondences)
    if inliers.size >= 8:
        F = estimateFundamentalMat(inliers)

    # estimate essential matrix E from F
    K = load_intrinsics()
    E = estimateEssentialMatrix(K, F)

    # estimate 4 possible camera poses from E
    poses = extractCameraPose(E)

    # first camera pose [I | 0]
    pose1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # disambiguate the second camera pose using correspondences
    pose2, world_points_est = disambiguatePose(
        poses, K, inliers, bool(plot_flags) and "p" in plot_flags
    )

    # construct 3x4 pose matrices P1, P2 = K [R | t] for nonlinear refinement
    P1 = K @ pose1
    P2 = K @ pose2
    pose_matrices = [P1, P2]

    # remove extreme outliers from triangulated points before nonlinear refinement
    inliers, world_points_est = filter_triangulation_outliers(
        inliers, world_points_est, P1, P2
    )

    # nonlinear triangulation to refine 3D points
    world_points_refined = nonlinearTriangulation(inliers, P1, P2, world_points_est)

    # for visualization: camera centers in world coordinates C1 = (0,0,0), C2 = -R^T t
    C1 = np.zeros(3)
    R2 = pose2[:, :3]
    t2 = pose2[:, 3]
    C2 = -R2.T @ t2
    camera_centers = [C1, C2]

    # estimate additional camera poses using PnP-RANSAC and nonlinear PnP
    if extra_image_ids is not None:
        # build dictionary of all poses
        all_poses_Rt = {
            current_image_id: pose1,
            other_image_id: pose2,
        }
        pose_matrices_pnp, pose_matrices_pnp_ref = [], []
        image_points_pnp, world_points_pnp = [], []
        for new_image_id in extra_image_ids:
            xs, Xs, corr_base_new = build_pnp_correspondences(
                inliers,
                world_points_refined,
                base_image_id=current_image_id,
                new_image_id=new_image_id,
            )

            if xs.shape[0] < 6:
                print(
                    f"Skipping image {new_image_id}: only {xs.shape[0]}/6 shared 2D-3D matches"
                )
                continue

            # PnP-RANSAC to estimate pose for new view
            R_pnp, t_pnp, xs_inliers, Xs_inliers = pnpRANSAC(xs, Xs, K)

            # non linear pnp
            R_pnp_refined, t_pnp_refined = nonlinearPnP(
                xs_inliers, Xs_inliers, K, R_pnp, t_pnp
            )

            # triangulate additional points for new refined view
            all_poses_Rt[new_image_id] = np.hstack(
                [R_pnp_refined, t_pnp_refined.reshape(3, 1)]
            )
            all_new_inliers, all_new_Xs = triangulate_additional_points(
                all_poses_Rt, new_image_id, current_image_id, corr_base_new, K
            )

            # for visualization: store PnP poses and camera centers
            pose_matrices_pnp.append(_pose_from_Rt(R_pnp, t_pnp, K))
            pose_matrices_pnp_ref.append(_pose_from_Rt(R_pnp_refined, t_pnp_refined, K))
            camera_centers.append(-R_pnp_refined.T @ t_pnp_refined)

            # for visualization: store inliers and corresponding 3D points
            image_points_pnp.append(all_new_inliers)
            world_points_pnp.append(all_new_Xs)

    # print, visualize outputs
    print_outputs(
        current_image_id,
        other_image_id,
        extra_image_ids,
        F,
        inliers,
        E,
        correspondences,
        pose_matrices,
        pose_matrices_pnp,
        pose_matrices_pnp_ref,
        world_points_est,
        world_points_refined,
        world_points_pnp,
        image_points_pnp,
        camera_centers,
        plot_flags,
    )

    return


def main() -> None:
    # TODO - add command line arguments for image ids, etc

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
        extra_image_ids=[3, 4, 5],
        plot_flags=plot_flags,
    )

    return


if __name__ == "__main__":
    main()
