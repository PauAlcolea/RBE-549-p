#!/usr/bin/env python

import numpy as np
from pathlib import Path
from typing import List, NamedTuple
from argparse import ArgumentParser
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix
from ExtractCameraPose import extractCameraPose
from LinearTriangulation import linearTriangulation
from NonlinearTriangulation import nonlinearTriangulation
from DisambiguateCameraPose import disambiguatePose
from PnPRANSAC import pnpRANSAC
from NonlinearPnP import nonlinearPnP
from BuildVisibilityMatrix import visibilityMatrix
from BundleAdjustment import bundleAdjustment
from Visualization import (
    ba_reproj_error,
    base_pair_reproj_errors,
    _compute_reproj_err,
    filter_ba_for_reporting,
    filter_correspondence_set_for_plot,
    filter_triangulation_outliers,
    pnp_reproj_errors,
    point_cloud_mask,
    triangulation_outlier_mask,
    plot_correspondences,
    plot_triangulation,
    plot_reprojection,
    plot_epipolar_lines,
)


class BasePairOutputs(NamedTuple):
    proj_matrices: List[np.ndarray]
    inliers: np.ndarray
    world_linear: np.ndarray
    world_refined: np.ndarray
    camera_centers: List[np.ndarray]


class PnPOutputs(NamedTuple):
    image_ids: List[int]
    proj_matrices_linear: List[np.ndarray]
    proj_matrices_refined: List[np.ndarray]
    world_points: List[np.ndarray]
    image_points: List[np.ndarray]
    world_points_new: List[np.ndarray]
    image_points_new: List[np.ndarray]
    camera_centers: List[np.ndarray]


class BundleAdjustmentOutputs(NamedTuple):
    proj_matrices: List[np.ndarray]
    points: np.ndarray
    observed_points: List[np.ndarray]
    visibility: np.ndarray
    camera_centers: List[np.ndarray]


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


def build_pnp_correspondences(
    base_inliers: np.ndarray,
    world_points: np.ndarray,
    base_image_id: int,
    new_image_id: int,
    thresh: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    helper to build 2D-3D correspondences between base image and new image for PnP,
    by finding overlap in base pair inliers and pixel-space correspondences between base and new image

    :param base_inliers: (M, 2, 2) array of inlier correspondences for the
    initial pair of images, where each row is [[u_base, v_base], [u_other, v_other]]
    :param world_points: (M, 3) array of 3D points corresponding to the inliers
    :param base_image_id: id of the base image (1-based)
    :param new_image_id: id of the new image to find matches with (1-based)
    :param thresh: pixel threshold for matching known inlier points with correspondences in base image

    :return (xs, Xs, xs_base): tuple where
    :return xs: (N, 2) array of 2D points in the new image
    :return Xs: (N, 3) array of corresponding 3D world points
    :return matches_w_base: (N, 2, 2) array of all pixel-space correspondences between base and new image
    """

    # load pixel-space correspondences between base image and the new image
    matches_w_base = load_pair_matches(base_image_id, new_image_id)
    if matches_w_base.size == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 2), dtype=float),
        )

    base_points = base_inliers[:, 0, :]  # inlier points in base image

    xs_list: list[np.ndarray] = []  # 2D points in new image
    Xs_list: list[np.ndarray] = []  # corresponding 3D world points
    base_list: list[np.ndarray] = []  # corresponding 2D points in base image

    for uv_base, uv_new in matches_w_base:
        # of correspondences between base and new image, keep those that match pre-computed inliers
        diffs = np.linalg.norm(base_points - uv_base, axis=1)
        idx = np.where(diffs < thresh)[0]
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
    return xs, Xs, matches_w_base


def _proj_from_Rt(R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    helper to construct 3x4 projection matrix P = K [R | t] from rotation R and translation t
    """
    return K @ np.hstack([R, t.reshape(3, 1)])


def _triangulate_for_new_view(
    pose1: np.ndarray,
    R_pnp_refined: np.ndarray,
    t_pnp_refined: np.ndarray,
    matches_w_base: np.ndarray,
    K: np.ndarray,
    filter_thresh: float = 3.0,
) -> np.ndarray:
    """
    helper to use new refined PnP pose to triangulate additional points for single new view

    :param pose1: 3x4 pose matrix [R|t] for initial view
    :param R_pnp_refined: refined rotation from nonlinear PnP for new view
    :param t_pnp_refined: refined translation from nonlinear PnP for new view
    :param matches_w_base: (N, 2, 2) array of pixel-space correspondences between base image and new image, where each row is [[u_base, v_base], [u_new, v_new]]
    :param K: camera intrinsics matrix
    :param filter_thresh: threshold for filtering triangulated points based on distance from median, as a multiple of the median distance

    :return: (M, 2, 2) array of inlier correspondences between base and new image, and (M, 3) array of corresponding refined 3D points
    """
    X_est_list = []
    for uv_base, uv_new in matches_w_base:
        X_est = linearTriangulation(
            K,
            pose1,
            np.hstack([R_pnp_refined, t_pnp_refined.reshape(3, 1)]),
            uv_base,
            uv_new,
        )
        X_est_list.append(X_est)
    X_est = np.stack(X_est_list, axis=0)
    P1 = K @ pose1
    P_pnp = _proj_from_Rt(R_pnp_refined, t_pnp_refined, K)
    inliers_new, X_est = filter_triangulation_outliers(
        matches_w_base, X_est, P1, P_pnp, inlier_thresh=4.0
    )
    X_refined = nonlinearTriangulation(inliers_new, P1, P_pnp, X_est)
    # filter outliers based on distance from median
    r = np.linalg.norm(X_refined, axis=1)
    r_med = np.median(r)
    mask = (r > 0) & (r < filter_thresh * r_med)
    X_refined = X_refined[mask]
    inliers_new = inliers_new[mask]

    return inliers_new, X_refined


def triangulate_additional_points(
    all_poses_Rt: dict[int, np.ndarray],
    new_image_id: int,
    current_image_id: int,
    matches_w_base: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    triangulate additional points for new view with refined PnP pose, using all previous views,
    i.e. call `_triangulate_for_new_view` for the new view and each previous view

    :param all_poses_Rt: dictionary mapping image ids to their 3x4 pose matrices [R|t] in world coordinates
    :param new_image_id: id of the new image to find matches with (1-based)
    :param current_image_id: id of the current image (1-based) that was used for initial triangulation
    :param matches_w_base: (N, 2, 2) array of pixel-space correspondences between base image and new image, where each row is [[u_base, v_base], [u_new, v_new]]
    :param K: camera intrinsics

    :return: list of inlier correspondences and corresponding refined 3D points for each previous view
    """

    all_new_inliers, all_new_Xs = [], []
    for base_img_id in sorted(all_poses_Rt.keys()):
        if base_img_id >= new_image_id:
            continue

        correspondences = (
            matches_w_base
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


def _build_base_pair_outputs(
    inliers: np.ndarray,
    world_linear: np.ndarray,
    world_refined: np.ndarray,
    proj_matrices: List[np.ndarray],
    camera_centers: List[np.ndarray],
) -> BasePairOutputs:
    linear_mask = triangulation_outlier_mask(
        inliers, world_linear, proj_matrices[0], proj_matrices[1]
    )
    refined_mask = triangulation_outlier_mask(
        inliers, world_refined, proj_matrices[0], proj_matrices[1]
    )
    plot_mask = (
        linear_mask
        & refined_mask
        & point_cloud_mask(world_linear)
        & point_cloud_mask(world_refined)
    )
    if not np.any(plot_mask):
        plot_mask = refined_mask & point_cloud_mask(world_refined)
    if not np.any(plot_mask):
        plot_mask = np.ones(len(inliers), dtype=bool)

    return BasePairOutputs(
        proj_matrices=proj_matrices,
        inliers=inliers[plot_mask],
        world_linear=world_linear[plot_mask],
        world_refined=world_refined[plot_mask],
        camera_centers=camera_centers,
    )


def _build_pnp_outputs(
    image_ids: List[int],
    proj_matrices_linear: List[np.ndarray],
    proj_matrices_refined: List[np.ndarray],
    world_points: List[np.ndarray],
    image_points: List[np.ndarray],
    world_points_new: List[np.ndarray],
    image_points_new: List[np.ndarray],
    camera_centers: List[np.ndarray],
) -> PnPOutputs:
    world_points_report: List[np.ndarray] = []
    image_points_report: List[np.ndarray] = []

    for matches_uv, points_xyz in zip(image_points_new, world_points_new):
        matches_uv_filt, points_xyz_filt = filter_correspondence_set_for_plot(
            matches_uv, points_xyz
        )
        image_points_report.append(matches_uv_filt)
        world_points_report.append(points_xyz_filt)

    return PnPOutputs(
        image_ids=image_ids,
        proj_matrices_linear=proj_matrices_linear,
        proj_matrices_refined=proj_matrices_refined,
        world_points=world_points,
        image_points=image_points,
        world_points_new=world_points_report,
        image_points_new=image_points_report,
        camera_centers=camera_centers,
    )


def _build_ba_outputs(
    proj_matrices: List[np.ndarray],
    points: np.ndarray,
    observed_points: List[np.ndarray],
    visibility: np.ndarray,
    camera_centers: List[np.ndarray],
) -> BundleAdjustmentOutputs:
    points_report, observations_report, visibility_report = filter_ba_for_reporting(
        points, observed_points, visibility
    )
    return BundleAdjustmentOutputs(
        proj_matrices=proj_matrices,
        points=points_report,
        observed_points=observations_report,
        visibility=visibility_report,
        camera_centers=camera_centers,
    )


def print_outputs(
    current_image_id: int,
    other_image_id: int,
    extra_image_ids: list[int] | None,
    F: np.ndarray,
    E: np.ndarray,
    correspondences: np.ndarray,
    base_pair_outputs: BasePairOutputs,
    pnp_outputs: PnPOutputs,
    ba_outputs: BundleAdjustmentOutputs,
    plot_flags: set[str] | None = None,
) -> None:
    """
    helper for printing and visualizing pipeline outputs
    """
    base_pair_err_linear, base_pair_err_refined = base_pair_reproj_errors(
        base_pair_outputs
    )
    pnp_err_linear, pnp_err_refined = pnp_reproj_errors(pnp_outputs)
    ba_err = ba_reproj_error(ba_outputs)

    print(
        "Linear triangulation reprojection error "
        f"(mean across both views): {base_pair_err_linear:.4f} px"
    )
    print(
        "Nonlinear triangulation reprojection error "
        f"(mean across both views): {base_pair_err_refined:.4f} px"
    )

    if extra_image_ids:
        for img_id, err_lin, err_ref in zip(
            pnp_outputs.image_ids,
            pnp_err_linear,
            pnp_err_refined,
        ):
            print(f"Linear PnP reprojection error for image {img_id}: {err_lin:.4f} px")
            print(
                f"Nonlinear PnP reprojection error for image {img_id}: {err_ref:.4f} px"
            )

    print(
        "Bundle adjustment reprojection error "
        f"(mean across all visible observations): {ba_err:.4f} px"
    )

    if plot_flags is None:
        return

    # visualize matches and inliers
    if "i" in plot_flags:
        plot_correspondences(
            current_image_id,
            other_image_id,
            correspondences,
            inliers=base_pair_outputs.inliers,
            title=f"Correspondences between images {current_image_id} and {other_image_id}",
        )

    # visualize triangulated points
    if "t" in plot_flags:
        plot_triangulation(
            base_pair_outputs.world_linear,
            base_pair_outputs.world_refined,
            camera_centers=base_pair_outputs.camera_centers,
            title=f"Triangulated points for images {current_image_id} and {other_image_id}",
            set_labels=["Linear Triangulation", "Nonlinear Triangulation"],
        )

        plot_triangulation(
            *pnp_outputs.world_points,
            camera_centers=pnp_outputs.camera_centers,
            title=f"PnP points for all views",
        )

        plot_triangulation(
            ba_outputs.points,
            camera_centers=ba_outputs.camera_centers,
            title=f"Bundle Adjustment points for all views",
        )

        pre_ba_sets = [base_pair_outputs.world_refined] + pnp_outputs.world_points
        pre_ba_points = (
            np.concatenate(pre_ba_sets, axis=0)
            if len(pre_ba_sets) > 1
            else pre_ba_sets[0]
        )
        pre_ba_points = pre_ba_points[
            point_cloud_mask(pre_ba_points, filter_thresh=3.0)
        ]

        plot_triangulation(
            pre_ba_points,
            ba_outputs.points,
            camera_centers=ba_outputs.camera_centers,
            title=f"PnP and Bundle Adjustment points for all views",
            set_labels=["pre-Bundle Adjustment", "post-Bundle Adjustment"],
        )

    # visualize reprojection
    if "r" in plot_flags:
        for i, pose in enumerate(base_pair_outputs.proj_matrices, start=1):
            observed = base_pair_outputs.inliers[:, i - 1, :]
            plot_reprojection(
                i,
                observed,
                pose,
                world_points=base_pair_outputs.world_linear,
                title=f"Linear Reprojection of 3D points for image {i}",
            )
            plot_reprojection(
                i,
                observed,
                pose,
                world_points=base_pair_outputs.world_refined,
                title=f"Nonlinear Reprojection of 3D points for image {i}",
            )

        if extra_image_ids:
            for j, (pose, img_id) in enumerate(
                zip(pnp_outputs.proj_matrices_linear, pnp_outputs.image_ids)
            ):
                plot_reprojection(
                    img_id,
                    pnp_outputs.image_points_new[j][:, 1, :],
                    pose,
                    world_points=pnp_outputs.world_points_new[j],
                    title=f"Linear PnP Reprojection of 3D points for image {img_id}",
                )

            for k, (pose, img_id) in enumerate(
                zip(pnp_outputs.proj_matrices_refined, pnp_outputs.image_ids)
            ):
                plot_reprojection(
                    img_id,
                    pnp_outputs.image_points_new[k][:, 1, :],
                    pose,
                    world_points=pnp_outputs.world_points_new[k],
                    title=f"Nonlinear PnP Reprojection of 3D points for image {img_id}",
                )

        ba_image_ids = [current_image_id, other_image_id] + pnp_outputs.image_ids
        for cam_idx, (pose, img_id) in enumerate(
            zip(ba_outputs.proj_matrices, ba_image_ids)
        ):
            plot_reprojection(
                img_id,
                ba_outputs.observed_points[cam_idx],
                pose,
                world_points=ba_outputs.points[
                    np.where(ba_outputs.visibility[cam_idx] == 1)[0]
                ],
                title=f"Bundle Adjustment Reprojection of 3D points for image {img_id}",
            )

    # visualize epipolar lines
    if "e" in plot_flags:
        total_pose_count = len(base_pair_outputs.proj_matrices) + len(
            pnp_outputs.proj_matrices_linear
        )
        for i in range(1, total_pose_count):
            plot_epipolar_lines(
                i,
                i + 1,
                F,
                base_pair_outputs.inliers,
                title=f"Epipolar lines for images {i} and {i+1}",
            )


def sfm_pipeline(
    current_image_id: int,
    other_image_id: int,
    extra_image_ids: list[int] | None = None,
    plot_flags: set[str] | None = None,
    verbose: bool = False,
) -> None:
    """
    main wrapper function for SfM pipeline

    :param current_image_id: id of current image (1-based)
    :param other_image_id: id of other image to find matches with (1-based)
    :param extra_image_ids: list of additional image ids to estimate poses for using PnP-RANSAC
    :param plot_flags: optional set of flags to control which plots to show
    :param verbose: whether to print outputs along the way
    """

    # load correspondences for current image and other image
    base_pair_matches = load_pair_matches(current_image_id, other_image_id)

    if base_pair_matches.shape[0] < 8:
        raise ValueError(
            f"Not enough correspondences between image {current_image_id} and {other_image_id}: "
            f"found {base_pair_matches.shape[0]}, need at least 8"
        )

    # estimate fundamental matrix F using inliers from RANSAC
    F, base_pair_inlier_matches = getInliersRANSAC(
        correspondences_2d=base_pair_matches, inlier_thresh=40.0
    )

    # estimate essential matrix E from F
    K = load_intrinsics()
    E = estimateEssentialMatrix(K, F)

    # estimate 4 possible camera poses from E
    candidate_poses = extractCameraPose(E)

    # initialize first view to have identity rotation and zero translation (i.e. camera center at origin)
    pose1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # disambiguate the second camera pose with linear triangulation/chirality check
    pose2, base_pair_XYZ_est = disambiguatePose(
        camera_poses=candidate_poses,
        K=K,
        correspondences_2d=base_pair_inlier_matches,
        plot=bool(plot_flags) and "p" in plot_flags,
    )

    # construct 3x4 projection matrices K [R | t] for nonlinear refinement
    base_P1 = K @ pose1
    base_P2 = K @ pose2
    base_pair_proj_matrices = [base_P1, base_P2]

    # nonlinear triangulation to refine 3D points
    base_pair_XYZ_ref = nonlinearTriangulation(
        base_pair_inlier_matches, base_P1, base_P2, base_pair_XYZ_est
    )

    # for visualization: camera centers in world coordinates C1 = (0,0,0), C2 = -R^T t
    C1 = np.zeros(3)
    R2 = pose2[:, :3]
    t2 = pose2[:, 3]
    C2 = -R2.T @ t2
    initial_camera_centers = [C1, C2]

    # projection matrices, point correspondences, and camera centers used for PnP
    proj_matrices_pnp: List[np.ndarray] = []
    proj_matrices_pnp_ref: List[np.ndarray] = []
    inliers_pnp_uv: List[np.ndarray] = []
    inliers_pnp_XYZ: List[np.ndarray] = []
    pnp_points_uv: List[np.ndarray] = []
    pnp_points_XYZ: List[np.ndarray] = []
    camera_centers_pnp = initial_camera_centers.copy()
    pnp_image_ids: List[int] = []
    # for added views: PnP/nonlinear PnP to estimate pose, then triangulate additional points with all previous views
    if extra_image_ids is not None:
        # build dictionary of all poses [R|t]
        all_poses_Rt = {
            current_image_id: pose1,
            other_image_id: pose2,
        }
        for new_image_id in extra_image_ids:
            uv_new_view, XYZ_new_view, matches_w_base = build_pnp_correspondences(
                base_inliers=base_pair_inlier_matches,
                world_points=base_pair_XYZ_ref,
                base_image_id=current_image_id,
                new_image_id=new_image_id,
                thresh=1,
            )
            if verbose:
                print(
                    f"Image {new_image_id}: found {uv_new_view.shape[0]}/{matches_w_base.shape[0]} 2D-3D correspondences for PnP"
                )

            if uv_new_view.shape[0] < 6:
                print(
                    f"Skipping image {new_image_id}: only {uv_new_view.shape[0]}/6 shared 2D-3D matches"
                )
                continue

            # PnP-RANSAC to estimate pose for new view
            linear_pnp_err = np.inf
            while linear_pnp_err > 1000:
                R_pnp_est, t_pnp_est, inliers_pnp_uv_tmp, inliers_pnp_XYZ_tmp = (
                    pnpRANSAC(uv_new_view, XYZ_new_view, K, inlier_thresh=40)
                )
                proj_pnp_est = _proj_from_Rt(R_pnp_est, t_pnp_est, K)
                linear_pnp_err = _compute_reproj_err(
                    inliers_pnp_XYZ_tmp, inliers_pnp_uv_tmp, proj_pnp_est
                )

            # non linear pnp to refine pose for new view
            R_pnp_refined, t_pnp_refined = nonlinearPnP(
                inliers_pnp_uv_tmp, inliers_pnp_XYZ_tmp, K, R_pnp_est, t_pnp_est
            )

            # triangulate additional points for new refined view
            all_poses_Rt[new_image_id] = np.hstack(
                [R_pnp_refined, t_pnp_refined.reshape(3, 1)]
            )
            new_matches_uv, new_triangulated_XYZ = triangulate_additional_points(
                all_poses_Rt, new_image_id, current_image_id, matches_w_base, K
            )

            # for visualization: store PnP projection matrices and camera centers
            proj_matrices_pnp.append(proj_pnp_est)
            proj_matrices_pnp_ref.append(_proj_from_Rt(R_pnp_refined, t_pnp_refined, K))
            camera_centers_pnp.append(-R_pnp_refined.T @ t_pnp_refined)
            pnp_image_ids.append(new_image_id)

            # for visualization: store inliers and corresponding 3D points
            inliers_pnp_uv.append(inliers_pnp_uv_tmp)
            inliers_pnp_XYZ.append(inliers_pnp_XYZ_tmp)
            pnp_points_uv.append(new_matches_uv)
            pnp_points_XYZ.append(new_triangulated_XYZ)

    # combine all camera projection matrices K [R | t]: initial pair + additional views
    all_proj_matrices = base_pair_proj_matrices + proj_matrices_pnp_ref

    # world points for all views
    all_world_points = np.concatenate([base_pair_XYZ_ref] + pnp_points_XYZ, axis=0)

    # build visibility matrix and 2D point correspondences for bundle adjustment
    # each camera observes specific 3D points based on triangulation

    # for initial pair: both observe refined triangulated points from base pair
    points_for_poses = [base_pair_XYZ_ref, base_pair_XYZ_ref]
    # for additional views: each observes the world points triangulated for that view
    points_for_poses.extend(pnp_points_XYZ)

    # build visibility matrix where V[i, j] = 1 if camera i observes world point j, else 0
    V = visibilityMatrix(all_world_points, points_for_poses)
    if verbose:
        print("Number of world points:", all_world_points.shape[0])
        print("Number of observations (nonzero entries in V):", np.sum(V))
        print(
            "Average number of points observed per camera:",
            np.sum(V) / len(all_proj_matrices),
        )

    # Organize 2D observations for bundle adjustment
    # Each views's 2D points must correspond to 3D points where V[i, :] == 1
    all_image_points = []
    all_image_points.append(base_pair_inlier_matches[:, 0, :])  # view 1
    all_image_points.append(base_pair_inlier_matches[:, 1, :])  # view 2
    # for additional views, use the inliers from PnP triangulation which correspond to the points they observe
    all_image_points.extend(i[:, 1, :] for i in pnp_points_uv)
    all_image_points = [np.asarray(pts, dtype=float) for pts in all_image_points]

    # bundle adjustment to refine all camera poses and 3D points
    final_poses, final_points = bundleAdjustment(
        K, V, all_proj_matrices, all_world_points, all_image_points
    )

    # for visualization: update camera centers for poses refined by bundle adjustment
    refined_camera_centers = []
    for Rt in final_poses:
        R, t = Rt[:, :3], Rt[:, 3]
        C = -R.T @ t
        refined_camera_centers.append(C)
    final_proj_matrices = [_proj_from_Rt(Rt[:, :3], Rt[:, 3], K) for Rt in final_poses]

    # for visualization: organize outputs into named tuples for cleaner print function
    base_pair_outputs = _build_base_pair_outputs(
        inliers=base_pair_inlier_matches,
        world_linear=base_pair_XYZ_est,
        world_refined=base_pair_XYZ_ref,
        proj_matrices=base_pair_proj_matrices,
        camera_centers=initial_camera_centers,
    )

    pnp_outputs = _build_pnp_outputs(
        image_ids=pnp_image_ids,
        proj_matrices_linear=proj_matrices_pnp,
        proj_matrices_refined=proj_matrices_pnp_ref,
        world_points=inliers_pnp_XYZ,
        image_points=inliers_pnp_uv,
        world_points_new=pnp_points_XYZ,
        image_points_new=pnp_points_uv,
        camera_centers=camera_centers_pnp,
    )

    ba_outputs = _build_ba_outputs(
        proj_matrices=final_proj_matrices,
        points=final_points,
        observed_points=all_image_points,
        visibility=V,
        camera_centers=refined_camera_centers,
    )

    # print, visualize outputs
    print_outputs(
        current_image_id,
        other_image_id,
        extra_image_ids,
        F,
        E,
        base_pair_matches,
        base_pair_outputs,
        pnp_outputs,
        ba_outputs,
        plot_flags,
    )

    return


def main() -> None:
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print additional information while running",
    )

    args = parser.parse_args()

    # Interpret plotting flags: None => no plotting; [] => all plots.
    if args.plot is None:
        plot_flags = None
    elif len(args.plot) == 0:
        plot_flags = {"i", "t", "r", "p", "e"}
    else:
        plot_flags = set(args.plot)

    while True:
        try:
            sfm_pipeline(
                current_image_id=1,
                other_image_id=2,
                extra_image_ids=[3, 4, 5],
                plot_flags=plot_flags,
                verbose=args.verbose,
            )
        except IndexError:
            if args.verbose:
                print("Error encountered in bundle adjustment; retrying...")
            continue
        break

    return


if __name__ == "__main__":
    main()
