#!/usr/bin/env python

import numpy as np
from typing import List
from argparse import ArgumentParser
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import estimateEssentialMatrix
from ExtractCameraPose import extractCameraPose
from NonlinearTriangulation import nonlinearTriangulation
from DisambiguateCameraPose import disambiguatePose
from PnPRANSAC import pnpRANSAC
from NonlinearPnP import nonlinearPnP
from BuildVisibilityMatrix import visibilityMatrix
from BundleAdjustment import bundleAdjustment
from Visualization import _compute_reproj_err
from Utils import (
    load_pair_matches,
    load_intrinsics,
    build_pnp_correspondences,
    triangulate_additional_points,
    _proj_from_Rt,
    _build_base_pair_outputs,
    _build_pnp_outputs,
    _build_ba_outputs,
    print_outputs,
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
        choices=["i", "t", "r", "p"],
        default=None,
        metavar="FLAG",
        help=(
            "Which plots to show: i=inliers, t=triangulation, r=reprojection, p=possible poses"
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
        plot_flags = {"i", "t", "r", "p"}
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
