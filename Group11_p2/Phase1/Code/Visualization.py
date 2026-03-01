import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List
from NonlinearTriangulation import project_point


######## filtering helper functions to avoid having to zoom in the plots ########
def filter_triangulation_outliers(
    inliers: np.ndarray,
    world_points_est: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    inlier_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    filter out triangulated points with high reprojection error or negative depth
    """
    rms_thresh = 2.0 * inlier_thresh
    mask_list: list[bool] = []

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

        mask_list.append((z1 > 0.0) and (z2 > 0.0) and (rms <= rms_thresh))

    mask = np.array(mask_list, dtype=bool)
    if not np.any(mask):
        return inliers, world_points_est
    return inliers[mask], world_points_est[mask]


def triangulation_outlier_mask(
    inliers: np.ndarray,
    world_points: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    inlier_thresh: float = 4.0,
) -> np.ndarray:
    filtered_inliers, _ = filter_triangulation_outliers(
        inliers, world_points, P1, P2, inlier_thresh=inlier_thresh
    )
    if len(filtered_inliers) == len(inliers):
        return np.ones(len(inliers), dtype=bool)
    return np.isin(inliers, filtered_inliers).all(axis=(1, 2))


def point_cloud_mask(points: np.ndarray, filter_thresh: float = 3.0) -> np.ndarray:
    """
    build a plot/error-only mask that removes gross spatial outliers.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    finite_mask = np.all(np.isfinite(points), axis=1)
    radii = np.linalg.norm(points, axis=1)
    positive_radii = radii[(radii > 0) & finite_mask]
    if len(positive_radii) == 0:
        return finite_mask

    r_med = np.median(positive_radii)
    radius_mask = (radii > 0) & (radii < filter_thresh * r_med)
    mask = finite_mask & radius_mask
    if not np.any(mask):
        return finite_mask
    return mask


def filter_correspondence_set_for_plot(
    correspondences: np.ndarray,
    world_points: np.ndarray,
    filter_thresh: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    filter a triangulated point set and its paired image correspondences for reporting only.
    """
    mask = point_cloud_mask(world_points, filter_thresh=filter_thresh)
    if len(mask) == 0:
        return correspondences, world_points
    return correspondences[mask], world_points[mask]


def filter_ba_for_reporting(
    world_points: np.ndarray,
    observed_points: List[np.ndarray],
    visibility: np.ndarray,
    filter_thresh: float = 3.0,
) -> tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    filter points for plotting while preserving observation alignment.
    """
    mask = point_cloud_mask(world_points, filter_thresh=filter_thresh)
    if len(mask) == 0 or np.all(mask):
        return world_points, observed_points, visibility

    filtered_points = world_points[mask]
    filtered_visibility = visibility[:, mask]
    filtered_observations: list[np.ndarray] = []

    for cam_idx, pts in enumerate(observed_points):
        visible_indices = np.where(visibility[cam_idx] == 1)[0]
        keep = mask[visible_indices]
        filtered_observations.append(np.asarray(pts, dtype=float)[keep])

    return filtered_points, filtered_observations, filtered_visibility


######## reprojection error ########
def _compute_reproj_err(
    world_points: np.ndarray,
    image_points: np.ndarray | List[np.ndarray],
    proj_matrices: np.ndarray | List[np.ndarray],
    visibility: np.ndarray | None = None,
) -> float:
    """
    compute reprojection error for one or more views
    """

    def _as_list_of_arrays(x):
        return (
            [np.asarray(v, dtype=float) for v in x]
            if isinstance(x, list)
            else [np.asarray(x, dtype=float)]
        )

    world_points = np.asarray(world_points, dtype=float)
    image_points_list = _as_list_of_arrays(image_points)
    proj_matrix_list = _as_list_of_arrays(proj_matrices)

    all_errors = []
    for i, (x_obs, P) in enumerate(zip(image_points_list, proj_matrix_list)):
        visible_world_points = (
            world_points
            if visibility is None
            else world_points[np.where(visibility[i] == 1)[0]]
        )
        if len(x_obs) == 0:
            continue
        view_errors = [
            np.linalg.norm(x - project_point(P, X))
            for X, x in zip(visible_world_points, x_obs)
        ]
        all_errors.extend(view_errors)

    return float(np.mean(all_errors))


def base_pair_reproj_errors(base_pair_outputs) -> tuple[float, float]:
    observations = [
        base_pair_outputs.inliers[:, 0, :],
        base_pair_outputs.inliers[:, 1, :],
    ]
    return (
        _compute_reproj_err(
            base_pair_outputs.world_linear,
            observations,
            base_pair_outputs.proj_matrices,
        ),
        _compute_reproj_err(
            base_pair_outputs.world_refined,
            observations,
            base_pair_outputs.proj_matrices,
        ),
    )


def pnp_reproj_errors(pnp_outputs) -> tuple[List[float], List[float]]:
    linear_errors: List[float] = []
    refined_errors: List[float] = []

    for pose_lin, pose_ref, matches_uv, points_xyz in zip(
        pnp_outputs.proj_matrices_linear,
        pnp_outputs.proj_matrices_refined,
        pnp_outputs.image_points,
        pnp_outputs.world_points,
    ):
        observations = (
            np.asarray(matches_uv, dtype=float)
            if len(matches_uv)
            else np.empty((0, 2), dtype=float)
        )
        linear_errors.append(_compute_reproj_err(points_xyz, observations, pose_lin))
        refined_errors.append(_compute_reproj_err(points_xyz, observations, pose_ref))

    return linear_errors, refined_errors


def ba_reproj_error(ba_outputs) -> float:
    return _compute_reproj_err(
        ba_outputs.points,
        ba_outputs.observed_points,
        ba_outputs.proj_matrices,
        visibility=ba_outputs.visibility,
    )


######## visualization functions ########
def plot_triangulation(
    *point_sets: np.ndarray,
    camera_centers: np.ndarray | None = None,
    normalize: bool = False,
    set_labels: list[str] | None = None,
    title: str = "triangulation",
) -> None:
    """
    visualize up to four 3D point sets (N, 3) in the x-z plane.

    :param point_sets: variable number of point sets to plot, each of shape (N, 3)
    :param camera_centers: optional (M, 3) array of camera centers to show as triangles
    :param normalize: whether to apply a single normalization across all points and centers for better visualization
    :param set_labels: optional list of labels for the point sets to show in the legend
    :param title: figure title
    """

    if not point_sets:
        raise ValueError("At least one point set must be provided")
    if len(point_sets) > 4:
        raise ValueError("plot_triangulation supports at most 4 point sets")

    colors = ["b", "r", "g", "m"]

    point_sets_arr = [np.asarray(P, dtype=float) for P in point_sets]

    # apply single normalization across all points and centers if true
    if normalize:
        cam_arr = (
            np.atleast_2d(np.asarray(camera_centers, dtype=float))
            if camera_centers is not None
            else None
        )

        all_pts = [P for P in point_sets_arr]
        if cam_arr is not None:
            all_pts.append(cam_arr)

        stacked = np.vstack(all_pts)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std == 0] = 1.0

        def _norm(X: np.ndarray) -> np.ndarray:
            return (X - mean) / std

        point_sets_plot = [_norm(P) for P in point_sets_arr]
        cam_plot = _norm(cam_arr) if cam_arr is not None else None
    else:
        point_sets_plot = point_sets_arr
        cam_plot = (
            np.atleast_2d(np.asarray(camera_centers, dtype=float))
            if camera_centers is not None
            else None
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, Pn in enumerate(point_sets_plot):
        ax.scatter(
            Pn[:, 0],
            Pn[:, 2],
            s=2,
            c=colors[idx],
            alpha=0.7,
            label=(
                set_labels[idx]
                if set_labels and idx < len(set_labels)
                else f"set {idx+1}"
            ),
        )

    # draw camera centers as triangles
    if cam_plot is not None:
        cam_colors = plt.cm.Set1(np.linspace(0, 1, cam_plot.shape[0]))
        for i in range(cam_plot.shape[0]):
            ax.scatter(
                cam_plot[i, 0],
                cam_plot[i, 2],
                s=80,
                color=cam_colors[i],
                marker="^",
                label=f"cam {i+1}",
            )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("auto", adjustable="box")
    ax.legend(loc="best")
    plt.show()


def _load_image(image_id: int) -> np.ndarray:
    """Load an image by id (1-based) from the Phase1/Data folder."""

    phase1_dir = Path(__file__).resolve().parents[1]
    data_dir = phase1_dir / "Data"
    img_path = data_dir / f"{image_id}.png"
    img = plt.imread(img_path)
    return img


def plot_correspondences(
    current_image_id: int,
    other_image_id: int,
    correspondences: np.ndarray,
    inliers: np.ndarray = None,
    title: str | None = None,
) -> None:
    """
    show two images side by side with correspondences connected by lines

    Parameters
    ----------
    current_image_id : int
        Id of the left image (1-based, used to load `<id>.png`).
    other_image_id : int
        Id of the right image.
    correspondences : (N, 2, 2) ndarray
        Each row is [[u1, v1], [u2, v2]].
    inliers : (N, 2, 2) ndarray
        Subset of correspondences that are inliers
    title : str, optional
        Figure title.
    """

    inlier_mask = (
        np.isin(correspondences, inliers).all(axis=(1, 2))
        if inliers is not None
        else None
    )

    corr = correspondences

    img_left = _load_image(current_image_id)
    img_right = _load_image(other_image_id)

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    h = max(h1, h2)

    # Create a canvas that places the two images side by side.
    canvas = np.ones((h, w1 + w2, 3), dtype=float)

    # Normalize / convert images to float if needed
    def _to_float(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        return img.astype(float)

    img_left_f = _to_float(img_left)
    img_right_f = _to_float(img_right)

    canvas[:h1, :w1, : img_left_f.shape[2]] = img_left_f
    canvas[:h2, w1 : w1 + w2, : img_right_f.shape[2]] = img_right_f

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(canvas)

    # Draw lines between correspondences: green for inliers, red for outliers.
    for idx, (p1, p2) in enumerate(corr):
        u1, v1 = p1
        u2, v2 = p2
        is_inlier = inlier_mask is not None and inlier_mask[idx]
        line_color = "green" if is_inlier else "red"
        # right image x-coordinate is offset by w1
        ax.plot([u1, u2 + w1], [v1, v2], color=line_color, linewidth=0.5)
        ax.scatter([u1, u2 + w1], [v1, v2], c=[line_color, line_color], s=5)
    # show legend for inliers and outliers
    if inlier_mask is not None:
        ax.scatter([], [], c="green", label="inliers", s=5)
        ax.scatter([], [], c="red", label="outliers", s=5)
        ax.legend(loc="best")

    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def _epipolar_line(F: np.ndarray, u: float, v: float) -> np.ndarray:
    """Compute epipolar line l = F x for point (u, v, 1)."""

    x = np.array([u, v, 1.0], dtype=float)
    return F @ x


def _draw_line(
    ax, a: float, b: float, c: float, x_offset: float, w: int, h: int, **kwargs
):
    """Draw line ax + by + c = 0 within image bounds [0,w) x [0,h)."""

    eps = 1e-8
    if abs(b) > eps:
        xs = np.array([0.0, float(w - 1)])
        ys = -(a * xs + c) / b
    else:
        # vertical line x = -c / a
        if abs(a) < eps:
            return
        x = -c / a
        xs = np.array([x, x])
        ys = np.array([0.0, float(h - 1)])

    ax.plot(xs + x_offset, ys, **kwargs)


def plot_epipolar_lines(
    current_image_id: int,
    other_image_id: int,
    F: np.ndarray,
    correspondences: np.ndarray,
    max_lines: int = 50,
    title: str | None = None,
) -> None:
    """Overlay epipolar lines induced by matched points on both images.

    For each correspondence (u1, v1) <-> (u2, v2):
    - draw the epipolar line of (u1, v1) in image 2 (using F x1)
    - draw the epipolar line of (u2, v2) in image 1 (using F^T x2)
    """

    F = np.asarray(F, dtype=float)
    corr = np.asarray(correspondences, dtype=float)

    if corr.ndim != 3 or corr.shape[1:] != (2, 2):
        raise ValueError("correspondences must have shape (N, 2, 2)")

    img_left = _load_image(current_image_id)
    img_right = _load_image(other_image_id)

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    h = max(h1, h2)

    canvas = np.ones((h, w1 + w2, 3), dtype=float)

    def _to_float(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        return img.astype(float)

    img_left_f = _to_float(img_left)
    img_right_f = _to_float(img_right)

    canvas[:h1, :w1, : img_left_f.shape[2]] = img_left_f
    canvas[:h2, w1 : w1 + w2, : img_right_f.shape[2]] = img_right_f

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(canvas)

    N = corr.shape[0]
    if N == 0:
        raise ValueError("No correspondences provided for epipolar visualization")

    # Subsample if too many lines
    step = max(1, N // max_lines) if N > max_lines else 1
    indices = np.arange(0, N, step)

    for i in indices:
        (u1, v1), (u2, v2) = corr[i]

        # Line in image 2 from point in image 1
        a2, b2, c2 = _epipolar_line(F, u1, v1)
        _draw_line(
            ax,
            a2,
            b2,
            c2,
            x_offset=w1,
            w=w2,
            h=h,
            color="orange",
            linewidth=0.75,
            alpha=0.8,
        )

        # Line in image 1 from point in image 2
        a1, b1, c1 = _epipolar_line(F.T, u2, v2)
        _draw_line(
            ax,
            a1,
            b1,
            c1,
            x_offset=0.0,
            w=w1,
            h=h,
            color="cyan",
            linewidth=0.75,
            alpha=0.8,
        )

    # Also plot the original points for reference
    pts1 = corr[:, 0, :]
    pts2 = corr[:, 1, :]
    ax.scatter(pts1[:, 0], pts1[:, 1], c="red", s=3, label="pts img1")
    ax.scatter(pts2[:, 0] + w1, pts2[:, 1], c="lime", s=3, label="pts img2")

    if title is None:
        title = "Epipolar lines in both images"
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_reprojection(
    image_id: int,
    observed: np.ndarray,
    P: np.ndarray,
    world_points: np.ndarray,
    title: str | None = None,
) -> None:
    """
    visualize reprojection of 3D points onto image

    Parameters
    ----------
    image_id : int
        Id of the image (1-based, used to load `<id>.png`).
    observed : (N, 2) ndarray
        Observed 2D points in the image.
    P: (3, 4) ndarray
        Camera projection matrix P = K [R | t] for the view.
    world_points : (N, 3) ndarray
        3D points associated with the observed points.
    title : str, optional
        Figure title.
    """

    def _to_float(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        return img.astype(float)

    observed = np.asarray(observed)
    world_points = np.asarray(world_points)

    if observed.shape[0] != world_points.shape[0]:
        raise ValueError("observed and world_points must have the same length")

    img = _load_image(image_id)
    img_f = _to_float(img)

    # Project 3D points into camera
    X_h = np.hstack([world_points, np.ones((world_points.shape[0], 1), dtype=float)])

    proj1 = (P @ X_h.T).T

    # Avoid division by zero / points behind camera.
    eps = 1e-12
    mask1 = proj1[:, 2] > eps

    u1_hat = proj1[:, 0] / np.where(mask1, proj1[:, 2], 1.0)
    v1_hat = proj1[:, 1] / np.where(mask1, proj1[:, 2], 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img_f)

    # Observed 2D points (red) and reprojected points (green) for each image.
    ax.scatter(
        observed[:, 0], observed[:, 1], c="red", s=5, label=f"obs img {image_id}"
    )
    ax.scatter(
        u1_hat[mask1],
        v1_hat[mask1],
        c="lime",
        s=5,
        label=f"reproj img {image_id}",
    )

    # draw small line segments from observed to reprojected locations.
    for i in range(world_points.shape[0]):
        if mask1[i]:
            ax.plot(
                [observed[i, 0], u1_hat[i]],
                [observed[i, 1], v1_hat[i]],
                color="yellow",
                linewidth=0.5,
            )

    if title is None:
        title = f"Reprojection in image {image_id}"
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
