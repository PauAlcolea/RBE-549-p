import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _normalize(P: np.ndarray) -> np.ndarray:
    return (P - P.mean(axis=0)) / P.std(axis=0)


def plot_triangulation(
    *point_sets: np.ndarray,
    title: str = "triangulation",
) -> None:
    """
    normalize and visualize up to four 3D point set (N, 3) in the x-z plane.
    """

    if not point_sets:
        raise ValueError("At least one point set must be provided")
    if len(point_sets) > 4:
        raise ValueError("plot_triangulation supports at most 4 point sets")

    colors = ["b", "r", "g", "m"]

    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, P in enumerate(point_sets):
        P_norm = _normalize(np.asarray(P))
        ax.scatter(
            P_norm[:, 0],
            P_norm[:, 2],
            s=2,
            c=colors[idx],
            alpha=0.7,
            label=f"set {idx+1}",
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


def plot_reprojection(
    current_image_id: int,
    other_image_id: int,
    correspondences: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    world_points: np.ndarray,
    title: str | None = None,
) -> None:
    """
    visualize reprojection of 3D points into both images

    Parameters
    ----------
    current_image_id : int
        Id of the left image (1-based, used to load `<id>.png`).
    other_image_id : int
        Id of the right image.
    correspondences : (N, 2, 2) ndarray
        Each row is [[u1, v1], [u2, v2]] for the same 3D point.
    P1, P2 : (3, 4) ndarray
        Camera projection matrices P = K [R | t] for the two views.
    world_points : (N, 3) ndarray
        3D points associated with the correspondences.
    title : str, optional
        Figure title.
    """

    correspondences = np.asarray(correspondences)
    world_points = np.asarray(world_points)

    if correspondences.shape[0] != world_points.shape[0]:
        raise ValueError("correspondences and world_points must have the same length")

    img_left = _load_image(current_image_id)
    img_right = _load_image(other_image_id)

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    h = max(h1, h2)

    # Create a canvas that places the two images side by side.
    canvas = np.ones((h, w1 + w2, 3), dtype=float)

    def _to_float(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        return img.astype(float)

    img_left_f = _to_float(img_left)
    img_right_f = _to_float(img_right)

    canvas[:h1, :w1, : img_left_f.shape[2]] = img_left_f
    canvas[:h2, w1 : w1 + w2, : img_right_f.shape[2]] = img_right_f

    # Project 3D points into both cameras.
    X_h = np.hstack([world_points, np.ones((world_points.shape[0], 1), dtype=float)])

    proj1 = (P1 @ X_h.T).T  # (N, 3)
    proj2 = (P2 @ X_h.T).T  # (N, 3)

    # Avoid division by zero / points behind camera.
    eps = 1e-12
    mask1 = proj1[:, 2] > eps
    mask2 = proj2[:, 2] > eps

    u1_hat = proj1[:, 0] / np.where(mask1, proj1[:, 2], 1.0)
    v1_hat = proj1[:, 1] / np.where(mask1, proj1[:, 2], 1.0)
    u2_hat = proj2[:, 0] / np.where(mask2, proj2[:, 2], 1.0)
    v2_hat = proj2[:, 1] / np.where(mask2, proj2[:, 2], 1.0)

    obs1 = correspondences[:, 0, :]
    obs2 = correspondences[:, 1, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(canvas)

    # Observed 2D points (red) and reprojected points (green) for each image.
    ax.scatter(obs1[:, 0], obs1[:, 1], c="red", s=5, label="obs img1")
    ax.scatter(
        u1_hat[mask1],
        v1_hat[mask1],
        c="lime",
        marker="+",
        s=25,
        label="reproj img1",
    )

    ax.scatter(obs2[:, 0] + w1, obs2[:, 1], c="red", s=5, label="obs img2")
    ax.scatter(
        u2_hat[mask2] + w1,
        v2_hat[mask2],
        c="cyan",
        marker="+",
        s=25,
        label="reproj img2",
    )

    # Optionally draw small line segments from observed to reprojected locations.
    for i in range(world_points.shape[0]):
        if mask1[i]:
            ax.plot(
                [obs1[i, 0], u1_hat[i]],
                [obs1[i, 1], v1_hat[i]],
                color="yellow",
                linewidth=0.5,
            )
        if mask2[i]:
            ax.plot(
                [obs2[i, 0] + w1, u2_hat[i] + w1],
                [obs2[i, 1], v2_hat[i]],
                color="yellow",
                linewidth=0.5,
            )

    if title is None:
        title = "Reprojection in both images"
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
