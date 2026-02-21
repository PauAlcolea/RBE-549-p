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
