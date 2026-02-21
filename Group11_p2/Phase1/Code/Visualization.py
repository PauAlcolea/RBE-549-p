import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_triangulation(points: np.ndarray, title: str = "triangulation"):
    """
    visualize triangulated 3D points in x-z plane (top-down view)
    """

    X = points[:, 0]
    Z = points[:, 2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X, Z, s=2, c="b", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("auto", adjustable="box") # FIXME auto vs equal
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
    inliers_only: bool = False,
    inlier_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Show two images side by side with correspondences connected by lines.

    Parameters
    ----------
    current_image_id : int
        Id of the left image (1-based, used to load `<id>.png`).
    other_image_id : int
        Id of the right image.
    correspondences : (N, 2, 2) ndarray
        Each row is [[u1, v1], [u2, v2]].
    inliers_only : bool, optional
        If True, only draw matches where inlier_mask is True.
    inlier_mask : (N,) ndarray of bool, optional
        Mask indicating which correspondences are inliers.
    title : str, optional
        Figure title.
    """

    if inliers_only:
        if inlier_mask is None:
            raise ValueError("inliers_only=True but inlier_mask is None")
        if inlier_mask.shape[0] != correspondences.shape[0]:
            raise ValueError("inlier_mask and correspondences must have same length")
        corr = correspondences[inlier_mask]
    else:
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

    # Draw lines between correspondences.
    for p1, p2 in corr:
        u1, v1 = p1
        u2, v2 = p2
        # right image x-coordinate is offset by w1
        ax.plot([u1, u2 + w1], [v1, v2], color="green", linewidth=0.5)
        ax.scatter([u1, u2 + w1], [v1, v2], c=["cyan", "red"], s=5)

    ax.set_axis_off()
    if title is None:
        title = "Correspondences" if not inliers_only else "RANSAC inliers"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
