import numpy as np


def _normalize_points(pts_2d: np.ndarray):
    """
    hartley normalization for better stability
    """
    pts_2d = pts_2d.astype(float)
    # center points at origin
    mean = pts_2d.mean(axis=0)
    pts_centered = pts_2d - mean

    # scale points so avg dist from origin is sqrt(2)
    dists = np.sqrt((pts_centered**2).sum(axis=1))
    mean_dist = dists.mean()
    if mean_dist < 1e-8:
        s = 1.0
    else:
        s = np.sqrt(2.0) / mean_dist

    # normalization transform T
    T = np.array(
        [
            [s, 0.0, -s * mean[0]],
            [0.0, s, -s * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    pts_h = np.hstack([pts_2d, np.ones((pts_2d.shape[0], 1), dtype=float)])
    pts_norm_h = (T @ pts_h.T).T
    return pts_norm_h[:, :2], T


def estimateFundamentalMat(eight_correspondences: np.ndarray) -> np.ndarray:
    """
    This estimates the Fundamental matrix between two images given eight point correspondences

    :param eight_correspondences: This is a numpy array with 8 correspondences between two images, shape (8, 2, 2) -> ([ [x1, y1], [x1', y2'] ], [], ...)
    :return: estimated 3x3 Fundamental Matrix, enforced Rank 2
    """
    pts1 = eight_correspondences[:, 0, :].astype(float)
    pts2 = eight_correspondences[:, 1, :].astype(float)
    # normalize points in each image
    pts1_n, T1 = _normalize_points(pts1)
    pts2_n, T2 = _normalize_points(pts2)
    eight_correspondences_n = np.stack([pts1_n, pts2_n], axis=1)

    # build A matrix that will satisfy the equation: Af = 0
    # In this case f would be the 9D vector with all the elements of the fundamental matrix F
    # A begins as a list and then becomes an array will be np.array of size 8x9
    # x2 and y2 are the projections of x1 and y1 on the second image (x' and y')
    A_list = []
    for (x1, y1), (x2, y2) in eight_correspondences_n:

        # append one row per correspondence
        A_list.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    A = np.array(A_list)

    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]  # last row of Vt is null space of A, i.e. solution f
    F = f.reshape(
        (3, 3)
    ).T  # get the transpose because the notes have f in a different order (f11, f21, f31 ...) instead of (f11, f12, f13 ...)

    # enforce rank-2 constraint on F via SVD
    U, S, Vt_F = np.linalg.svd(F)
    S[-1] = 0  # set smallest singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt_F

    # denormalize
    F_final = T2.T @ F_rank2 @ T1

    return F_final
