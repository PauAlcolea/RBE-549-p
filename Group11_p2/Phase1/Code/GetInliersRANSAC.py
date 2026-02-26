import numpy as np
from EstimateFundamentalMatrix import estimateFundamentalMat


def getInliersRANSAC(
    correspondences: np.ndarray, n_iterations: int = 1000, inlier_thresh: float = 50.0
) -> np.ndarray:
    """
    get inliers from RANSAC
    error metric is Sampson distance, as used in MATLAB estimateFundamentalMatrix()

    :param correspondences: (N, 2, 2) array, each [[u1, v1], [u2, v2]]
    :param n_iterations: number of RANSAC iterations
    :param inlier_thresh: threshold for inlier detection, in pixels^2
    :return: (M, 2, 2) array of inliers
    """
    best_inlier_indices = None
    best_count = 0

    num_corr = len(correspondences)

    for _ in range(n_iterations):
        # randomly select 8 correspondences
        indices = np.random.choice(num_corr, size=8, replace=False)
        subset = correspondences[indices]

        # estimate F for this subset
        F_subset = estimateFundamentalMat(subset)

        # form homogenous coordinates for all correspondences
        pts1 = np.hstack(
            (
                correspondences[:, 0, :],
                np.ones((num_corr, 1), dtype=correspondences.dtype),
            )
        )
        pts2 = np.hstack(
            (
                correspondences[:, 1, :],
                np.ones((num_corr, 1), dtype=correspondences.dtype),
            )
        )

        # algebraic error x2^T @ F @ x1 has arbitrary scale so
        # we use Sampson distance to get more meaningful error in pixels

        # numerator: (x2^T F x1)^2
        num = np.einsum("bi,bi->b", pts2 @ F_subset, pts1) ** 2
        # Fx: epipolar lines of pts1 in image 2
        Fx1 = (F_subset @ pts1.T).T
        # F^T x': epipolar lines of pts2 in image 1
        Ftx2 = (F_subset.T @ pts2.T).T
        # denominator: squared gradients
        den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        # avoid divide by zero
        den[den == 0] = np.finfo(float).eps

        errors = num / den  # pixels^2
        inlier_mask = errors < inlier_thresh
        inlier_indices = np.where(inlier_mask)[0]
        inlier_count = inlier_mask.sum()

        if inlier_count > best_count:
            best_count = inlier_count
            best_inlier_indices = inlier_indices

    if best_inlier_indices is None or best_count == 0:
        return np.empty((0, 2, 2), dtype=float)

    best_inliers = correspondences[best_inlier_indices].astype(float)
    return best_inliers
