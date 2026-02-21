import numpy as np
from EstimateFundamentalMatrix import estimateFundamentalMat


def getInliersRANSAC(
    correspondences: np.ndarray, n_iterations: int, inlier_thresh: float
) -> np.ndarray:
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

        # use einsum to compute point2.T @ F_subset @ point1
        errors = np.abs(np.einsum("bi,ij,bj->b", pts2, F_subset, pts1))
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
