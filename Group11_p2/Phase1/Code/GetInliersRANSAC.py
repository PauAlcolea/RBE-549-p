import numpy as np
from EstimateFundamentalMatrix import estimateFundamentalMat


def getInliersRANSAC(
    correspondences: np.ndarray, n_iterations: int, inlier_thresh: float
) -> np.ndarray:
    best_inliers = []
    best_count = 0

    for _ in range(n_iterations):
        # randomly select 8 correspondences
        indices = np.random.choice(len(correspondences), size=8, replace=False)
        subset = correspondences[indices]

        # estimate F for this subset
        F_subset = estimateFundamentalMat(subset)

        # find inliers for this F_subset
        inliers = []
        for (x1, y1), (x2, y2) in correspondences:
            point1 = np.array([x1, y1, 1])
            point2 = np.array([x2, y2, 1])
            # compute epipolar error using F
            error = abs(point2.T @ F_subset @ point1)
            if error < inlier_thresh:
                inliers.append((x1, y1, x2, y2))

        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers

    # reshape best_inliers to (N, 2, 2) format
    if best_inliers:
        best_inliers = np.array(
            [[[x1, y1], [x2, y2]] for (x1, y1, x2, y2) in best_inliers], dtype=float
        )

    return np.array(best_inliers)
