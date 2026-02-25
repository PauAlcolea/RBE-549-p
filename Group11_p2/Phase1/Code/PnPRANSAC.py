import numpy as np
from LinearPnP import linearPnP


def pnpRANSAC(xs, Xs, K, n_iterations=1000, inlier_thresh=30):
    """
    this function performs RANSAC to add more views to the map, it does so to remove outliers

    :param xs: all of the 2D points for the new view that are common with the existing map
    :param Xs: all of the 3D points that the 2D points are the projection of
    :param K: the camera internal matrix
    :param n_iterations: number of RANSAC iterations
    :param inlier_thresh: threshold for inlier detection, in pixels^2
    """
    n = 0  # maximum inlier count so far, used to select the best set
    N = 6  # How many inliers are going to be passed to the Linear PnP

    for _ in range(n_iterations):
        # choose 6 points and get the 2d and the 3d
        indx = np.random.choice(len(xs), N, replace=False)
        x_N = xs[indx]
        X_N = Xs[indx]

        R, T = linearPnP(x_N, X_N, K)

        inliers_xs = []
        inliers_Xs = []

        # estimated projection matrix
        P = K @ np.hstack([R, T.reshape(3, 1)])
        P_1 = P[0, :]
        P_2 = P[1, :]
        P_3 = P[2, :]

        for i, X in enumerate(Xs):
            u, v = xs[i]
            X_tilde = np.append(X, 1)

            # measure the reprojection error
            e = (u - (P_1 @ X_tilde) / (P_3 @ X_tilde)) ** 2 + (
                v - (P_2 @ X_tilde) / (P_3 @ X_tilde)
            ) ** 2

            # if error is below threshold, it counts as inlier
            if e < inlier_thresh:
                inliers_xs.append(xs[i])
                inliers_Xs.append(Xs[i])

        if len(inliers_Xs) > n:
            n = len(inliers_Xs)
            most_inliers_xs = inliers_xs  # keep track of the set of points that has the most inliers, these will be fed back to pnp for final R, T
            most_inliers_Xs = inliers_Xs

    # now pass that set of most inliers back to the linear pnp to get the real pose
    final_R, final_T = linearPnP(
        xs=np.array(most_inliers_xs), Xs=np.array(most_inliers_Xs), K=K
    )

    return final_R, final_T
