import numpy as np
from LinearPnP import linearPnP

def pnpRANSAC(xs, Xs, K):
    """
    this function performs RANSAC to add more views to the map, it does so so to remove outliers

    :param xs are all of the 2D points for the new view that are common with the existing map
    :param Xs are all of the 3D points that the 2D points are the projection of
    :param K the camera internal matrix
    """

    n = 0           # maximum inlier count so far, used to select the best set
    M = 1000        # How many iterations will this run for?
    N = 6           # How many inliers are going to be passed to the Linear PnP
    e_threshold = 4

    for _ in range(M):
        # choose 6 points and get the 2d and the 3d 
        indx = np.random.choice(len(xs), N, replace=False)
        x_N = xs[indx]
        X_N = Xs[indx]

        R, T = linearPnP(x_N, X_N, K)

        inliers_xs = []
        inliers_Xs = []

        # estimated projection matrix
        P = K @ np.hstack([R, T.reshape(3,1)])
        P_1 = P[0, :]
        P_2 = P[1, :]
        P_3 = P[2, :]

        for i, X in enumerate(Xs):
            u, v = xs[i]
            X_tilde = np.append(X, 1)

            # measure the reprojection error
            e = (u - (P_1 @ X_tilde)/(P_3 @ X_tilde))**2 + (v - (P_2 @ X_tilde)/(P_3 @ X_tilde))**2

            # if error is below threshold, it counts as inlier
            if e < e_threshold:
                inliers_xs.append(xs[i])
                inliers_Xs.append(Xs[i])

        if len(inliers_Xs) > n:
            n = len(inliers_Xs)
            most_inliers_xs = inliers_xs      # keep track of the set of points that has the most inliers, these will be fed back to pnp for final R, T
            most_inliers_Xs = inliers_Xs

    
    final_R, final_T = linearPnP(xs=np.array(most_inliers_xs), Xs=np.array(most_inliers_Xs), K=K)
    # now pass that set of most inliers back to the linear pnp to get the real pose

    return final_R, final_T


def test_pnp_synthetic(num_points=100, outlier_ratio=0.2):
    np.random.seed(1)

    # --- Ground truth rotation ---
    theta = np.radians(15)
    R_gt = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # --- Ground truth translation ---
    C_gt = np.array([0.5, -0.2, -4.0])
    t_gt = -R_gt @ C_gt

    # --- Camera intrinsics ---
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    # --- Generate 3D points ---
    Xs = np.random.uniform(-2, 2, (num_points, 3))

    # --- Projection matrix ---
    P_gt = K @ np.hstack([R_gt, t_gt.reshape(3,1)])

    xs = np.zeros((num_points, 2))

    for i in range(num_points):
        X_h = np.append(Xs[i], 1)
        x_h = P_gt @ X_h
        x_h /= x_h[2]
        xs[i] = x_h[:2]
   
    R_est, T_est = pnpRANSAC(xs, Xs, K)

    
    
    # Check rotation orthogonality
    print("det(R):", np.linalg.det(R_est))  # Should be ~1.0
    print("R R^T:\n", R_est @ R_est.T)       # Should be ~identity

    # Check reprojection error
    t_est = T_est
    P_est = K @ np.hstack([R_est, t_est.reshape(3,1)])
    errors = []
    for x, X in zip(xs, Xs):
        X_h = np.append(X, 1)
        x_proj = P_est @ X_h
        x_proj /= x_proj[2]
        err = np.linalg.norm(x_proj[:2] - x)
        errors.append(err)
    print("Mean reprojection error:", np.mean(errors))  # Should be ~0.0

    
    return

test_pnp_synthetic()
