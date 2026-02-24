import numpy as np

def linearPnP(xs: np.ndarray, Xs: np.ndarray, K: np.ndarray):
    """
    this function is used to calculate the pose of the camera for the added viewpoint
    https://www.youtube.com/watch?v=tv56SgT_e_8&t=1s

    
    :param x is a list of 6 2D points, they are the image projected points of X onto the image plane
        each point is an np.ndarray (x, y)
    :param X is a list of 6 3D points that correspond to x
    :K camera intrinsic matrix (3,3)
    """

    N = xs.shape[0]
    A = np.zeros((2*N, 12))

    # build A so that A*p = 0
    # p is the vector of P the projection matrix, it is of size (12,)
    for i in range(N):
        x, y = xs[i]
        X, Y, Z = Xs[i]

        A[2*i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A[2*i + 1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])

    U, D, Vt = np.linalg.svd(A)

    # vector of the projection matrix, shape (12,)
    p = Vt[-1, :]

    P = np.zeros(shape=(3,4))
    P = p.reshape(3, 4)             # Get the shape of the projection matrix back to a 3x4

    P3 = P[:, 0:3]
    K_inv = np.linalg.inv(K)
    U2, D2, Vt2 = np.linalg.svd((K_inv @ P3))
    
    scale_factor = D2[0]
    R = U2 @ Vt2

    # check for the left hand handed system
    if np.linalg.det(R) < 0:
        R = - R
        T = -(K_inv @ P[:, 3]) / scale_factor
    else:
        T = (K_inv @ P[:, 3]) / scale_factor

    return R, T



# Testing function with LLM
def test_linearPnP_synthetic():
    # Define a known ground truth pose
    # Simple rotation (e.g., 10 degrees around Y-axis)
    theta = np.radians(10)
    R_gt = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [0,              1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    C_gt = np.array([1.0, 0.5, -3.0])  # camera center in world

    # Camera intrinsics (known)
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=float)

    # Generate random 3D world points
    np.random.seed(42)
    num_points = 6  # minimum required per assignment
    Xs = Xs = np.random.uniform(-1, 1, (num_points, 3))

    # Project using P = K R [I | -C]
    t_gt = -R_gt @ C_gt
    P_gt = K @ np.hstack([R_gt, t_gt.reshape(3,1)])

    xs = np.zeros((num_points, 2))
    for i in range(num_points):
        X_h = np.append(Xs[i], 1)
        x_h = P_gt @ X_h
        x_h /= x_h[2]
        xs[i] = x_h[:2]

    # Run your function
    R_est, T_est = linearPnP(xs, Xs, K)

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

# test_linearPnP_synthetic()