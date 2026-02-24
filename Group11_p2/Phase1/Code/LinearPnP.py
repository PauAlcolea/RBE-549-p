import numpy as np


def linearPnP(xs: np.ndarray, Xs: np.ndarray, K: np.ndarray):
    """
    this function is used to calculate the pose of the camera for the added viewpoint
    https://www.youtube.com/watch?v=tv56SgT_e_8&t=1s


    :param x: a list of 6 2D points, they are the image projected points of X onto the image plane
        each point is an np.ndarray (x, y)
    :param X: a list of 6 3D points that correspond to x
    :param K: camera intrinsic matrix (3,3)
    """

    N = xs.shape[0]
    A = np.zeros((2 * N, 12))

    # build A so that A*p = 0
    # p is the vector of P the projection matrix, it is of size (12,)
    for i in range(N):
        x, y = xs[i]
        X, Y, Z = Xs[i]

        A[2 * i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A[2 * i + 1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])

    U, D, Vt = np.linalg.svd(A)

    # vector of the projection matrix, shape (12,)
    p = Vt[-1, :]

    P = np.zeros(shape=(3, 4))
    P = p.reshape(3, 4)  # Get the shape of the projection matrix back to a 3x4

    P3 = P[:, 0:3]
    K_inv = np.linalg.inv(K)
    U2, D2, Vt2 = np.linalg.svd((K_inv @ P3))

    scale_factor = D2[0]
    R = U2 @ Vt2

    # check for the left hand handed system
    if np.linalg.det(R) < 0:
        R = -R
        T = -(K_inv @ P[:, 3]) / scale_factor
    else:
        T = (K_inv @ P[:, 3]) / scale_factor

    return R, T
