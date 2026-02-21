import numpy as np


def _buildProjectionMatrices(K: np.ndarray, Pose: np.ndarray) -> np.ndarray:
    """
    build projection matrix P = K [R | t] from camera intrinsic matrix K and pose [R | t]

    :param K: camera internal matrix, shape = (3,3)
    :param Pose: one pose, which is [R|t], shape = (3,4)
    """
    # Pose is [R | t]; projection matrix is P = K [R | t]
    # the projection equation is taken from the slides: https://youtu.be/Ap6qft93cCM?t=2195
    Projection = K @ Pose
    return Projection


def linearTriangulation(
    K, pose1: np.ndarray, pose2: np.ndarray, x1: np.ndarray, x2: np.ndarray
):
    """
    this function triangulates 3D points

    :param pose1: Camera pose, this will always be zero because that is the reference
    :param pose2: another camera pose
    :param K: camera intrinsic matrix
    :param correspondences between the two cameras, x1 and x2 np.ndarrays
    : expecting x1 and x2 in the form of [x1 y1] (np.array of shape (2,))
    :return: 3D point as a NumPy array of shape (3,), in homogeneous coordinates normalized to X[3] = 1
    """
    # get the projections from the poses
    P1 = _buildProjectionMatrices(K, pose1)
    P2 = _buildProjectionMatrices(K, pose2)

    x1_3d = np.hstack([x1, np.array([1])])
    x2_3d = np.hstack([x2, np.array([1])])

    # Having trouble with the cross product between [x1, y1, 1] and the projection matrix, which has three rows and four columns
    # instead, divide P into the three rows of four elements
    P1r1 = P1[0, :]
    P1r2 = P1[1, :]
    P1r3 = P1[2, :]

    P2r1 = P2[0, :]
    P2r2 = P2[1, :]
    P2r3 = P2[2, :]

    # equation for the cross product with determinants
    # https://www.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/x786f2022:vectors-and-matrices/a/cross-products-mvc
    cross_x1_p1 = np.stack(
        [
            (x1_3d[1] * P1r3 - x1_3d[2] * P1r2),
            (x1_3d[2] * P1r1 - x1_3d[0] * P1r3),
            (x1_3d[0] * P1r2 - x1_3d[1] * P1r1),
        ]
    )

    cross_x2_p2 = np.stack(
        [
            (x2_3d[1] * P2r3 - x2_3d[2] * P2r2),
            (x2_3d[2] * P2r1 - x2_3d[0] * P2r3),
            (x2_3d[0] * P2r2 - x2_3d[1] * P2r1),
        ]
    )

    A = np.vstack([cross_x1_p1, cross_x2_p2])
    U, D, Vt = np.linalg.svd(A)
    X = Vt[-1, :]

    # TODO: normalize this correctly
    X = X / X[3]
    return X[0:3]
