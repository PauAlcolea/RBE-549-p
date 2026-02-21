import numpy as np
from scipy.optimize import least_squares


def _project_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    project a world point X to pixel coords using a 3x4 camera matrix P = K [R | t].

    :param P: 3x4 camera matrix
    :param X: 3D world point (3,)
    :return: 2D pixel coordinates (u, v)
    """
    X_h = np.hstack([X, 1.0])  # homogeneous world point
    x, y, z = P @ X_h

    # FIXME: avoid divide by zero / points behind camera
    if z <= 1e-12:
        return np.array([1e9, 1e9], dtype=float)

    u = x / z
    v = y / z
    return np.array([u, v], dtype=float)


def _reprojection_residual(
    X: np.ndarray,
    correspondence: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    compute reprojection residuals of world point X in two views

    :param X: 3D world point (3,)
    :param correspondence: [[u1, v1], [u2, v2]]
    :param P1: 3x4 camera matrix for view 1
    :param P2: 3x4 camera matrix for view 2
    :return: residuals [u1 - u1_hat, v1 - v1_hat, u2 - u2_hat, v2 - v2_hat]
    """
    x1_uv, x2_uv = correspondence
    uv1_hat = _project_point(P1, X)
    uv2_hat = _project_point(P2, X)
    return np.hstack([x1_uv - uv1_hat, x2_uv - uv2_hat])


def _nonlinear_triangulate_point(
    correspondence: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    world_point_est: np.ndarray,
):
    """
    refine a single world point with nonlinear least squares

    :param correspondence: [[u1, v1], [u2, v2]]
    :param P1: 3x4 camera matrix for view 1
    :param P2: 3x4 camera matrix for view 2
    :param world_point_est: initial 3D estimate from linear triangulation
    :return: refined 3D point and optimization result
    """

    fun = lambda X: _reprojection_residual(X, correspondence, P1, P2)

    res = least_squares(fun, x0=np.asarray(world_point_est, dtype=float), method="lm")
    return res.x, res


def nonlinearTriangulation(
    correspondences: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    world_points_est: np.ndarray,
) -> np.ndarray:
    """
    nonlinear refinement of linearly triangulated world points for a stereo pair

    :param correspondences: (N, 2, 2) array, each [[u1, v1], [u2, v2]]
    :param P1: 3x4 camera matrix for view 1
    :param P2: 3x4 camera matrix for view 2
    :param world_points_est: (N, 3) initial 3D points from linear triangulation
    :return: (N, 3) refined 3D points
    """

    X_out = np.zeros_like(world_points_est, dtype=float)
    for i in range(world_points_est.shape[0]):
        Xi, res = _nonlinear_triangulate_point(
            correspondences[i], P1, P2, world_points_est[i]
        )
        if not res.success:
            print(
                f"Warning: nonlinear triangulation optimization failed for point {i} with message: {res.message}"
            )
        X_out[i] = Xi
    return X_out
