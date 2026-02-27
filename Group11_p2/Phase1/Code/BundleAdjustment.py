import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2


def project_point(X, R, t, K):
    """
    helper function to take a 3d point and project it onto an image plane
    """
    X_h = np.hstack([X, 1.0])
    P = K @ np.hstack([R, t.reshape(3, 1)])
    x_proj_h = P @ X_h
    x_proj = x_proj_h[0:2] / x_proj_h[2]
    return x_proj


def pack(Rs, ts, Xs) -> np.ndarray:
    """
    This helper function is used for the optimization. The parameters need to be packed so that they can be inputted into scipy.optimize
    it takes the rodrigues vectors of the Rotation matrix, which is simpler and packs easier
    the packing goes [R1, t1, R2, t2, ... Rm, tm, X1, X2, X3 ... Xn]
    """
    params = []
    for R, t in zip(Rs, ts):
        rvec, _ = cv2.Rodrigues(R)
        params.append(
            np.concatenate(
                (
                    rvec.reshape(
                        3,
                    ),
                    t,
                )
            )
        )

    for X in Xs:
        params.append(X)

    return np.concatenate(params)


def unpack(params, num_cams, num_points):
    """
    Unpack the 1D parameter vector into camera rotations, translations, and 3D points.

    :param params: 1D array of parameters (output of pack)
    :param num_cams: number of camera poses
    :param num_points: number of 3D points
    :return: Rs list of rotation matrices, ts list of translations, Xs list of all world points
    """
    Rs = []
    ts = []
    Xs = []

    idx = 0
    for _ in range(num_cams):
        rvec = params[idx : idx + 3]
        R, _ = cv2.Rodrigues(rvec)
        idx += 3
        t = params[idx : idx + 3]
        idx += 3
        Rs.append(R)
        ts.append(t)

    for _ in range(num_points):
        X = params[idx : idx + 3]
        idx += 3
        Xs.append(X)

    return Rs, ts, Xs


def residual(params, points_2d, n_cameras, n_points, V, K, point2d_idx_map):
    """
    this function will return the residual and will be used for the actual least squares optimization
    points_world are all of the 3d points on the map, The optimizer is going to need to know what the 2d point projection of each X
    is in the image plane in which it apears for each image

    the visibility matrix tells if camera i observes 3d point Xj
    :param params are the packed parameters including all of the Rs, Ts and Xs
    :param points_2d are all of the 2d points, a list or array where each element is a list or array for all the points of one camera
    :param n_cameras number of cameras
    :param n_points number of points
    :param K intrinsic camera matrix
    :param V is visibility matrix
    :param point2d_idx_map is to connect every 2d point with the 3d point that it is a projection of
    """
    Rs, ts, Xs = unpack(params, n_cameras, n_points)

    res = []

    # go through every camera and every point
    for i, (R, t) in enumerate(zip(Rs, ts)):
        for j, X in enumerate(Xs):
            # check if that camera actually sees that point
            if V[i, j] == 1:
                x_proj = project_point(X, R, t, K)

                # see what row in points_2d corresponds with X_j
                ind_in_points2d = point2d_idx_map[i][j]
                x_obs = points_2d[i][ind_in_points2d]
                residual = x_obs - x_proj
                res.append(residual)

    return np.concatenate(res)


def build_sparcity(num_cams, num_points, V):
    """
    This function is used to make a spacrity that cna be passed to the optimizer so that it runs much faster
    it knonws which derivatives are zero so that the optimizer doesn't have to calulate them
    """
    n_params = num_cams * 6 + num_points * 3
    # each pair of camera and points will give you 2 residuals, error for u and error for v
    n_residuals = int(np.sum(V)) * 2

    sparsity = lil_matrix((n_residuals, n_params), dtype=int)
    res_idx = 0
    for i in range(num_cams):
        for j in range(num_points):
            if V[i, j] == 1:
                # each camera has 6 parameters, 3 for rotation and 3 for translation
                # 2 residuals depend on these 6 params
                sparsity[res_idx : res_idx + 2, i * 6 : i * 6 + 6] = 1
                # each point has 3 parameters, X, Y and Z
                # 2 residuals depend on these 3 params
                sparsity[
                    res_idx : res_idx + 2,
                    num_cams * 6 + j * 3 : num_cams * 6 + j * 3 + 3,
                ] = 1
                res_idx += 2
    return sparsity


def bundleAdjustment(
    K: np.ndarray,
    V: np.ndarray,
    all_poses: np.ndarray,
    points_world: np.ndarray,
    points_2d,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine the camera poses and 3D world points by optimization and minimization of the reprojection error

    :param K is the camera intrinsitc matrix, used to get R and t back
    :param V is the visibility matrix
    :param all poses are all of the poses of all of the cameras, these will need to be changed to R and t
    :param final_3D_world is all of the 3D world points after pnp
    :param points_2d these should be the observed points for each camera, in the form of a list of array with an initial variable of number of cams
        (num_points, 2) -> [u,v]
    :return the adjusted final poses and the adjustted final 3D world points in the forms of np.arrays
    """

    # the poses are gotten from K @ np.hstack([R, t.reshape(3, 1)]), so to get R and t I must reverse
    K_inv = np.linalg.inv(K)
    Rts = [K_inv @ P for P in all_poses]
    Rs = [Rt[:, 0:3] for Rt in Rts]
    ts = [Rt[:, 3] for Rt in Rts]

    n_cameras = V.shape[0]
    n_points = V.shape[1]

    # map all of the 2d points to their 3d counterpart to be able to do the residual calculation properly
    # making a dictionary to 3d points corresponding to which wors in points_2d[i]
    # the counter is what keeps track of the row in which the point in question is
    cam_point_ind_map = []
    for i in range(n_cameras):
        idx_map = {}
        point_counter = 0
        for j in range(n_points):
            if V[i, j] == 1:
                idx_map[j] = point_counter
                point_counter += 1
        cam_point_ind_map.append(idx_map)

    initial_params = pack(Rs, ts, points_world)
    sparcity = build_sparcity(n_cameras, n_points, V)
    result = least_squares(
        residual,
        initial_params,
        args=(points_2d, n_cameras, n_points, V, K, cam_point_ind_map),
        jac_sparsity=sparcity,
    )

    Rs_opt, ts_opt, Xs_opt = unpack(result.x, n_cameras, n_points)
    # make array of a list created with list comprehension
    final_poses = np.array(
        [K @ np.hstack([R, t.reshape(3, 1)]) for R, t in zip(Rs_opt, ts_opt)]
    )
    final_points = np.array(Xs_opt)

    return final_poses, final_points
