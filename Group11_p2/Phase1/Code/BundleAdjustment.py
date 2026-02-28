import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def project_points(Xs, R, t, K):
    """
    helper function to take N 3d points and project them onto an image plane
    this is faster than looping through each point and projecting it separately

    :param Xs: (N, 3) array of 3D points
    :param R, t, K: camera parameters
    :return: (N, 2) array of projected points
    """
    Xs_h = np.column_stack([Xs, np.ones(len(Xs))])  # Add homogeneous coordinate
    P = K @ np.hstack([R, t.reshape(3, 1)])
    x_proj_h = (P @ Xs_h.T).T  # (N, 3)
    x_proj = x_proj_h[:, :2] / x_proj_h[:, 2:3]  # Normalize by z
    return x_proj


def pack(Rs, ts, Xs) -> np.ndarray:
    """
    This helper function is used for the optimization. The parameters need to be packed so that they can be inputted into scipy.optimize
    it takes the rodrigues vectors of the Rotation matrix, which is simpler and packs easier
    the packing goes [R1, t1, R2, t2, ... Rm, tm, X1, X2, X3 ... Xn]
    """
    params = []
    for R, t in zip(Rs, ts):
        rvec = Rot.from_matrix(R).as_rotvec()
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
        R = Rot.from_rotvec(rvec).as_matrix()
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

    :param params: the packed parameters including all of the Rs, Ts and Xs
    :param points_2d: all of the 2d points, a list or array where each element is a list or array for all the points of one camera
    :param n_cameras: number of cameras
    :param n_points: number of points
    :param K: intrinsic camera matrix
    :param V: visibility matrix
    :param point2d_idx_map: connects every 2d point with the 3d point that it is a projection of
    """
    Rs, ts, Xs = unpack(params, n_cameras, n_points)

    res = []

    # go through every camera
    for i in range(n_cameras):
        # get all points visible to this camera
        visible_point_indices = [j for j in range(n_points) if V[i, j] == 1]
        if not visible_point_indices:
            continue

        # vectorized projection for all visible points
        visible_Xs = np.array([Xs[j] for j in visible_point_indices])
        x_proj = project_points(visible_Xs, Rs[i], ts[i], K)

        # get corresponding observed points
        ind_in_points2d = [point2d_idx_map[i][j] for j in visible_point_indices]
        x_obs = points_2d[i][ind_in_points2d]

        # compute residuals for this camera
        residual = x_obs - x_proj
        res.append(residual.flatten())

    return np.concatenate(res)


def build_sparsity(num_cams, num_points, V):
    """
    This function is used to make a sparsity matrix that can be passed to the optimizer so that it runs much faster
    it knows which derivatives are zero so that the optimizer doesn't have to calculate them
    """
    n_params = num_cams * 6 + num_points * 3
    # each pair of camera and points will give you 2 residuals, error for u and error for v
    n_residuals = int(np.sum(V)) * 2

    # pre-allocate arrays for sparse matrix construction
    rows, cols = [], []

    res_idx = 0
    for i in range(num_cams):
        for j in range(num_points):
            if V[i, j] == 1:
                # Each observation contributes 2 residuals (u, v)
                for r in range(2):
                    row = res_idx + r

                    # each camera has 6 parameters, 3 for rotation and 3 for translation
                    # 2 residuals depend on these 6 params
                    camera_cols = list(range(i * 6, i * 6 + 6))
                    rows.extend([row] * 6)
                    cols.extend(camera_cols)

                    # each point has 3 parameters, X, Y and Z
                    # 2 residuals depend on these 3 params
                    point_cols = list(
                        range(num_cams * 6 + j * 3, num_cams * 6 + j * 3 + 3)
                    )
                    rows.extend([row] * 3)
                    cols.extend(point_cols)

                res_idx += 2

    # create sparse matrix
    data = np.ones(len(rows))
    sparsity = lil_matrix((n_residuals, n_params), dtype=int)
    sparsity[rows, cols] = data

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
    # making a dictionary to 3d points corresponding to which rows in points_2d[i]
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
    sparsity = build_sparsity(n_cameras, n_points, V)
    result = least_squares(
        residual,
        initial_params,
        args=(points_2d, n_cameras, n_points, V, K, cam_point_ind_map),
        jac_sparsity=sparsity,
    )

    Rs_opt, ts_opt, Xs_opt = unpack(result.x, n_cameras, n_points)
    # make array of a list created with list comprehension
    final_poses = np.array(
        [K @ np.hstack([R, t.reshape(3, 1)]) for R, t in zip(Rs_opt, ts_opt)]
    )
    final_points = np.array(Xs_opt)

    return final_poses, final_points
