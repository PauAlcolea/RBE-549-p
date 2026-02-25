import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot 


def reprojection_error(params: np.ndarray, K: np.ndarray, xs:np.ndarray, Xs:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This helper function is used in the optimizer, it gets inputted the parameters in a 1d array and then extracts the T and q to 
    calculate the geometric difference

    :param params are T and q in the form of a 1d array
    :param K is the camera intrinsic matrix
    :param xs and Xs are the 2d image and 3d world points that correspond to each other
    :return the residuals to be optimized
    """

    # extract all of the parameters
    T = params[0:3]
    q = params[3:7]

    # normalization of quaternion
    q = q / np.linalg.norm(q)

    # turn quaternion back into a rotation matrix
    R = Rot.from_quat(q).as_matrix()

    P = K @ np.hstack([R, T.reshape(3,1)])
    P_1 = P[0, :]
    P_2 = P[1, :]
    P_3 = P[2, :]

    loss_list = []
    for (u, v), X in zip(xs, Xs):

        X_tilde = np.append(X, 1)

        e1 = (u - (P_1 @ X_tilde) / (P_3 @ X_tilde))
        e2 = v - (P_2 @ X_tilde) / (P_3 @ X_tilde)
        
        loss_list.append([e1, e2])

    residuals = np.concatenate(loss_list)
    return residuals


def nonlinearPnP(xs: np.ndarray, Xs: np.ndarray, K: np.ndarray, R0, T0) -> tuple[np.ndarray, np.ndarray]:
    """
    this function is used to calculate the pose of the camera for the added viewpoint in a non-linear manner
    This involves proper optimization by reducing the reprojection error as opposed to algebraic error minimized by the linear pnp
    
    :praram xs and Xs are the correspondences between projected points and world poitns
    :param K camera intrinsic matrix (3,3)
    :param R0 and T0 are initial values used as initial parameters for the optimization, these can come from linear pnp or pnpRansac in Wrapper
    :return the optimized version of R and T
    """  
    # use library to transform rotation matrices to quaternions and viceversa
    q0 = Rot.from_matrix(R0).as_quat()

    initial_parameters = np.hstack((T0, q0))

    print("Starting Non-Linear Optimization: ")
    result = least_squares(reprojection_error, 
                           initial_parameters, 
                           args=(K, xs, Xs), 
                           method='lm')
    print("Optimization Complete")

    optimized_params = result.x
    T_opt = optimized_params[0:3]
    q_opt = optimized_params[3:7]

    q_opt = q_opt / np.linalg.norm(q_opt)
    R_opt = Rot.from_quat(q_opt).as_matrix()

    return R_opt, T_opt