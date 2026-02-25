import numpy as np
from LinearPnP import linearPnP
from scipy.optimize import least_squares


def reprojection_error(guess: tuple[np.ndarray, np.ndarray], K: np.ndarray, xs:np.ndarray, Xs:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    :return R and T, this will be optimiezd by the least squares
    """
    loss_list = []

    R0, T0 = guess
    P = K @ np.hstack([R0, T0.reshape(3,1)])
    P_1 = P[0, :]
    P_2 = P[1, :]
    P_3 = P[2, :]


    return loss_list


def nonlinearPnP(xs: np.ndarray, Xs: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    this function is used to calculate the pose of the camera for the added viewpoint in a non-linear manner
    This involves proper optimization by reducing the reprojection error as opposed to algebraic error minimized by the linear pnp

    
    :param x is a list of 6 2D points, they are the image projected points of X onto the image plane
        each point is an np.ndarray (x, y)
    :param X is a list of 6 3D points that correspond to x
    :K camera intrinsic matrix (3,3)
    """

    initial_guess = linearPnP(xs, Xs, K)

    print("STARTING NON-LINEAR OPTIMIZATION: ")
    result = least_squares(reprojection_error, initial_guess, args=(K, xs, Xs), method='lm')
    print("OPTIMIZATION COMPLETE")

    pass