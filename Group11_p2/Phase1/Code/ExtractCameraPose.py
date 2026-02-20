import numpy as np
import sys


def extractCameraPose(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    This function will extract the four different possible camera poses out of the Essential Matrix E

    :param E: Essential matrix E
    :param K: Camera intrinsic Matrix
    :return: 4 possible poses that will get narrowed down with the triangulation
    """

    # SVD from essential matrix
    U, _, Vt = np.linalg.svd(E)

    S1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    S2 = S1.T

    # Two possible rotations that come from E = Tx R and the fact that U and Vt for svd(R) are the same as for svd(E)
    R1 = U @ S1 @ Vt
    R2 = U @ S2 @ Vt
    R = (R1, R2)

    # Two possible translations because epipole will lie on plane (left nullspace is in second image T.T @ E = 0)
    # T1 = u3
    T1 = U[:, 2]
    T2 = -T1
    T = (T1, T2)

    # print(np.column_stack((i, T[0])))

    P = []
    for t in T:
        for r in R:
            # use local variables to avoid overwriting t and r in the next loop iteration
            t_corr = t
            r_corr = r
            # Check determinant and correct camera pose if it's -1.
            if np.linalg.det(r_corr) < 0:
                t_corr = -t_corr
                r_corr = -r_corr
            pose = K @ r_corr @ np.column_stack((np.identity(3), t_corr))
            P.append(pose)

    return np.array(P)
