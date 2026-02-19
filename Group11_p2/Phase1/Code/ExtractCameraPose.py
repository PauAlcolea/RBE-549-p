import numpy as np
import sys

def extractCameraPose(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Docstring for extractCameraPose, this function will extract the four different possible camera poses out of the Essential Matrix E
    
    :param E: Essential matrix E
    :param K: Camera intrinsic Matrix
    :return: 4 possible poses that will get narrowed down with the triangulation
    """
    
    # SVD from essential matrix
    U, D, Vt = np.linalg.svd(E)

    S1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    S2 = S1.T

    # Two possible rotations that come from E = Tx R and the fact that U and Vt for svd(R) are the same as for svd(E)
    R1 = U @ S1 @ Vt
    R2 = U @ S2 @ Vt
    R = (R1, R2)

    # Two possible translations because epipole will lie on plane (left nullspace is in second image T.T @ E = 0)
    # T1 = u3
    T1 = U[:,2]
    T2 = -T1
    T = (T1, T2)

    # print(np.column_stack((i, T[0])))

    P = []
    for t in T:
        for r in R:
            # Check determinant and correct camera pose if it's -1. (is it supposed to be equal to 1??)
            if np.linalg.det(r) < 0:
                t = -t
                r = -r
            pose = K @ r @ np.column_stack((np.identity(3), t))
            P.append(pose)


    return np.array(P)

# For testing purposes:
# exampleE = [[1, 2, 0], 
#             [3, 4, 5], 
#             [6, 7, 8]]

# K = [[531.122155322710, 0, 407.192550839899],
#      [0, 531.541737503901, 313.308715048366],
#      [0, 0, 1]]

# print(extractCameraPose(exampleE, K))