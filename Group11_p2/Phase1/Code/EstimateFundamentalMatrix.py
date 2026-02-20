import numpy as np

def estimateFundamentalMat(eight_correspondances: np.ndarray) -> np.ndarray:
    """
    Docstring for estimateFundamentalMat
    This estiamtes the Fundamental matrix between two images given eight point correspondances
    
    :param eight_points: This is a numpy array with 8 correspondances between two images, shape (8, 2, 2) -> ([ [x1, y1], [x1', y2'] ], [], ...)
    :return: estimated 3x3 Fundamental Matrix, enforced Rank 2
    """

    # build A matrix that will satisfy the equation: Af = 0
    # In this case f would be the 9D vector with all the elements of the fundamental matrix F
    # A begins as a list and then becomes an array will be np.array of size 8x9
    #x2 and y2 are the projections of x and y on the second image (x' and y')
    A_list = []
    for (x, y), (x2, y2) in eight_correspondances:
        
        # append one row per correspondance
        A_list.append([x*x2, x*y2, x, y*x2, y*y2, y, x2, y2, 1])
    A = np.array(A_list)

    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]                       # last row of Vt is null space of A, i.e. solution f
    f[-1] = 0                           # enforce the rank 2
    F = f.reshape((3, 3)).T             # get the transpose because the notes have f in a different order (f11, f21, f31 ...) instead of (f11, f12, f13 ...)
    return F




# For testing purposes:
# B = np.array([[[0, 0], [5, 1]], [[1, 1], [64, 8]], [[92, 22], [2, 3]]])
# print(estimateFundamentalMat(B))
