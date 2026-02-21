import numpy as np

def estimateEssentialMatrix(K , F):
    """
    estimate the essential matrix E from the fundamental matrix F and camera intrinsics K

    :param K: 3x3 camera intrinsic matrix
    :param F: 3x3 fundamental matrix
    :return: 3x3 essential matrix
    """
    E = K.T @ F @ K
    return E