import numpy as np

def disambiguatePose(camera_poses: np.ndarray) -> np.ndarray:
    """
    Checking cheirality condition to remove ambiguity
    
    :param camera_poses: this is a numpy array with 4 camera poses with a shape of (4, 3, 4)
    :return: only one camera pose (3,4) np array
    """
    pass

a = np.array([[-8.82430806e+01, -7.27387578e+01, -6.59407878e+02, -2.56770069e+02],
              [-7.90547804e+01,  5.60840883e+02, -2.44758688e+02, -5.74798177e+02],
              [ 5.98381760e-01,  3.23000663e-01, -7.33218823e-01, -7.71963692e-01]])

print(a.shape)