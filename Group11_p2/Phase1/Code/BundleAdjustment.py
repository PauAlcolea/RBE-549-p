import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def bundleAdjustment(V: np.ndarray, 
                     all_poses:np.ndarray, 
                     points_world:np.ndarray,
                     camera_indices,
                     point_indices,
                     points_2d) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine the camera poses and 3D world points by optimization and minimization of the reprojection error

    :param V is the visibility matrix
    :param all poses are all of the poses of all of the cameras
    :param final_3D_world is all of the 3D world points after pnp
    :return the adjusted final poses and the adjustted final 3D world points in the forms of np.arrays
    """
    n_cameras = all_poses.shape[0]
    n_points2 = points_world.shape[0]



    # print(n_points, n_points2)
    pass
    # return final_poses, final_points


# def pack(all_poses:np.ndarray, points_world:np.ndarray) -> np.ndarray:
#     pass