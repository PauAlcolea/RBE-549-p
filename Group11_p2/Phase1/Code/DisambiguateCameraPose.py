import numpy as np
from LinearTriangulation import linearTriangulation
from Visualization import plot_triangulation


def disambiguatePose(
    camera_poses: np.ndarray, K, correspondences, plot: bool = False
) -> np.ndarray:
    """
    Checking cheirality condition to remove ambiguity
    This function must go through of the poses, and get the triangulated points for each one

    :param camera_poses: this is a numpy array with 4 camera poses with a shape of (4, 3, 4)
    :param K: camera intrinsic matrix
    :param correspondences: this is a numpy array of shape (N, 2, 2) where each row is [[u1, v1], [u2, v2]]
    :param plot: whether to visualize the triangulated points for each candidate pose
    :return: the correct camera pose (3,4) and the triangulated points for that pose
    """

    identity_pose = np.hstack((np.identity(3), np.zeros((3, 1))))

    # the following list will contain the number of points that satisfy the cheirality condition
    valid_points_per_pose: list[int] = []
    # store triangulated points for each pose
    X_per_pose: list[list[np.ndarray]] = []

    # iterate through the poses and compare them to the identity pose
    for pose in camera_poses:
        # for each pose, go through all of the correspondences, between poses
        X_list: list[np.ndarray] = []

        for x1, x2 in correspondences:
            X_list.append(linearTriangulation(K, identity_pose, pose, x1, x2))

        valid_counter = 0

        R = pose[:, 0:3]
        t = pose[:, 3]
        for X in X_list:
            # count how many points are valid
            z1 = X[2]
            z2 = R[2, :] @ X + t[2]
            if z1 > 0 and z2 > 0:
                valid_counter += 1

        valid_points_per_pose.append(valid_counter)
        X_per_pose.append(X_list)

    # convert triangulated lists to arrays
    X_per_pose_arrays = [np.array(X_list) for X_list in X_per_pose]
    if plot:
        # visualize initial triangulations for all candidate poses
        try:
            plot_triangulation(
                *X_per_pose_arrays, title="initial triangulation for candidate poses"
            )
        except Exception:
            pass

    if all(v == 0 for v in valid_points_per_pose):
        raise ValueError("All poses have zero valid points, cannot disambiguate")

    best_index = valid_points_per_pose.index(max(valid_points_per_pose))
    best_pose = camera_poses[best_index]
    X_best = X_per_pose_arrays[best_index]
    return best_pose, X_best
