import numpy as np
from LinearTriangulation import linearTriangulation


def disambiguatePose(camera_poses: np.ndarray, K, correspondences) -> np.ndarray:
    """
    Checking cheirality condition to remove ambiguity
    This function must go through of the poses, and get the triangulated points for each one

    :param camera_poses: this is a numpy array with 4 camera poses with a shape of (4, 3, 4)
    :return: only one camera pose (3,4) np array
    """

    identity_pose = np.hstack((np.identity(3), np.zeros((3, 1))))

    # the following list will contain the number of points that satisfy the cheirality condition
    valid_points_per_pose: list[int] = []
    # store triangulated points for each pose
    X_per_pose: list[list[np.ndarray]] = []

    # iterate through the poses and compare them to the identity pose
    for pose in camera_poses:
        # for each pose, go through all of the correspondances, between poses
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

    best_index = valid_points_per_pose.index(max(valid_points_per_pose))
    best_pose = camera_poses[best_index]
    X_best = np.array(X_per_pose[best_index])
    print(best_pose)
    return best_pose, X_best


a = np.array(
    [
        [-8.82430806e01, -7.27387578e01, -6.59407878e02, -2.56770069e02],
        [-7.90547804e01, 5.60840883e02, -2.44758688e02, -5.74798177e02],
        [5.98381760e-01, 3.23000663e-01, -7.33218823e-01, -7.71963692e-01],
    ]
)
