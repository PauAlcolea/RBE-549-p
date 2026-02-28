import numpy as np


def closeEnough(observed_Xs, final_3D_world, threshold=1e-6) -> bool:
    """
    this is a helper function that works to figure out if observed points are close enough to world points to assume that they are the same
    this is used to see if a point is being seen by a camera or not.

    :param observed_Xs: the 3D points observed by a camera
    :param final_3D_world: the set of all 3D points in the world for all cameras
    :param threshold: the distance threshold to consider points as matching

    :return: a boolean array indicating which world points are close enough to the observed points
    """
    observed_Xs = np.asarray(observed_Xs)
    final_3D_world = np.asarray(final_3D_world)
    # compute squared distances between all observed points and all world points
    diffs = observed_Xs[:, np.newaxis, :] - final_3D_world[np.newaxis, :, :]
    distances_sq = np.sum(diffs**2, axis=2)

    # get booleans indicating if any observed point is close enough to each world point
    matches = np.any(distances_sq < threshold, axis=0)
    return matches


def visibilityMatrix(final_3D_world, points_for_poses) -> np.ndarray:
    """
    This function builds the Visibility Matrix, a binary matrix that represents if a point is seen by a camera


    :param final_3D_world: is the set of all 3D points available, no matter what camera sees them
    :param points_for_poses: I envision this being a list with the same number of elements as poses / cameras / viewpoints
        each element should be a list or array of some sort containing all of the 3D or 2D points for that viewpoint
    :return: the visubility matrix ixj
    """
    V = np.zeros(shape=(len(points_for_poses), len(final_3D_world)))

    # go through all of the poses
    # take vectorized approach: for each view, compute distances to all world points at once and check for matches
    for i, observed_Xs in enumerate(points_for_poses):
        match_mask = closeEnough(observed_Xs, final_3D_world)
        V[i, match_mask] = 1
    return V
