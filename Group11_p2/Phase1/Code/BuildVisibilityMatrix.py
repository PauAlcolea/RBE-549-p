import numpy as np

def closeEnough(point1: np.ndarray, point2: np.ndarray, threshold=0.01) -> bool:
    """
    this is a helper function that works to figure out if point1 is close enough to point2 to assume that it is referring to the same one
    this is used to see if a point is being seen by a camera or not. Point1 might be calculated from a 2d point / being projected 
    point3 is the world point

    :param point1 3d projected point from a pose and a 2d ponint
    :param point2 world point (3d)  that point1 should be close to
    :param threshold dictates what is close enough or not
    :return whether it is a valid match (true) or not (false)
    """
    distance = np.linalg.norm(point1 - point2)
    return (distance < threshold)

def visibilityMatrix(all_poses, final_3D_world, points_for_poses) -> np.ndarray:
    """
    This function builds the Visibility Matrix, a binary matrix that represents if a point is seen by a camera

    :param all_poses in the system, these correspond to all the different views from were we are reconstructing the image
    :param final_3D_world is the set of all 3D points available, no matter what camera sees them
    :param points_for_poses I envision this beeing a list with the same number of elements as poses / cameras / viewpoints
        each element should be a list or array of some sort containing all of the 3D or 2D points for that viewpoint
    :return the visubility matrix ixj
    """
    V = np.zeros(shape=(len(all_poses), len(final_3D_world)))

    # go through all of the poses
    # for each pose, go through all of the points
    # check that point along with the projection of all of the 2d correspondences of that pose projected onto 3d
        # if any point is close enough to that point, mark the cell as 1, otherwise as 0

    for i, pose in enumerate(all_poses):
        for j, point in enumerate(final_3D_world):
            for point_2d in points_for_poses[i]:
                point3d = point_2d @ pose
                if closeEnough(point3d, point, threshold=0.01):
                    V[i][j] = 1
    
    return V