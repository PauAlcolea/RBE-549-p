import numpy as np

def buildProjectionMatrices(K, Pose):
    """
    Docstring for buildProjectionMatrices
    
    :param K: camera internal matrix
    :param Pose: one pose, which is [R|t]
    """
    Projection = K @ Pose
    return Projection

def linearTriangulation(K, pose1: np.ndarray, pose2: np.ndarray, x1:np.ndarray, x2:np.ndarray):
    """
    this function triangulates 3D points
    
    :param pose1: Camera pose, this will always be zero because that is the reference
    :param pose2: another camera pose
    :param K: camera intrinsic maatrix
    :param correspondences between the two cameras, x1 and x2 np.ndarrays
    : expecting x1 and x2 in the form of [x1 y1 1] (np.array of shape (3,))
    :return 3d point, this will be 
    """
    # get the projections from the poses
    P1 = K @ pose1
    P2 = K @ pose2


    A = np.column_stack(np.cross(x1, P1), np.cross(x2, P2))
    U, D, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    
    return X


# buildProjectionMatrices(np.empty, np.column_stack((np.identity(3), np.ones(3))))
K = [[531.122155322710, 0, 407.192550839899],
     [0, 531.541737503901, 313.308715048366],
     [0, 0, 1]]
pose1 = np.column_stack((np.identity(3), np.zeros(3)))
pose2 = np.empty
x1 = np.empty
x2 = np.empty
linearTriangulation(K, pose1, pose2, x1, x2)