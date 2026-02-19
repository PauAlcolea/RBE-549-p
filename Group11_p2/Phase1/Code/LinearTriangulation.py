import numpy as np

def buildProjectionMatrices(K, P):
    """
    Docstring for buildProjectionMatrices
    
    :param K: camera internal matrix
    :param P: one pose
    """
    R = P[:,0:3]
    C = P[:,3]

    

    pass

def linearTriangulation(pose1: np.ndarray, pose2: np.ndarray, x1:np.ndarray, x2:np.ndarray):
    """
    this function triangulates 3D points
    
    :param pose1: Camera pose, this will always be zero because that is the reference
    :param pose2: another camera pose
    :param K: camera intrinsic maatrix
    :param correspondences between the two cameras, x1 and x2 np.ndarrays
    :return 3d points
    """
    pose1 = np.column_stack((np.identity(3), np.zeros(3)))
    print(pose1)

    pass

# linearTriangulation(np.empty, np.empty)
# buildProjectionMatrices(np.empty, np.column_stack((np.identity(3), np.ones(3))))