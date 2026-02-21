#!/usr/bin/env python

def main():
    
    ######## NOTES ##########

    # when calling estimateFundamentalMat(correspondances), the correspondances should be an np.array of shape (8, 2, 2)
    # it should look like this: 
    # np.array([[[x1, y1], [x1', y1']], [[x2, y2], [x2', y2']], [[x3, y3], [x3', y3']] , ... , ... [[x8, y8], [x8', y8']] ])
    
    # the camera poses are already in the form [R|t], so that can be passed directly as one matrix to the linear triangulation function and things
    


    # Questions:
    # for the linear triangluation, do i need to make the projective matrices transposes? what is the original equation x1 = PX?
    # What really is the camera pose from ExtractCameraPose.py? is it already the Projection Matrix?
    # There's a chance that the the pose is extracted wrong, maybe it should be a 4x4, not a 3x4, look at the notes,
    pass

if __name__ == "__main__":
    main()