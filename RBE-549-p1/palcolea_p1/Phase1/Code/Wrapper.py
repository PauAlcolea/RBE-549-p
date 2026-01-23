#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:
import numpy as np
import cv2
import os
import copy

"""
This function takes a path where a set of images to be made panorama is
It transforms the images into gray and opens them
@returns images wich is a list of np.arrays with the images in RGB
@returns images_gray which is the same except in gray scale, which is needed for the harris detector
"""
def read_set(image_set_path: str) -> tuple[list[np.array], list[np.array]]:
	images = []
	images_gray = []
	
	#iterate through all of the files in the Image_set folder
	for file in os.listdir(image_set_path):
		if file.lower().endswith(".jpg"):
			# join the two names to make overall path of the image to then open with imread
			# image has to be gray and a float for the harris detector
			img_path = os.path.join(image_set_path, file)
			img = cv2.imread(img_path)
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_gray = np.float32(img_gray)
			images.append(img_rgb)
			images_gray.append(img_gray)

	return images, images_gray


"""
this functin finds the corners in an image with the harris filter
used the opencv2 documentation: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
@params, two lists of images, in color and in grayscale
@returns one list of images with the corners
"""
def corner_detection(images_color: list[np.array], images_gray: list[np.array]) -> list[np.array]:
	scores = []
	
	# iterate through all of the images in the set
	for index, img_gray in enumerate(images_gray):
		
		# (H,W) array with all of the scores
		# block size = 2, ksize = 3, Harris detector free parameter = 0.02
		change_score = cv2.cornerHarris(img_gray, 2, 3, 0.04)
		scores.append(change_score)

		# make a boolean mask if any score is more than 1% of the maximum corner score
		# wherever the mask is true (corner), change the pixel to 0,0,255 (aka red)
		images_color[index][change_score>0.01*change_score.max()]=[0,0,255]
		images_cornered = copy.deepcopy(images_color)

		# The following code is for visualization
		# cv2.imshow('dst',images_cornered[index])
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
	
	return images_cornered, scores


"""
This function will
@param corner_images is a list with all np.arrays (each array is the score for corners as a 2D mask for one photo)
@param Nbest is the number of best corners needed
@return the coordinates of all of the final corners in list
this will all be in a list that contains all of the corners for all the imgs
"""
def anms(corner_images: list[np.array], Nbest: int):
	for cimg in corner_images:
		break
	
	return


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
    Read a set of images for Panorama stitching
    """
	images, images_gray = read_set("RBE-549-p1/palcolea_p1/Phase1/Data/Train/Set1/")

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	corners, scores = corner_detection(images, images_gray)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	anms(scores, 40)

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""

	"""
	Refine: RANSAC, Estimate Homography
	"""	
    

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	return


if __name__ == "__main__":
    main()
