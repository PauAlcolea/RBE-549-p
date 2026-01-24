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
from sklearn.preprocessing import StandardScaler


"""
Draws ANMS corners on each image and shows them using cv2.imshow.
@param images_color: list of RGB images
@param anms_corners: list of list of the tuples (x, y) from ANMS
"""
def visualize_anms(images_color: list[np.array], anms_corners: list[list[tuple]]):
    for idx, img in enumerate(images_color):
        # Make a copy and convert RGB to BGR to display in OpenCV
        img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        corners = anms_corners[idx]

        # Draw a small red circle for each corner
        for (x, y) in corners:
            cv2.circle(img_copy, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # Show the image
        cv2.imshow(f"ANMS Corners Image {idx+1}", img_copy)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()


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
This function will take a dense score image and extract the best corners
@param cimg is the score for an image, its a 2d matrix size (H, W)
@returns the x and y coordinates of the local maxima corners
@returns strengths which are the scores of each of the corners
"""
def local_maxima(cimg: np.array):
	Xs =[]
	Ys = []
	strengths = []

	#dilate the image, which means that each pixel gets replaced with the max value of its neighborhood, 3x3 neighborhood default
	#then when the values of the original are the same as the dilated, that original was a local max
	#local_max is just a boolean mask
	dilated = cv2.dilate(cimg, None)
	local_max = (cimg == dilated)
	
	#get rid of really small peaks 
	#calculate the threshold with a percentage of the maximum score
	threshold = 0.01 * cimg.max()
	local_max &= (cimg > threshold)

	# x,y for local maxima are inverted
	# get x and y coordinates
	Ys, Xs = np.where(local_max)
	strengths = cimg[Ys, Xs]

	return Xs, Ys, strengths

"""
This function will get the Nbest corners in the image based on their strength and their distance to another strong corner
@param corner_images is a list with all np.arrays (each array is the score for corners as a 2D mask for one photo)
@param Nbest is the number of best corners needed
@return the coordinates of all of the final corners in list
this will all be in a list that contains all of the corners for all the imgs
"""
def anms(corner_images: list[np.array], Nbest: int) -> list[list[tuple[int, int]]]:
	# list with a list of tuples, each tuple being the coordinates and the sublist being an image
	final_corners_images = []

	for cimg in corner_images:
		# each image has to be filtered to only keep the local maxima so any pixel with a value is not considered a corner
		# keep all of their coordinates
		corner_xs, corner_ys, strengths = local_maxima(cimg)
		Nstrong = len(strengths)
		
		# if no strong corners go to next image
		if Nstrong == 0:
			final_corners_images.append([])
			continue

		# make a list of size Nstrong with np.infinity to compare
		r = np.full(Nstrong, np.inf)

		# iterate through all of the local maxima corners
		for i in range(Nstrong):
			for j in range(Nstrong):
				# if a stronger corner found, calcualte ED
				if strengths[j] > strengths[i]:
					ED = (corner_xs[j] - corner_xs[i])**2 + (corner_ys[j] - corner_ys[i])**2
					if ED < r[i]:
						r[i] = ED

		# get the indices of what would be the corners if sorted in descending
		sorted_indexes = np.argsort(-r)	
		best_indexes = sorted_indexes[:Nbest]
		best_corners = list(zip(corner_xs[best_indexes], corner_ys[best_indexes]))
		final_corners_images.append(best_corners)
	return final_corners_images


"""
This function will take the different corners acquired by the anms and encode them into feature vectors to be identified in other images
@param images are the gray images, this will be used for size 
@param final_corners are the are the coordinates of the corners
@return a list of all of the feature vectors (64 x 1) inside a list for each image
"""
def feature_descriptor(images_gray: list[np.array], final_corners: list[list[tuple[int, int]]]) -> list[list[np.array]]:
	feature_vectors = []
	for i, image in enumerate(final_corners):
		image_vectors = []
		for corner in image:
			x = corner[0]
			y = corner[1]
			a = 20 #this is to make the patch of size 41x41 (range(1+20:1-20) -> 41)
			h, w = images_gray[i].shape

			# if out of bounds go to next corner
			if (x < a) or (y < a) or (x >= w - a) or (y >= h - a):
				continue

			# make a patch by slicing the gray image, add one becuase slicing excludes the final value
			# y and then x because the array is different like this
			patch = images_gray[i][(y-a):(y+a+1), (x-a):(x+a+1)]

			blurred_patch = cv2.GaussianBlur(patch, ksize=(5,5), sigmaX=1, sigmaY=1)
			patch_8x8 = cv2.resize(blurred_patch,(8,8), interpolation=cv2.INTER_LINEAR)
			vector = patch_8x8.reshape(64,)

			# standardize
			scaler = StandardScaler()
			standardized_vector = scaler.fit_transform(vector)

			image_vectors.append(standardized_vector)

		feature_vectors.append(image_vectors)	
	
	return feature_vectors


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
	final_corners = anms(scores, 40)
	# visualize_anms(images, final_corners)
	
	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	feature_descriptor(images_gray, final_corners)

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
