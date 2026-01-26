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
import random

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
	for file in sorted(os.listdir(image_set_path)):
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
		img_copy = images_color[index].copy()
		img_copy[change_score>0.01*change_score.max()]=[0,0,255]
		images_cornered = copy.deepcopy(img_copy)

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
this function goes over all of the corners produced by anms and removes the ones that are too close to the border for the feature descriptor
"""
def remove_corners(all_corners, images_gray, patch_radius=20):
    good_corners_all_images = []

	#iterate through every image
    for i, image in enumerate(all_corners):
        h, w = images_gray[i].shape
        good_corners = []

        for (x, y) in image:
            if (
                x >= patch_radius and
                y >= patch_radius and
                x < w - patch_radius and
                y < h - patch_radius
            ):
                good_corners.append((x, y))

        good_corners_all_images.append(good_corners)

    return good_corners_all_images

"""
This function will get the Nbest corners in the image based on their strength and their distance to another strong corner
@param corner_images is a list with all np.arrays (each array is the score for corners as a 2D mask for one photo)
@param Nbest is the number of best corners needed
@return the coordinates of all of the final corners in list
this will all be in a list that contains all of the corners for all the imgs
"""
def anms(corner_images: list[np.array], Nbest: int, gray_images) -> list[list[tuple[int, int]]]:
	# list with a list of tuples, each tuple being the coordinates and the sublist being an image
	corners_images = []

	for cimg in corner_images:
		# each image has to be filtered to only keep the local maxima so any pixel with a value is not considered a corner
		# keep all of their coordinates
		corner_xs, corner_ys, strengths = local_maxima(cimg)
		Nstrong = len(strengths)
		
		# if no strong corners go to next image
		if Nstrong == 0:
			corners_images.append([])
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
		corners_images.append(best_corners)
	
		#remove the corners that are too close to the edges
		final_corners_images = copy.deepcopy(remove_corners(corners_images, gray_images))
	return final_corners_images

"""
This function will take the different corners acquired by the anms and encode them into feature vectors to be identified in other images
@param images are the gray images, this will be used for size 
@param final_corners are the are the coordinates of the corners
@return a list of all of the feature vectors (64 x 1) inside a list for each image
"""
def feature_descriptor(images_gray: list[np.array], final_corners: list[list[tuple[int, int]]], images_color) -> list[list[np.array]]:
	feature_vectors = []
	for i, image in enumerate(final_corners):
		image_vectors = []
		for corner in image:
			x = corner[0]
			y = corner[1]
			a = 20 #this is to make the patch of size 41x41 (range(1+20:1-20) -> 41)
			h, w = images_gray[i].shape

			# if out of bounds go to next corner
			# if (x < a) or (y < a) or (x >= w - a) or (y >= h - a):
			# 	continue

			# make a patch by slicing the gray image, add one becuase slicing excludes the final value
			# y and then x because the array is different like this
			patch = images_gray[i][(y-a):(y+a+1), (x-a):(x+a+1)]
			patch_color = images_color[i][(y-a):(y+a+1), (x-a):(x+a+1)]

			descriptor = []
			#iterate through every channel and get the 
			for c in range(3):
				#blur the color patch (only one channel)
				#make the patch smaller
				#resize it into one dimension
				channel = patch_color[:,:,c]
				channel = cv2.GaussianBlur(channel, (7,7), sigmaX=2)
				small = cv2.resize(channel, (8,8), interpolation=cv2.INTER_LINEAR)
				descriptor.append(small.flatten())

			# make a big feature vector with info from each channel / color, which ends up being 192x1
			vector = np.concatenate(descriptor).astype(np.float32)

			# blurred_patch = cv2.GaussianBlur(patch, ksize=(7,7), sigmaX=1, sigmaY=1)
			# patch_8x8 = cv2.resize(blurred_patch,(8,8), interpolation=cv2.INTER_LINEAR)
			# vector = patch_8x8.reshape(64,1)

			# standardize
			vector /= np.linalg.norm(vector) + 1e-6
			image_vectors.append(vector)

		feature_vectors.append(image_vectors)	
	
	return feature_vectors

"""
this function is meant to take all of the feature vectors in an image and match them (is possible) to the feature vector in a different picture
@param feature_vectors1 and feature_vectors2: are lists with vectors, one for each image in the comparison
"""
def feature_matching_2_imgs(feature_vect1: list[np.array], feature_vect2: list[np.array]):		
	#will be a list of tuples between these two images
	matches = []
	
	# go through each vector in the first image and then inside of that iterate through all of the vectors on the second image
	for i, vector1 in enumerate(feature_vect1):
		
		#initialize the best and second best matches
		#match is where the pair is going to go, 
		best_match = np.inf
		second_best = np.inf
		j_best = -1
		match = [[],[]]
		
		for j, vector2 in enumerate(feature_vect2):
			#square of differences between the two vectors
			sum_sq_diff = np.sum((vector2 - vector1)**2)

			#update bests as you go
			if sum_sq_diff < best_match:
				second_best = best_match
				best_match = sum_sq_diff
				j_best = j
				match[0] = copy.deepcopy(vector1)
				match[1] = copy.deepcopy(vector2)

		#check that there is some match
		#check if the distances for the matches are significant or not with a threshold
		#if they are, accept the match by making a 
		#i and j_best are the indices into keypoints for one image each
		if j_best!=-1 and best_match/second_best < 0.5:
			matches.append(cv2.DMatch(i ,j_best , best_match))

	return(matches)
		
"""
helper function to put corners into cv2 KeyPoint objects for the visualization of the feature matching
"""
def corners_to_keypnts_one_img(final_corners: list[tuple[int, int]]):
	keypoints = []

	# go through all of coordinate tuples in an image
	for (x, y) in final_corners:
		kp = cv2.KeyPoint(x=float(x), y=float(y), size=1)
		keypoints.append(kp)
	return keypoints

"""
helper function that computes homography
sources and destinations are lists with the 4 randomly chosen points (one for the ones in image 1 and the others for image 2)
"""
def homography(sources, destinations):
	H = []
	for i in range(len(sources)):
		p = np.array([sources[i][0], sources[i][1], 1]).T
		p_prime = np.array([destinations[i][0], destinations[i][1], 1]).T
		return
	return H

"""
function that runs RANSAC, but only for two images
@param matched_pairs is a list with all of the matched pairs in cv2.DMatch objects
@param key1 and key2 are the keypoints for corners, the matched pairs contain the indexes that can pull the actual points
@param iterations are the maximum iterations
"""
def RANSAC(matched_pairs, key1, key2, iterations):
	if len(matched_pairs) < 4:
		print("Not enough matches for RANSAC")
		return None, None
	
	#convert the DMatch objects into two list of arrays with all the source points and destination points
	#they are each lists of tuples
	src_points = np.array([key1[match.queryIdx].pt for match in matched_pairs], dtype=np.float32)
	dest_points = np.array([key2[match.trainIdx].pt for match in matched_pairs], dtype=np.float32)

	# this function does all of the RANSAC for me, but I still know what it does because I studied it. 
	# The general understanding that I have is that H is a transofmration and I need to iterate through sets of 4 because 4 pairs of points 
	# are what is needed to fully define the equations, the iteration is to make sure that the connections were mostly correct, which it releases
	H, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, ransacReprojThreshold=5, maxIters=iterations)

	inlier_matches = [matched_pairs[i] for i in range(len(matched_pairs)) if mask[i]]

	return H, inlier_matches


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
    Read a set of images for Panorama stitching
    """
	images, images_gray = read_set("RBE-549-p1/palcolea_p1/Phase1/Data/Train/Set3/")

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	corners, scores = corner_detection(images, images_gray)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	final_corners = anms(scores, 150, images_gray)
	# visualize_anms(images, final_corners)
	
	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	feature_vectors = feature_descriptor(images_gray, final_corners, images)

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	#feature vectors for each image
	vectors1 = feature_vectors[0]
	vectors2 = feature_vectors[1]

	#original images in question
	image1 = images[0]
	image2 = images[1]

	#corners of an image after anms but in keypoint objects
	key1 = corners_to_keypnts_one_img(final_corners[0])
	key2 = corners_to_keypnts_one_img(final_corners[1])

	# Visualization of the images to be stitched
	# cv2.imshow("image1", image1)
	# cv2.waitKey(0)
	# cv2.imshow("image2", image2)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	matched_pairs = feature_matching_2_imgs(vectors1, vectors2)
	
	#visualization
	matched_image = cv2.drawMatches(image1, key1, image2, key2, matched_pairs, outImg=None)
	# cv2.imshow("Feature Matches", matched_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	"""
	Refine: RANSAC, Estimate Homography
	"""	
	H, inliers = RANSAC(matched_pairs, key1, key2, iterations=2000)

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	#get the corners for image one (actual 4 corners, not the corners we've been doing until now)
	#shape[:2] forgets about the color channel
	#the reshape gets separates every point so that the transform works right
	h1, w1 = image1.shape[:2]
	corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1, 1, 2)
	#warp the corners from 1 with H, this is used to see where the picture will be and to ensure that all the images will be in the frame
	warped_corners1 = cv2.perspectiveTransform(corners1, H)

	# this is to see where image 2 lies in space, no need to warp
	h2, w2 = image2.shape[:2]
	corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1, 1, 2)

	#combine all of the corners vertically
	all_corners = np.vstack((warped_corners1, corners2))
	
	#bounding box by looking at the coordinates for each corner, 
	xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
	xmax, ymax = np.int32(all_corners.max(axis=0).ravel())
	
	#now we need a transformation to move everything by the minimum x and y to make that the origin so everything is in view
	T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)

	# do the actual homography application for the panorama and then the translation
	# T @ H is matrix multiplication, combining the homographies
	panorama = cv2.warpPerspective(image1, T @ H, (xmax - xmin, ymax - ymin))
	# add image 2 to the panorama, shifting the position because of the T translation
	panorama[-ymin : h2-ymin, -xmin : w2-xmin] = image2

	cv2.imshow("Panorama", panorama)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return


if __name__ == "__main__":
    main()
