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
def corner_detection_single(image_color: np.array, image_gray: np.array):
	scores = []
	
	# (H,W) array with all of the scores
	# block size = 2, ksize = 3, Harris detector free parameter = 0.02
	change_score = cv2.cornerHarris(image_gray, 2, 3, 0.04)

	# make a boolean mask if any score is more than 1% of the maximum corner score
	# wherever the mask is true (corner), change the pixel to 0,0,255 (aka red)
	img_copy = image_color.copy()
	img_copy[change_score>0.01*change_score.max()]=[0,0,255]
	images_cornered = copy.deepcopy(img_copy)

	# The following code is for visualization
	# cv2.imshow('dst',images_cornered[index])
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	
	return images_cornered, change_score

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
def remove_corners_single(corners, image_gray, patch_radius=20):
	h, w = image_gray.shape
	good_corners = []

	for (x, y) in corners:
		if (
			x >= patch_radius and
			y >= patch_radius and
			x < w - patch_radius and
			y < h - patch_radius
		):
			good_corners.append((x, y))
	
	return good_corners

"""
This function will get the Nbest corners in the image based on their strength and their distance to another strong corner
@param corner_images is a list with all np.arrays (each array is the score for corners as a 2D mask for one photo)
@param Nbest is the number of best corners needed
@return the coordinates of all of the final corners in list
this will all be in a list that contains all of the corners for all the imgs
"""
def anms_single(corner_image: np.array, Nbest: int, gray_image) -> list[tuple[int, int]]:

	# each image has to be filtered to only keep the local maxima so any pixel with a value is not considered a corner
	# keep all of their coordinates
	corner_xs, corner_ys, strengths = local_maxima(corner_image)
	Nstrong = len(strengths)
	
	# if no strong corners go to next image
	if Nstrong == 0:
		return None

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
	# list of tuples, each tuple being the coordinates
	best_corners = list(zip(corner_xs[best_indexes], corner_ys[best_indexes]))
	
	#remove the corners that are too close to the edges
	final_corners = copy.deepcopy(remove_corners_single(best_corners, gray_image))
	return final_corners

"""
This function will take the different corners acquired by the anms and encode them into feature vectors to be identified in other images
@param images are the gray images, this will be used for size 
@param final_corners are the are the coordinates of the corners
@return a list of all of the feature vectors (64 x 1) inside a list for each image
"""
def feature_descriptor_single(final_corners: list[tuple[int, int]], image_color) -> list[list[np.array]]:
	image_vectors = []
	for corner in final_corners:
		x = corner[0]
		y = corner[1]
		a = 20 #this is to make the patch of size 41x41 (range(1+20:1-20) -> 41)

		# make a patch by slicing the gray image, add one becuase slicing excludes the final value
		# y and then x because the array is different like this
		# patch = image_gray[(y-a):(y+a+1), (x-a):(x+a+1)]
		patch_color = image_color[(y-a):(y+a+1), (x-a):(x+a+1)]

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
	
	return image_vectors

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

"""
Look at the corners of the images to ensure that both images stay in view, and then use the homography to stitch one to another
"""
def warp(image_from, image_to, homography):
	#get the corners for image one (actual 4 corners, not the corners we've been doing until now)
	#shape[:2] forgets about the color channel
	#the reshape gets separates every point so that the transform works right
	h1, w1 = image_from.shape[:2]
	corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1, 1, 2)
	#warp the corners from 1 with H, this is used to see where the picture will be and to ensure that all the images will be in the frame
	warped_corners1 = cv2.perspectiveTransform(corners1, homography)

	# this is to see where image 2 lies in space, no need to warp
	h2, w2 = image_to.shape[:2]
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
	panorama = cv2.warpPerspective(image_from, T @ homography, (xmax - xmin, ymax - ymin))
	# add image 2 to the panorama, shifting the position because of the T translation
	panorama[-ymin : h2-ymin, -xmin : w2-xmin] = image_to

	#crop any excess black pixels around the stitehced panorama so that you can view it properly
	gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
	return_val, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
	# bounding rectangle with the threshold
	x, y, w, h = cv2.boundingRect(threshold)
	panorama_cropped = panorama[y:y+h, x:x+w]

	return panorama_cropped


def compute_global_bbox(image_H_pairs):
    all_corners = []

    for img, H in image_H_pairs:
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped)

    all_corners = np.vstack(all_corners)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    return xmin, ymin, xmax, ymax

def simple_blend(base, new):
    mask = (new.sum(axis=2) > 0)

    blended = base.copy()
    blended[mask] = new[mask]
    return blended

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

	#begin with panorama as image one to stitch things to, calculate everything accordingly
	panorama = images[1]

	#n is the number of images, and the 
	n = len(images)
	middle_index = (n // 2)

	H_to_middle = [None] * n
	H_to_middle[middle_index] = np.eye(3, dtype=np.float32)

	# These two for loops are intended to go through every image and getting the homography between an image and the middle image
	# this is a list with all of the homographies to the RIGHT of the middle index (if it is 2 and there's 5 images, it would be H23, H34)
	H_right = []
	H_left = []

	# this for loop goes form the middle index up
	for i in range(middle_index, n-1):
		img1 = images[i]
		img2 = images[i+1]
		img_gray1 = images_gray[i]
		img_gray2 = images_gray[i+1]

		img_corners1, scores1 = corner_detection_single(img1, img_gray1)
		img_corners2, scores2 = corner_detection_single(img2, img_gray2)
		final_corners1 = anms_single(scores1, 120, img_gray1)
		final_corners2 = anms_single(scores2, 120, img_gray2)

		# visualize_anms([img1, img2],[final_corners1, final_corners2])

		feature_vectors1 = feature_descriptor_single(final_corners1, img1)
		feature_vectors2 = feature_descriptor_single(final_corners2, img2)

		key1 = corners_to_keypnts_one_img(final_corners1)
		key2 = corners_to_keypnts_one_img(final_corners2)

		#in here, feature vectors 2 and 1 are switched so that the homographies point towards the middle image
		matched_pairs = feature_matching_2_imgs(feature_vectors2, feature_vectors1)

		# matched_image = cv2.drawMatches(img1, key1, img2, key2, matched_pairs, outImg=None)
		# cv2.imshow("Feature Matches", matched_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		# key 2 and 1 are switched so that the homographies point towards the middle imgae
		H, inliers = RANSAC(matched_pairs, key2, key1, iterations=2000)
		H_right.append(H)

	# this is a list with all of the homographies to the LEFT of the middle index (if middle index is 2 and there's 5 images, it would be H21, H10)
	# this goes from one less than the middle index backwards to 0
	for j in reversed(range(n)[:middle_index]):
		img3 = images[j+1]
		img4 = images[j]
		img_gray3 = images_gray[j+1]
		img_gray4 = images_gray[j]
		
		img_corners3, scores3 = corner_detection_single(img3, img_gray3)
		img_corners4, scores4 = corner_detection_single(img4, img_gray4)
		final_corners3 = anms_single(scores3, 120, img_gray3)
		final_corners4 = anms_single(scores4, 120, img_gray4)

		# visualize_anms([img3, img4],[final_corners3, final_corners4])

		feature_vectors3 = feature_descriptor_single(final_corners3, img3)
		feature_vectors4 = feature_descriptor_single(final_corners4, img4)

		key3 = corners_to_keypnts_one_img(final_corners3)
		key4 = corners_to_keypnts_one_img(final_corners4)

		matched_pairs = feature_matching_2_imgs(feature_vectors4, feature_vectors3)

		# matched_image = cv2.drawMatches(img4, key4, img3, key3, matched_pairs, outImg=None)
		# cv2.imshow("Feature Matches", matched_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		H2, inliers = RANSAC(matched_pairs, key4, key3, iterations=2000)
		H_left.append(H2)

	# this iterates through all of the homographies, for the right and for the left
	# each of the homographies that is not directly going to the reference image (the one in the middle) 
	# gets recalculated to be in reference to that image
	for indexR in range(len(H_right)):
		if indexR != 0:
			H_right[indexR] =  H_right[indexR-1] @ H_right[indexR]

	for indexL in range(len(H_left)):
		if indexL != 0:
			H_left[indexL] = H_left[indexL-1] @ H_left[indexL]

	# make a list with the images and their corresponding homography in relation to the middle one
	# middle image is identity because it is the reference image
	image_H_pairs = []
	image_H_pairs.append((images[middle_index], np.eye(3)))

	# populate the list of all of the images and their corresponding homography to use later
	for i in range(len(H_right)):
		H = H_right[i]
		# H = H / H[2,2]
		image_H_pairs.append((images[middle_index + i + 1], H))

	for i in range(len(H_left)):
		H = H_left[i]
		# H = H / H[2,2]
		image_H_pairs.append((images[middle_index - i - 1], H))

	#make a canvas that will fit all of the homographies and images for the final panorama
	#empty canvas, all zeroes
	xmin, ymin, xmax, ymax = compute_global_bbox(image_H_pairs)
	canvas_w = xmax - xmin
	canvas_h = ymax - ymin
	panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

	#This translation is used for the warping
	T = np.array([[1, 0, -xmin],
				[0, 1, -ymin],
				[0, 0, 1]], dtype=np.float32)
	
	# calculate the warped view of the image and then mix it together nicely with the simple_blend helper function
	for img, H in image_H_pairs:
		warped = cv2.warpPerspective(img, T @ H, (canvas_w, canvas_h))
		panorama = simple_blend(panorama, warped)
		panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
		cv2.imshow("Final Panorama", panorama_bgr)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	#visualize but first change color
	panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
	cv2.imshow("Final Panorama", panorama_bgr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return
	




	# now that all fo the homographies are ready and have been made in relation to the middle image we can stitch
	# for i in range(len(H_right)):
	# 	H_right[i] = np.linalg.inv(H_right[i])
	# 	panorama = warp(images[middle_index + i + 1], panorama, H_right[i])

	# for i in range(len(H_left)):
	# 	H_left[i] = np.linalg.inv(H_left[i])
	# 	panorama = warp(images[middle_index - i - 1], panorama, H_left[i])

	a = [4, 5, 6]
	b = [3, 2, 1]
	b.reverse()
	c =  b + a

	# go through all of the images 
	for x in range(len(images) - 1):
		if x > 0:
			img1 = panorama
			img_gray1 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
		else:
			img1 = images[x]
			img_gray1 = images_gray[x]

		img2 = images[x+1]
		img_gray2 = images_gray[x+1]

		img_corners1, scores1 = corner_detection_single(img1, img_gray1)
		img_corners2, scores2 = corner_detection_single(img2, img_gray2)
		final_corners1 = anms_single(scores1, 200, img_gray1)
		final_corners2 = anms_single(scores2, 200, img_gray2)

		# visualize_anms([img1, img2],[final_corners1, final_corners2])

		feature_vectors1 = feature_descriptor_single(final_corners1, img1)
		feature_vectors2 = feature_descriptor_single(final_corners2, img2)

		key1 = corners_to_keypnts_one_img(final_corners1)
		key2 = corners_to_keypnts_one_img(final_corners2)

		matched_pairs = feature_matching_2_imgs(feature_vectors1, feature_vectors2)

		matched_image = cv2.drawMatches(img1, key1, img2, key2, matched_pairs, outImg=None)
		cv2.imshow("Feature Matches", matched_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		H, inliers = RANSAC(matched_pairs, key1, key2, iterations=2000)

		if H is None:
			print("Bad Homography between these two images, skip to next pairing")
			cv2.imshow("image1 problem", img1)
			cv2.waitKey(0)
			cv2.imshow("image2 problem", img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			continue

		panorama = warp(img1, img2, H)
		cv2.imshow("Final Panorama", panorama)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
	cv2.imshow("Final Panorama", panorama_bgr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return


if __name__ == "__main__":
    main()
