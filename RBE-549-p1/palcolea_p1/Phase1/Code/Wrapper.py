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

def read_set(image_set_path: str) -> list[np.array]:
	images = []
	
	#iterate through all of the files in the Image_set folder
	for file in os.listdir(image_set_path):
		if file.lower().endswith(".jpg"):
			# join the two names to make overall path of the image to then open with imread
			img_path = os.path.join(image_set_path, file)
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			images.append(img)

	return images

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
    Read a set of images for Panorama stitching
    """
	images = read_set("RBE-549-p1/palcolea_p1/Phase1/Data/Train/Set1/")
	print(images)


	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

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
