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
import argparse
import os
from matplotlib import pyplot as plt
from skimage.feature import corner_peaks


def cylindrical_warp(flat_image, cylinder_radius):
    """
    warp a flat image onto a cylindrical surface.
    used to mitigate distortion of image sets with large FOV
    """
    height, width = flat_image.shape[:2]

    # create (x,y) coordinate grid for cylinder image
    y_on_cylinder, x_on_cylinder = np.indices((height, width))
    center_x = width / 2
    center_y = height / 2

    # set of angles from center to each pixel
    theta = (x_on_cylinder - center_x) / cylinder_radius

    # set of height offsets from center to each pixel
    height_offset = (y_on_cylinder - center_y) / cylinder_radius

    # get flat image coordinates for each theta on cylinder
    x_on_flat_image = np.tan(theta) * cylinder_radius + center_x

    # get flat image coordinates for each height offset on cylinder
    y_on_flat_image = (
        height_offset * np.sqrt(1 + np.tan(theta) ** 2) * cylinder_radius + center_y
    )

    # use maps to fill in cylinder image with corresponding pixels from flat image
    warped_image = cv2.remap(
        src=flat_image,
        map1=x_on_flat_image.astype(np.float32),
        map2=y_on_flat_image.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )

    return warped_image


def locate_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    C_img = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.06)
    return C_img


def ANMS(C_img, N_best=1000):
    # find local maxima
    coordinates = corner_peaks(C_img, min_distance=5)
    # sort corners by corner strength (descending)
    values = C_img[coordinates[:, 0], coordinates[:, 1]]
    sorted_indices = np.argsort(-values)
    sorted_coords = coordinates[sorted_indices]
    # find minimum distance from each corner i to a stronger corner j
    N = len(sorted_coords)
    radii = np.full(N, np.inf)
    for i in range(1, N):
        yi, xi = sorted_coords[i]
        for j in range(i):  # iterate over corners stronger than i
            yj, xj = sorted_coords[j]
            dist = (xi - xj) ** 2 + (yi - yj) ** 2
            radii[i] = min(radii[i], dist)

    # select N_best corners with largest radii
    best_indices = np.argsort(-radii)[:N_best]
    return sorted_coords[best_indices][:, [1, 0]]  # swap from (y,x) to (x,y)


def encode_feature_points(image, corners):
    descriptors = []
    valid_corners = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    PATCH_RADIUS = 20
    for x, y in corners:
        # extract 41x41 patch around corner
        if (
            x - PATCH_RADIUS < 0
            or x + PATCH_RADIUS >= W
            or y - PATCH_RADIUS < 0
            or y + PATCH_RADIUS >= H
        ):
            continue
        patch = gray[
            y - PATCH_RADIUS : y + PATCH_RADIUS + 1,
            x - PATCH_RADIUS : x + PATCH_RADIUS + 1,
        ]
        patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=2)
        # downsample, normalize, and flatten
        small_patch = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        std = max(np.std(small_patch), 1e-5)
        norm_patch = (small_patch - np.mean(small_patch)) / std
        descriptor = norm_patch.flatten()
        descriptors.append(descriptor)
        valid_corners.append((x, y))
    # encoded feature descriptors, corresponding corner locations
    return np.array(descriptors), np.array(valid_corners)


def _ssd(point, points):
    # compute sum of squared distance from point to all points"""
    return np.sum((points - point) ** 2, axis=1)


def match_features(fd1, fd2, ratio_thresh=0.7):
    match_indices = []
    for i, d1 in enumerate(fd1):
        # compute sum of square difference with all descriptors in fd2
        distances = _ssd(d1, fd2)
        # find the two closest matches
        best_match_idx = np.argmin(distances)
        dist_best = distances[best_match_idx]
        distances[best_match_idx] = (
            np.inf
        )  # exclude the nearest neighbor to find the second nearest
        dist_second_best = distances[np.argmin(distances)]
        # apply ratio test
        if dist_best / dist_second_best < ratio_thresh:
            match_indices.append((i, best_match_idx))
    return match_indices  # list of (index in fd1, index in fd2)


def _corners_to_keypoints(corners):
    # convert corner coordinates (x,y) to cv2.KeyPoint objects
    keypoints = []
    for x, y in corners:
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=5)
        keypoints.append(kp)
    return keypoints


def _matches_to_dmatches(matches):
    # convert (i,j) tuples to cv2.DMatch objects
    dmatches = []
    for i, j in matches:
        dm = cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0)
        dmatches.append(dm)
    return dmatches


def draw_matches(image1, image2, matches, corners, window_name="Matches"):
    kp1 = _corners_to_keypoints(corners[0])
    kp2 = _corners_to_keypoints(corners[1])
    dmatches = _matches_to_dmatches(matches)
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, dmatches, None)
    cv2.imshow(window_name, img_matches)
    return img_matches


def _pairs_from_matches(matches, corners):
    # get ((x1,y1), (x2,y2)) pairs of corner coordinates from match indices
    pairs = []
    for i1, i2 in matches:
        pairs.append((corners[0][i1], corners[1][i2]))
    return np.array(pairs)  # Nx2 arrays of (x,y) coordinates


def _normalize_points(points):
    points = np.asarray(points)

    # shift points to have mean at origin
    mean = np.mean(points, axis=0)
    shifted = points - mean

    # scale so mean distance is sqrt(2)
    dists = np.linalg.norm(shifted, axis=1)
    mean_dist = np.mean(dists)
    scale = np.sqrt(2) / mean_dist

    # construct normalization matrix and normalize
    T = np.array(
        [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
    )
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_norm_h = (T @ points_h.T).T
    points_norm = points_norm_h[:, :2]

    return points_norm, T  # normalized points, normalization matrix


def _form_A_matrix(pairs):
    """
    Form the A matrix for homography estimation
    A = [x1 y1 1 0 0 0 -x2*x1 -x2*y1 -x2
         0 0 0 x1 y1 1 -y2*x1 -y2*y1 -y2]
    for each correspondence (x1,y1) <-> (x2,y2).
    A is expanded from rearranging
    [x2 y2 w2]^T = H [x1 y1 1]^T
    =>
    A*h = 0
    where w2 is scale factor that gets divided out
    """
    A = []
    for (x1, y1), (x2, y2) in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    return np.array(A)  # 2nx9 matrix


def _compute_homography(pairs):
    # normalize points
    points1_norm, T1 = _normalize_points(pairs[:, 0])
    points2_norm, T2 = _normalize_points(pairs[:, 1])
    norm_pairs = np.array(list(zip(points1_norm, points2_norm)))
    # form A matrix and solve A*h=0 using SVD
    A = _form_A_matrix(norm_pairs)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # last row of Vt is null space of A, i.e. solution h
    H = h.reshape((3, 3)) / np.max(np.abs(h))
    # denormalize H
    H = np.linalg.inv(T2) @ H @ T1
    return H


def RANSAC_homography(
    match_indices,
    valid_corners,
    n_iterations=10000,
    inlier_thresh=10,
    stop_thresh=0.85,
):
    best_inlier_matches = []

    for _ in range(n_iterations):
        # select four feature pairs at random
        random_indices = np.random.choice(len(match_indices), size=(4,), replace=False)
        random_matches = match_indices[random_indices]
        pairs = _pairs_from_matches(random_matches, valid_corners)
        # skip iteration if selected points are degenerate
        if np.linalg.matrix_rank(_form_A_matrix(pairs)) < 8:
            continue

        # compute homography from the four pairs; skip if SVD fails
        try:
            H = _compute_homography(pairs)
        except np.linalg.LinAlgError:
            continue

        # use H to map all points from image1 to image2, counting inliers
        inlier_matches = []
        for i1, i2 in match_indices:
            x1, y1 = valid_corners[0][i1]
            x2, y2 = valid_corners[1][i2]
            p1 = np.array([x1, y1, 1]).reshape((3, 1))
            p2_est = H @ p1
            p2_est = (p2_est[:2] / max(p2_est[2], 1e-8)).flatten()
            dist = np.linalg.norm(p2_est - np.array([x2, y2]), ord=1)
            if dist < inlier_thresh:
                inlier_matches.append((i1, i2))

        if len(inlier_matches) > len(best_inlier_matches):
            best_inlier_matches = inlier_matches

        if len(inlier_matches) > stop_thresh * len(match_indices):
            print(f"Early stopping RANSAC with {len(inlier_matches)} inliers")
            break

    # recompute homography using all inliers
    inlier_pairs = _pairs_from_matches(best_inlier_matches, valid_corners)
    H_final = _compute_homography(inlier_pairs)
    return H_final, best_inlier_matches


def get_panorama_dimensions(images, H_list):
    # transform each image's corners to panorama frame
    all_corners = []
    for img, H in zip(images, H_list):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        corners_transformed = cv2.perspectiveTransform(corners, H)
        all_corners.append(corners_transformed)

    # combine all corners to find panorama bounds
    all_corners = np.concatenate(all_corners, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # calculate translation to shift panorama to positive coordinates
    H_translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32
    )
    return (x_max - x_min, y_max - y_min), H_translation


def _create_mask(img):
    # create binary mask where image is non-zero
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)

    # compute weights, i.e. distance from the nearest zero pixel (the edge)
    weights = cv2.distanceTransform(src=mask, distanceType=cv2.DIST_L2, maskSize=5)
    weights = weights / np.max(weights)
    return weights


def blend_images(img1, img2):
    # create weight masks
    w1 = _create_mask(img1)
    w2 = _create_mask(img2)

    sum_weights = np.maximum(w1 + w2, 1e-8)

    # fix dimensions and perform weighted average
    w1 = w1[:, :, np.newaxis]
    w2 = w2[:, :, np.newaxis]
    sum_weights = sum_weights[:, :, np.newaxis]
    blended = (img1 * w1 + img2 * w2) / sum_weights
    return blended.astype(np.uint8), (sum_weights * 255).astype(np.uint8)


def main():
    # get path to current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default=f"{parent_dir}/Data/Train/Set1",
        help="Directory of input images",
    )
    ap.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Whether to save output images",
    )
    args = ap.parse_args()
    input_dir = args.dir
    save_output = args.output
    output_dir = f"{parent_dir}/Output"

    """
    Read a set of images for Panorama stitching
    """
    images = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_dir, filename))
            if img is not None:
                if "Train" in input_dir and "Set3" in input_dir:
                    img = cylindrical_warp(img, cylinder_radius=img.shape[1])
                images.append(img)
    print(f"Read {len(images)} images from {input_dir}")

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    raw_corners = [locate_corners(image) for image in images]
    if save_output:
        for i, (image, C_img) in enumerate(zip(images, raw_corners)):
            corners_vis = image.copy()
            # use cv2.circle to draw corners
            ys, xs = np.where(C_img > 0.01 * C_img.max())
            for x, y in zip(xs, ys):
                cv2.circle(corners_vis, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(output_dir, f"corners_{i}.png"), corners_vis)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    This function will get the Nbest corners in the image based on their strength and their distance to another strong corner
    """
    anms_corners = [ANMS(corners) for corners in raw_corners]
    if save_output:
        for i, (image, corners) in enumerate(zip(images, anms_corners)):
            anms_vis = image.copy()
            for x, y in corners:
                cv2.circle(anms_vis, (x, y), 3, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(output_dir, f"anms_{i}.png"), anms_vis)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    fd, valid_corners = zip(
        *[
            encode_feature_points(image, corners)
            for image, corners in zip(images, anms_corners)
        ]
    )
    # plot feature descriptors in 1-D
    if save_output:
        plt.figure(figsize=(8, 4))
        for i, desc in enumerate(fd):
            mean = np.mean(desc, axis=0)
            std = np.std(desc, axis=0)

            x = np.arange(len(mean))
            plt.plot(x, mean, label=f"Image {i}")
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

        plt.xlabel("Descriptor index")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Mean ± Std of Feature Descriptors")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fd.png"))
        plt.close()

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    match_indices = []
    for i in range(len(images) - 1):
        match_indices_i = match_features(fd[i], fd[i + 1])
        match_indices.append(match_indices_i)

        draw_matches(
            images[i],
            images[i + 1],
            match_indices_i,
            (valid_corners[i], valid_corners[i + 1]),
            window_name=f"Feature Matches {i}->{i+1}",
        )
        if save_output:
            img_matches = draw_matches(
                images[i],
                images[i + 1],
                match_indices_i,
                (valid_corners[i], valid_corners[i + 1]),
            )
            cv2.imwrite(
                os.path.join(output_dir, f"matching_{i}_{i+1}.png"), img_matches
            )

    """
    Refine: RANSAC, Estimate Homography
    """
    # list of homographies between consecutive images
    # valid_images keeps track of what pairs of images are usable for the panorama, rejects the ones that aren't
    pairwise_H = []
    valid_images = []
    # iterate through adjacent images, looking at i and comparing it to i+1 for the H_i calculation
    for i in range(len(images) - 1):
        H_i, inliers_i = RANSAC_homography(
            np.array(match_indices[i]), (valid_corners[i], valid_corners[i + 1])
        )

        print(
            f"Found {len(inliers_i)}/{len(match_indices[i])} inliers between images {i} and {i+1}"
        )
        # should make sure that if the number of inliers is too low, don't append that Homogrpahy, there are not enough matching features)
        if float(len(inliers_i) / len(match_indices[i])) < 0.30:
            print(
                f"Images {i} and {i+1} shouldn't go next to each other, SKIPPING this homography"
            )
            pairwise_H.append(None)
            continue
        pairwise_H.append(H_i)
        # only add i to the images but we can know that it goes with i+1
        valid_images.append(i)

        # refine valid_corners using inliers
        inlier_indices_1, inlier_indices_2 = zip(*inliers_i)
        inlier_corners = (
            valid_corners[i][list(inlier_indices_1)],
            valid_corners[i + 1][list(inlier_indices_2)],
        )
        # create matches with sequential indices for the filtered corners
        remapped_inlier_matches = [(i, i) for i in range(len(inlier_corners[0]))]

        draw_matches(
            images[i],
            images[i + 1],
            remapped_inlier_matches,
            (inlier_corners[0], inlier_corners[1]),
            window_name=f"Inlier Matches {i}->{i+1}",
        )
        if save_output:
            img_inlier_matches = draw_matches(
                images[i],
                images[i + 1],
                remapped_inlier_matches,
                (inlier_corners[0], inlier_corners[1]),
            )
            cv2.imwrite(
                os.path.join(output_dir, f"inlier_matching_{i}_{i+1}.png"),
                img_inlier_matches,
            )

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    # while making a panorama, keep track of all of the chains and then only keep the largest one, a valid chain occurs if it contains valid homographies
    best_start = 0
    best_len = 1
    curr_start = 0
    curr_len = 1

    # from this extract the right ones and then separate those images
    for i in range(len(pairwise_H)):
        if pairwise_H[i] is not None:
            curr_len += 1
        else:
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
            curr_start = i + 1
            curr_len = 1
    # check at the end
    if curr_len > best_len:
        best_len = curr_len
        best_start = curr_start

    # keep only the image indices where the homography is consequent
    valid_images_indices = list(range(best_start, best_start + best_len))

    # readjust images to be only the good images, same thing with the homographies, remove the nones and only keep one chain
    images = [images[i] for i in valid_images_indices]
    pairwise_H = [pairwise_H[i] for i in range(best_start, best_start + best_len - 1)]

    # use middle image as reference
    n = len(images)
    ref_idx = n // 2
    # compute homographies to reference frame
    H_to_ref = [np.eye(3, dtype=np.float32) for _ in range(n)]
    # left of reference (i < ref_idx)
    for j in range(ref_idx - 1, -1, -1):
        H_to_ref[j] = H_to_ref[j + 1] @ pairwise_H[j]
    # right of reference (i > ref_idx); use inverse H
    for j in range(ref_idx + 1, n):
        H_to_ref[j] = H_to_ref[j - 1] @ np.linalg.inv(pairwise_H[j - 1])

    # determine panorama size and translations
    pano_size, H_translation = get_panorama_dimensions(images, H_to_ref)

    # transform reference image to panorama frame using H_translation
    panorama = cv2.warpPerspective(
        images[ref_idx],
        H_translation @ H_to_ref[ref_idx],
        pano_size,
        flags=cv2.INTER_LANCZOS4,
    )

    # transform and blend remaining images into the panorama
    for i in range(n):
        if i == ref_idx:
            continue
        warped_i = cv2.warpPerspective(
            images[i], H_translation @ H_to_ref[i], pano_size, flags=cv2.INTER_LANCZOS4
        )
        panorama, vis_weights = blend_images(panorama, warped_i)
        if save_output and i == n - 1:
            cv2.imwrite(
                os.path.join(output_dir, f"weightmask.png"),
                vis_weights,
            )

    cv2.imshow("Panorama", panorama)
    if save_output:
        cv2.imwrite(os.path.join(output_dir, "mypano.png"), panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
