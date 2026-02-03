import numpy as np
import cv2
import os
import random
from argparse import ArgumentParser
import sys


def read_images(directory, num_images):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                if len(images) >= num_images:
                    break
    return images


def _get_random_patch(image, patch_size=128, margin=32, max_tries=200):
    h, w = image.shape[:2]
    for i in range(max_tries):
        try:
            x1 = random.randint(margin, w - margin - patch_size)
            y1 = random.randint(margin, h - margin - patch_size)
            break
        except ValueError:
            if i == max_tries - 1:
                return None, None
            continue
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return image[y1:y2, x1:x2], np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def _shift_corners(corners, image_shape, max_shift=32, max_tries=100):
    h, w = image_shape

    for _ in range(max_tries):
        shifts = np.random.randint(-max_shift, max_shift + 1, size=corners.shape)
        translation = np.random.randint(-max_shift, max_shift + 1, size=(1, 2))
        shifts += translation
        C_B = corners + shifts

        if (
            np.all(C_B[:, 0] >= 0)
            and np.all(C_B[:, 0] < w)
            and np.all(C_B[:, 1] >= 0)
            and np.all(C_B[:, 1] < h)
        ):
            return C_B, shifts

    return None, None


def generate_data_point(image_A):
    # cut random square P_A from safe region of image_A
    P_A, C_A = _get_random_patch(image_A)
    if P_A is None or C_A is None:
        print("Failed to get random patch.")
        return None, None
    # calculate random corner shifts
    C_B, shifts = _shift_corners(C_A, image_A.shape[:2])
    if C_B is None:
        print("Failed to shift corners.")
        return None, None
    # warp image_A to get image_B using inverse of corner shifts
    H = cv2.getPerspectiveTransform(C_B.astype(np.float32), C_A.astype(np.float32))
    image_B = cv2.warpPerspective(
        image_A, H, dsize=(image_A.shape[1], image_A.shape[0])
    )
    # cut square P_B from image_B
    P_B = image_B[C_A[0, 1] : C_A[2, 1], C_A[0, 0] : C_A[2, 0]]
    # stack patches depth-wise
    P = np.dstack((P_A, P_B))
    return P, shifts.reshape(
        -1
    )  # data pair is (P_A, P_B) and label is corner shifts H_4Pt


def main():
    ap = ArgumentParser()
    ap.add_argument(
        "-n", "--num_images", type=int, default=100, help="number of images to read"
    )
    ap.add_argument(
        "--all", action="store_true", help="read all images in the folder, ignoring -n"
    )
    ap.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["Train", "Val"],
        default="Train",
        help="type of dataset to generate",
    )
    args = ap.parse_args()
    num_images = args.num_images
    dataset_type = args.type
    if args.all:
        num_images = sys.maxsize

    # set up directories
    data_top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Data"
    input_dir = os.path.join(data_top_dir, dataset_type)
    output_dir = os.path.join(data_top_dir, "Generated", dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # read images
    images = read_images(input_dir, num_images)
    print(len(images), "images read from", input_dir)

    # generate data
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for idx, img in enumerate(images):
            for i in range(2):  # generate 2 data points per image
                P, shifts = generate_data_point(img)
                if P is None or shifts is None:
                    print(
                        f"Skipping image {idx} due to failure in data point generation."
                    )
                    break
                datapoint_name = f"data_{idx:06d}_{i}"
                np.save(os.path.join(output_dir, datapoint_name + ".npy"), P)

                line = datapoint_name + ".npy " + " ".join(str(int(v)) for v in shifts)
                f.write(line + "\n")


if __name__ == "__main__":
    main()
