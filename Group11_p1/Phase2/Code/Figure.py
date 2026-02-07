import cv2
import numpy as np
import torch
import os
import sys
from Network.Network import SupervisedHomographyModel
from Dataset import NORMALIZING_FACTOR
from GenerateData import _get_random_patch, _shift_corners
from Test import RANSAC_homography

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Phase1.Code.Wrapper import (
    locate_corners,
    ANMS,
    encode_feature_points,
    match_features,
)

def learning_pipeline(patches, model, device):
    patch_a, patch_b = patches
    xa = torch.from_numpy(patch_a).unsqueeze(0).unsqueeze(0).float() / 255.0
    xb = torch.from_numpy(patch_b).unsqueeze(0).unsqueeze(0).float() / 255.0
    xa = xa.to(device)
    xb = xb.to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(xa, xb)  # (1,8)
    pred = pred.detach().cpu().numpy().reshape(4, 2).astype(np.float32)

    pred_scaled = pred * float(NORMALIZING_FACTOR)
    return pred_scaled

def largest_square_patch_that_fits(img, center_xy):

    h, w = img.shape[:2]
    cx, cy = map(int, center_xy)

    half = min(cx, cy, w - 1 - cx, h - 1 - cy)
    # ensure at least 2x2
    half = max(half, 1)
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    patch = img[y1:y2, x1:x2]
    return patch, (x1, y1, x2, y2)  # bbox in original image coords


def _traditional_pipeline(images):
    raw_corners = [locate_corners(image) for image in images]
    anms_corners = [ANMS(corners) for corners in raw_corners]
    fds, kps = zip(
        *[
            encode_feature_points(image, corners)
            for image, corners in zip(images, anms_corners)
        ]
    )
    pairwise_matches = []
    idx_matches = match_features(fds[0], fds[1])
    kp1 = kps[0]
    kp2 = kps[1]
    coord_matches = [(kp1[i1], kp2[i2]) for i1, i2 in idx_matches]
    pairwise_matches = coord_matches
    pairwise_H = RANSAC_homography(pairwise_matches, n_iterations=50, inlier_thresh=10.0)
    return pairwise_H


def run_traditional_on_large_then_map_to_image(corners, img_gray, warped_gray,patch_size_net=128):
    cx = int((corners[0, 0] + corners[2, 0]) / 2)
    cy = int((corners[0, 1] + corners[2, 1]) / 2)
    patchA_large, bbox = largest_square_patch_that_fits(img_gray, (cx, cy))
    patchB_large, _    = largest_square_patch_that_fits(warped_gray, (cx, cy))

    x1, y1, _, _ = bbox
    H_large, W_large = patchA_large.shape[:2]

    # un traditional pipeline on large patches
    H_patch = _traditional_pipeline([
        cv2.cvtColor(patchA_large, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(patchB_large, cv2.COLOR_GRAY2BGR),
    ])

    # 128x128 patch inside large patch
    half_net = patch_size_net // 2
    cx, cy = W_large // 2, H_large // 2
    net_x1, net_y1 = cx - half_net, cy - half_net

    src_corners_large = np.array([
        [net_x1,              net_y1],
        [net_x1 + patch_size_net, net_y1],
        [net_x1 + patch_size_net, net_y1 + patch_size_net],
        [net_x1,              net_y1 + patch_size_net],
    ], dtype=np.float32)

    #  warp corners using H_patch
    dst_corners_large = cv2.perspectiveTransform(
        src_corners_large.reshape(1, -1, 2), H_patch
    ).reshape(-1, 2)

    # map from large patch coords to full image coords
    src_full = src_corners_large + np.array([x1, y1], dtype=np.float32)
    dst_full = dst_corners_large + np.array([x1, y1], dtype=np.float32)
    return H_patch, src_full, dst_full


def visualize_known_warp_and_prediction(
    img,
    supervised_model,
    unsupervised_model,
    patch_size=128,
    device=None,
    save_path="./debug_visualization.png",
):
    if img.dtype != np.uint8:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    if img_u8.ndim == 2:
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        img_gray = img_u8
    else:
        img_bgr = img_u8
        img_gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    H, W = img_gray.shape[:2]
    ps = patch_size

    patch, corners = _get_random_patch(img_gray)
    shifted_corners, shifts = _shift_corners(corners, img_gray.shape[:2], max_shift=24)

    # ground truth homography
    H_gt = cv2.getPerspectiveTransform(corners.astype(np.float32), shifted_corners.astype(np.float32))

    # warp full image
    warped_bgr = cv2.warpPerspective(img_bgr, H_gt, (W, H))
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # extract patch A and patch B at the same coordinates in both images
    patch_a = img_gray[corners[0, 1]:corners[2, 1], corners[0, 0]:corners[2, 0]]  # (ps, ps)
    patch_b = warped_gray[corners[0, 1]:corners[2, 1], corners[0, 0]:corners[2, 0]]  # (ps, ps)

    # get model predicted deltas
    pred_supervised = learning_pipeline((patch_a, patch_b), supervised_model, device=device)    
    pred_unsupervised = learning_pipeline((patch_a, patch_b), unsupervised_model, device=device) 

    # put model predictions in full image coordinates
    pred_dst_supervised = corners + pred_supervised
    pred_dst_unsupervised = corners + pred_unsupervised

    # compute EPE (mean L2 corner error) for both models
    err_supervised = float(np.mean(np.linalg.norm(shifted_corners - pred_dst_supervised, axis=1)))
    err_unsupervised = float(np.mean(np.linalg.norm(shifted_corners - pred_dst_unsupervised, axis=1)))

    # also use traditional pipeline (with largest possible patch)
    H_patch, src_full, pred_traditional = run_traditional_on_large_then_map_to_image(corners, img_gray, warped_gray, patch_size_net=patch_size)
    err_traditional = float(np.mean(np.linalg.norm(shifted_corners - pred_traditional, axis=1)))

    # visualize
    left = img_bgr.copy()
    right = warped_bgr.copy()

    def draw_poly(im, pts, color, thick=2):
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(im, [pts_i], True, color, thick, cv2.LINE_AA)

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    CYAN = (255, 255, 0)
    ORANGE = (0, 165, 255)

    # original patch on left image
    draw_poly(left, corners, GREEN, thick=3)

    # map ground truth and model predictions into the warped frame
    gt_on_warped = shifted_corners.astype(np.float32)

    # Predictions are in original image coords; map to warped frame using H_gt^-1
    # (since warpPerspective uses H as output->input, content from p appears at H^-1 * p)
    gt_on_warped = shifted_corners.astype(np.float32)
    supervised_on_warped = pred_dst_supervised.astype(np.float32)
    unsupervised_on_warped = pred_dst_unsupervised.astype(np.float32)
    trad_on_warped = pred_traditional.astype(np.float32)

    # draw warped patches
    draw_poly(right, gt_on_warped, GREEN, thick=10)
    draw_poly(right, supervised_on_warped, RED, thick=3)
    draw_poly(right, unsupervised_on_warped, CYAN, thick=3)
    draw_poly(right, trad_on_warped, ORANGE, thick=3)

    # put both images side by side
    canvas = np.concatenate([left, right], axis=1)
    offset = W  # x offset for right image

    # Error text on right side
    cv2.putText(
        canvas,
        f"GT",
        (offset + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Trad.",
        (offset + 20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        ORANGE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Sup.",
        (offset + 20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        RED,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Unsup.",
        (offset + 20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        CYAN,
        2,
        cv2.LINE_AA,
    )

    if save_path is not None:
        cv2.imwrite(save_path, canvas)

    info = {
        "x0": corners[0, 0],
        "y0": corners[0, 1],
        "src_corners": corners,
        "gt_dst_corners": shifted_corners,
        "pred_dst_corners": pred_dst_supervised,
        "H_gt": H_gt,
        "mean_corner_error_supervised": err_supervised,
        "mean_corner_error_traditional": err_traditional,
        "mean_corner_error_unsupervised": err_unsupervised,
    }
    return canvas, info


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # initialize model
    supervised_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/supervised.pt"
    )
    supervised_model = SupervisedHomographyModel()
    checkpoint = torch.load(supervised_path, map_location=device)
    supervised_model.load_state_dict(checkpoint["model_state_dict"])
    supervised_model.to(device)
    supervised_model.eval()

    unsupervised_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/unsupervised.pt"
    )
    unsupervised_model = SupervisedHomographyModel()
    checkpoint = torch.load(unsupervised_path, map_location=device)
    unsupervised_model.load_state_dict(checkpoint["model_state_dict"])
    unsupervised_model.to(device)
    unsupervised_model.eval()

    # load a test image
    image1 = os.path.dirname(os.path.abspath(__file__)) + "/../Data/Test/Phase2/801.jpg"
    image2 = os.path.dirname(os.path.abspath(__file__)) + "/../Data/Test/Phase2/182.jpg"
    image3 = os.path.dirname(os.path.abspath(__file__)) + "/../Data/Test/Phase2/454.jpg"
    image4 = os.path.dirname(os.path.abspath(__file__)) + "/../Data/Test/Phase2/985.jpg"
    img = cv2.imread(image1)
    canvas, info = visualize_known_warp_and_prediction(
        img,
        supervised_model,
        unsupervised_model,
        patch_size=128,
        device=device,
        save_path="./debug_visualization.png",
    )
    print(f"Mean corner error (traditional): {info['mean_corner_error_traditional']:.2f} pixels")
    print(f"Mean corner error (supervised): {info['mean_corner_error_supervised']:.2f} pixels")
    print(f"Mean corner error (unsupervised): {info['mean_corner_error_unsupervised']:.2f} pixels")
