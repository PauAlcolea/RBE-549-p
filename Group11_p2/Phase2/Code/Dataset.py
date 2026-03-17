import os
import random
import numpy as np
import torch
import imageio
import json
from pathlib import Path
import cv2


def _as_tensor(data):
    return torch.from_numpy(np.stack(data)).float().clone()


class NeRFDataset:
    """
    dataset class for NeRF training (bypassing PyTorch Dataset/DataLoader)
    """

    def __init__(self, data_dir, batch_size, device="cuda", downscale=1):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.downscale = downscale

        self.json_data = self._load_json()
        self.images, self.poses = self._load_images_and_poses()

        self.h = self.images[0].shape[0]
        self.w = self.images[0].shape[1]

        self.K = self._compute_intrinsics()
        self.ray_directions = self._get_camera_ray_directions()

        # precompute and cache all world-space rays and RGB values up front
        # avoids recomputing R @ ray_directions on every get_sample() / get_batch() call
        self.all_ray_origins, self.all_ray_directions = self._precompute_all_rays()
        self.all_rgb = self.images.reshape(-1, 3)  # flatten the images into [num_images * H * W, 3]

    def _load_json(self):
        json_path = self.data_dir.parent / f"transforms_{self.data_dir.stem}.json"
        with open(json_path, "r") as f:
            return json.load(f)

    def _load_images_and_poses(self):
        images = []
        poses = []
        for frame in self.json_data["frames"]:
            file_path = frame["file_path"]
            img_path = self.data_dir / file_path
            img = imageio.imread(img_path) / 255.0
            if img.shape[-1] == 4:
                img = img[..., :3]

            if self.downscale != 1:
                h, w = img.shape[:2]
                img = cv2.resize(
                    img,
                    (w // self.downscale, h // self.downscale),
                    interpolation=cv2.INTER_AREA,
                )

            images.append(img)
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

        return _as_tensor(images), _as_tensor(poses)

    def _compute_intrinsics(self):
        FOV_x = self.json_data["camera_angle_x"]
        f = 0.5 * self.w / np.tan(0.5 * FOV_x)
        K = np.array(
            [[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]], dtype=np.float32
        )
        return _as_tensor(K)

    def _get_camera_ray_directions(self):
        """
        for each pixel compute ray direction from camera center through that pixel
        """
        # create grid of homogeneous pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(self.h), torch.arange(self.w), indexing="ij"
        )
        pixel_coords = torch.stack([j, i, torch.ones_like(i)], dim=-1).float()
        # compute ray directions in camera space
        ray_directions = pixel_coords @ torch.linalg.inv(self.K).T
        ray_directions[..., 1:] *= -1  # match NeRF convention
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        return ray_directions

    def _get_rays_for_image(self, idx):
        """
        transform camera space ray directions to world pose of given image
        """
        pose = self.poses[idx]
        R, t = pose[:3, :3], pose[:3, 3]
        ray_directions_world = self.ray_directions @ R.T
        ray_origins_world = t.view(1, 1, 3).expand_as(ray_directions_world)
        return ray_origins_world.reshape(-1, 3), ray_directions_world.reshape(-1, 3)

    def _precompute_all_rays(self):
        """
        precompute world-space ray origins and directions for every pixel in every image
        called once at init — eliminates the repeated R @ ray_directions matmul in get_sample()
        returns tensors of shape [num_images * H * W, 3]
        """
        all_origins = []
        all_directions = []
        for idx in range(len(self.images)):
            origins, directions = self._get_rays_for_image(idx)
            all_origins.append(origins)
            all_directions.append(directions)
        return torch.cat(all_origins, dim=0), torch.cat(all_directions, dim=0)

    def __len__(self):
        # total number of rays in all images
        return len(self.images) * self.h * self.w

    def _get_image(self, idx):
        rgb = self.images[idx]
        return rgb.reshape(-1, 3)

    def get_image_rays(self, idx):
        """
        get all rays and RGB values for a given image
        """
        ray_origins, ray_directions = self._get_rays_for_image(idx)
        rgb = self._get_image(idx)
        return ray_origins, ray_directions, rgb, self.h, self.w

    def get_sample(self, idx):
        """
        get a single ray sample (origin, direction) and its corresponding RGB color
        uses precomputed tensors
        """
        return (
            self.all_ray_origins[idx],
            self.all_ray_directions[idx],
            self.all_rgb[idx],
        )

    def get_random_sample(self):
        idx = random.randint(0, len(self) - 1)
        return self.get_sample(idx)

    def get_batch(self):
        """
        sample a random batch of rays with vectorised tensor indexing
        """
        idxs = torch.randint(0, len(self), (self.batch_size,))
        return (
            self.all_ray_origins[idxs].to(self.device),
            self.all_ray_directions[idxs].to(self.device),
            self.all_rgb[idxs].to(self.device),
        )

    def get_batch_from_index(self, start_idx):
        end_idx = min(start_idx + self.batch_size, len(self))
        batch = []
        for idx in range(start_idx, end_idx):
            batch.append(self.get_sample(idx))
        return batch
