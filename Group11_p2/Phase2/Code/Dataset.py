import os
import random
import numpy as np
import torch
import imageio
import json
from pathlib import Path


class NeRFDataset:
    """
    dataset class for NeRF training (bypassing PyTorch Dataset/DataLoader)
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.json_data = self._load_json()
        self.images = self._load_images()
        self.h = self.images[0].shape[0]
        self.w = self.images[0].shape[1]
        self.K = self._compute_intrinsics()
        self.poses = self._get_poses()
        self.ray_directions = self._get_camera_ray_directions()

    def _load_json(self):
        json_path = self.data_dir.parent / f"transforms_{self.data_dir.stem}.json"
        with open(json_path, "r") as f:
            return json.load(f)

    def _load_images(self):
        images = []
        for name in os.listdir(self.data_dir):
            img = imageio.imread(os.path.join(self.data_dir, name)) / 255.0
            images.append(img)
        return images

    def _compute_intrinsics(self):
        FOV_x = self.json_data["camera_angle_x"]
        f = 0.5 * self.w / np.tan(0.5 * FOV_x)
        K = np.array(
            [[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]], dtype=np.float32
        )
        return K

    def _get_poses(self):
        poses = []
        for frame in self.json_data["frames"]:
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
        return poses

    def _get_camera_ray_directions(self):
        """
        for each pixel compute ray direction from camera center through that pixel
        """
        # create grid of homogeneous pixel coordinates
        i, j = np.meshgrid(np.arange(self.w), np.arange(self.h), indexing="xy")
        pixel_coords = np.stack([i, j, np.ones_like(i)], axis=-1)
        # compute ray directions in camera space
        ray_directions = pixel_coords @ np.linalg.inv(self.K).T
        ray_directions[..., 1:] *= -1  # match NeRF convention
        ray_directions = ray_directions / np.linalg.norm(
            ray_directions, axis=-1, keepdims=True
        )
        return ray_directions

    def _get_rays_for_image(self, idx):
        """
        transform camera space ray directions to world pose of given image
        """
        pose = self.poses[idx]
        R, t = pose[:3, :3], pose[:3, 3]
        ray_directions_world = self.ray_directions @ R.T
        ray_origins_world = np.broadcast_to(t, ray_directions_world.shape)
        return ray_origins_world.reshape(-1, 3), ray_directions_world.reshape(-1, 3)

    def __len__(self):
        # total number of rays in all images
        return len(self.images) * self.h * self.w

    def _get_image(self, idx):
        rgb = self.images[idx]
        return rgb.reshape(-1, 3)

    def get_sample(self, idx):
        """
        get a single ray sample (origin, direction) and its corresponding RGB color
        """
        img_idx = idx // (self.h * self.w)
        pixel_idx = idx % (self.h * self.w)
        rgb = self._get_image(img_idx)
        ray_origins, ray_directions = self._get_rays_for_image(img_idx)
        return ray_origins[pixel_idx], ray_directions[pixel_idx], rgb[pixel_idx]

    def get_random_sample(self):
        idx = random.randint(0, len(self.images) - 1)
        return self.get_sample(idx)

    def get_batch(self, batch_size):
        for _ in range(batch_size):
            yield self.get_random_sample()

    def get_batch_from_index(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self.images) * self.h * self.w)
        batch = []
        for idx in range(start_idx, end_idx):
            batch.append(self.get_sample(idx))
        return batch
