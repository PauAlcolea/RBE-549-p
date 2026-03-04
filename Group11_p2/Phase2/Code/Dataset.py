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
        self.K = self._compute_intrinsics()

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
        h, w = self.images[0].shape[:2]
        f = 0.5 * w / np.tan(0.5 * FOV_x)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
        return K

    def __len__(self):
        return len(self.images)

    def get_sample(self, idx):
        pass

    def get_random_sample(self):
        pass

    def get_batch(self, batch_size):
        pass

    def get_batch_from_index(self, start_idx, batch_size):
        pass
