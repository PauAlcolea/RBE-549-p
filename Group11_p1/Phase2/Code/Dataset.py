import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HomographyDataset(Dataset):
    def __init__(self, data_dir, label_file):
        """
        data_dir: directory containing .npy files
        label_file: path to labels.txt
        """
        self.data_dir = data_dir
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                fname = parts[0]
                shifts = np.array(parts[1:], dtype=np.float32)
                self.samples.append((fname, shifts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, shifts = self.samples[idx]

        # load stacked patches
        P = np.load(os.path.join(self.data_dir, fname))

        if P.shape[2] == 6:
            # two RGB patches
            patch_a = P[:, :, :3]
            patch_b = P[:, :, 3:6]
        elif P.shape[2] == 2:
            # two grayscale patches; split then replicate to 3 channels
            pa = P[:, :, 0]
            pb = P[:, :, 1]
            patch_a = np.stack([pa] * 3, axis=-1)
            patch_b = np.stack([pb] * 3, axis=-1)

        # normalize to [0, 1]
        patch_a = patch_a / 255.0
        patch_b = patch_b / 255.0

        # Random brightness shift [-0.1, 0.1]
        delta_brightness = (np.random.rand() - 0.5) * 0.2
        patch_a = np.clip(patch_a + delta_brightness, 0, 1)
        patch_b = np.clip(patch_b + delta_brightness, 0, 1)

        # convert to tensors
        patch_a = (
            torch.from_numpy(patch_a).permute(2, 0, 1).contiguous().float().clone()
        )
        patch_b = (
            torch.from_numpy(patch_b).permute(2, 0, 1).contiguous().float().clone()
        )
        shifts = torch.from_numpy(shifts).float().clone()

        return patch_a, patch_b, shifts
