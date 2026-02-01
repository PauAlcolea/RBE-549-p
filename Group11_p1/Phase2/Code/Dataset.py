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

        # handle both (H, W, 2) and (2, H, W)
        if P.shape[0] == 2:
            patch_a = P[0]
            patch_b = P[1]
        else:
            patch_a = P[:, :, 0]
            patch_b = P[:, :, 1]

        # convert to tensors
        patch_a = torch.from_numpy(patch_a).float().unsqueeze(0)
        patch_b = torch.from_numpy(patch_b).float().unsqueeze(0)
        shifts = torch.from_numpy(shifts).float()

        return patch_a, patch_b, shifts
