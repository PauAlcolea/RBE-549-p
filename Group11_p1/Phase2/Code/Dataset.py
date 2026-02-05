import os
import random
import numpy as np
import torch

NORMALIZING_FACTOR = 64 # max shift + max translation is 64 pixels

class HomographyDataset:
    """
    dataset class to bypass torch.utils.data.Dataset
    """

    def __init__(self, data_dir, label_file):
        """
        data_dir: dir containing the generated .npy files
        label_file: path to the label text file
        """

        self.data_dir = data_dir
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    # Expect: filename + 8 shift values
                    continue
                fname = parts[0]
                shifts = np.array(parts[1:], dtype=np.float32)
                self.samples.append((fname, shifts))

    def __len__(self):
        return len(self.samples)

    def _augment_data(self, patch_a, patch_b, shifts):
        # random brightness shift [-0.1, 0.1]
        delta_brightness = (np.random.rand() - 0.5) * 0.2
        patch_a = np.clip(patch_a + delta_brightness, 0, 1)
        patch_b = np.clip(patch_b + delta_brightness, 0, 1)

        # randomly add noise
        if np.random.rand() < 0.5:
            noise_std = 0.02
            noise = np.random.randn(*patch_a.shape) * noise_std
            patch_a = np.clip(patch_a + noise, 0, 1)
            patch_b = np.clip(patch_b + noise, 0, 1)

        # convert to tensors
        patch_a = (
            torch.from_numpy(patch_a).permute(2, 0, 1).contiguous().float().clone()
        )
        patch_b = (
            torch.from_numpy(patch_b).permute(2, 0, 1).contiguous().float().clone()
        )

        # normalize labels
        shifts = shifts / NORMALIZING_FACTOR
        return patch_a, patch_b, shifts

    def _load_sample(self, fname, shifts):
        # load single data sample

        # load stacked patches
        P = np.load(os.path.join(self.data_dir, fname))

        if P.shape[2] == 6:
            # two RGB patches
            patch_a = P[:, :, :3]
            patch_b = P[:, :, 3:6]
        elif P.shape[2] == 2:
            # two grayscale patches
            pa = P[:, :, 0]
            pb = P[:, :, 1]
            patch_a = np.stack([pa], axis=-1)
            patch_b = np.stack([pb], axis=-1)

        # normalize to [0, 1]
        patch_a = patch_a / 255.0
        patch_b = patch_b / 255.0

        # data augmentation
        patch_a, patch_b, shifts = self._augment_data(patch_a, patch_b, shifts)

        shifts = torch.from_numpy(shifts).float().clone()

        return patch_a, patch_b, shifts

    def get_sample(self, idx):
        # get single sample by index
        fname, shifts = self.samples[idx]
        return self._load_sample(fname, shifts)

    def get_random_sample(self):
        # get single random sample
        idx = random.randint(0, len(self.samples) - 1)
        return self.get_sample(idx)

    def get_batch(self, batch_size):
        # return a random mini-batch of given size
        patch_a_list = []
        patch_b_list = []
        shifts_list = []

        for _ in range(batch_size):
            patch_a, patch_b, shifts = self.get_random_sample()
            patch_a_list.append(patch_a)
            patch_b_list.append(patch_b)
            shifts_list.append(shifts)

        patch_a_batch = torch.stack(patch_a_list)
        patch_b_batch = torch.stack(patch_b_list)
        shifts_batch = torch.stack(shifts_list)

        return patch_a_batch, patch_b_batch, shifts_batch

    def get_batch_from_index(self, start_idx, batch_size):
        # return a sequential mini-batch starting at start_idx

        end_idx = min(start_idx + batch_size, len(self.samples))

        patch_a_list = []
        patch_b_list = []
        shifts_list = []

        for idx in range(start_idx, end_idx):
            patch_a, patch_b, shifts = self.get_sample(idx)
            patch_a_list.append(patch_a)
            patch_b_list.append(patch_b)
            shifts_list.append(shifts)

        patch_a_batch = torch.stack(patch_a_list)
        patch_b_batch = torch.stack(patch_b_list)
        shifts_batch = torch.stack(shifts_list)

        return patch_a_batch, patch_b_batch, shifts_batch


def GenerateBatch(dataset, mini_batch_size):
    # wrapper to match starter code signature
    return dataset.get_batch(mini_batch_size)
