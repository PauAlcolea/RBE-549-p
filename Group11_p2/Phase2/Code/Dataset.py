import os
import random
import numpy as np
import torch


class NeRFDataset:
    """
    dataset class for NeRF training (bypassing PyTorch Dataset/DataLoader)
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def get_sample(self, idx):
        pass

    def get_random_sample(self):
        pass

    def get_batch(self, batch_size):
        pass

    def get_batch_from_index(self, start_idx, batch_size):
        pass
