#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

from Dataset import NeRFDataset

def train():
    pass


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    parser = ArgumentParser()
    # TODO: add arguments
    args = parser.parse_args()
    train()


if __name__ == "__main__":
    main()
