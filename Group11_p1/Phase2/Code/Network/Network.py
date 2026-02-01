"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


class SupervisedHomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SupervisedHomographyModel, self).__init__()
        self.hparams = hparams
        self.model = SupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = F.smooth_l1_loss(delta, gt)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = F.smooth_l1_loss(delta, gt)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class SupervisedNet(nn.Module):
    def __init__(self, OutputSize=8):
        """
        Inputs:
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 16 * 16, out_features=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=OutputSize),
        )
        self.features = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
        )

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        x = torch.cat([xa, xb], dim=1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class UnsupervisedHomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(UnsupervisedHomographyModel, self).__init__()
        self.hparams = hparams
        self.model = UnsupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = self.LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = self.LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def LossFn(delta, patch_a, patch_b, corners):
        B = delta.shape[0]
        pred_corners = corners + delta.view(B, 4, 2)
        H = kornia.geometry.get_perspective_transform(
            corners.float(), pred_corners.float()
        )
        warped_a = kornia.geometry.warp_perspective(
            patch_a,
            H,
            dsize=(patch_b.shape[-2], patch_b.shape[-1]),
            padding_mode="border",
        )
        loss = F.l1_loss(warped_a, patch_b)
        return loss


class UnsupervisedNet(nn.Module):

    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        ...
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        pass
