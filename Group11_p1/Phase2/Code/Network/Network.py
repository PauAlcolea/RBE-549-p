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

# Don't generate pyc codes
sys.dont_write_bytecode = True


class SupervisedHomographyModel(nn.Module):
    def __init__(self):
        super(SupervisedHomographyModel, self).__init__()
        self.model = SupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)


# resnet-like architecture
class SupervisedNet(nn.Module):
    def __init__(self, OutputSize=8):
        """
        Inputs:
        OutputSize - Size of the Output
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            self.ResidualBlock(in_channels=32, out_channels=64, stride=1),
            self.ResidualBlock(in_channels=64, out_channels=64, stride=1),
        )
        self.stage2 = nn.Sequential(
            self.ResidualBlock(in_channels=64, out_channels=128, stride=2),
            self.ResidualBlock(in_channels=128, out_channels=128, stride=1),
        )
        self.stage3 = nn.Sequential(
            self.ResidualBlock(in_channels=128, out_channels=128, stride=1),
            self.ResidualBlock(in_channels=128, out_channels=128, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=OutputSize),
        )
        self.features = nn.Sequential(
            self.conv1,
            self.stage1,
            self.stage2,
            self.stage3,
            self.head,
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        x = torch.cat([xa, xb], dim=1)
        return self.features(x)

    class ResidualBlock(nn.Module):
        """two conv layers with a skip connection"""

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.downsample = (
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                )
                if (stride != 1 or in_channels != out_channels)
                else None
            )

        def forward(self, x):
            residual = x
            out = self.conv2(self.conv1(x))
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = F.relu(out, inplace=True)
            return out


class UnsupervisedHomographyModel(nn.Module):
    def __init__(self, hparams):
        super(UnsupervisedHomographyModel, self).__init__()
        self.hparams = hparams
        self.model = UnsupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)

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
