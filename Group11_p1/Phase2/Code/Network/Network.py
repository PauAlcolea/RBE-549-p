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
    def __init__(self):
        super(UnsupervisedHomographyModel, self).__init__()
        self.model = UnsupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)

    @staticmethod
    def compute_loss(pred_delta, patch_a, patch_b, normalizing_factor=1.0):
        B = pred_delta.shape[0]
        h = patch_a.shape[2]
        w = patch_a.shape[3]

        # denormalize predictions to pixel space
        pred_delta_px = pred_delta * normalizing_factor

        # define source corners (corners of patch in pixel coords)
        src_corners = (
            torch.tensor(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                device=pred_delta.device,
                dtype=pred_delta.dtype,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        # direct linear transformation
        dst_corners = src_corners + pred_delta_px.view(B, 4, 2)
        H_pixel = kornia.geometry.get_perspective_transform(src_corners, dst_corners)

        # STN expects normalized values, but H_pixel is in pixel coordinates

        # converts from normalized->pixel coordinates
        denormalizing_mat = (
            torch.tensor(
                [[w / 2.0, 0, w / 2.0], [0, h / 2.0, h / 2.0], [0, 0, 1]],
                dtype=pred_delta.dtype,
                device=pred_delta.device,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        # converts from pixel->normalized coordinates
        normalizing_mat = torch.inverse(denormalizing_mat)
        # STN uses inverse H
        H_pixel_inv = torch.inverse(H_pixel)
        # converts normalized coordinates to pixel coordinates, applies H, then converts back to normalized coordinates
        H_normalized_inv = normalizing_mat @ H_pixel_inv @ denormalizing_mat
        # spatial transformer network
        warped_a = kornia.geometry.transform.warp_perspective(
            patch_a,
            H_normalized_inv,
            dsize=(h, w),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # photometric loss (L1)
        loss = F.l1_loss(warped_a, patch_b)

        return loss, warped_a


class UnsupervisedNet(nn.Module):
    """unsupervised homography network"""

    def __init__(self, OutputSize=8):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(2, 64),
            conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=OutputSize),
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xa, xb):
        x = torch.cat([xa, xb], dim=1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
