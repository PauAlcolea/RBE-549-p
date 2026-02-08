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


class BaseHomographyNet(nn.Module):
    """base resnet-like architecture for homography estimation"""
    
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
            ResidualBlock(in_channels=32, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
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
        out - output of the network (predicted corner shifts)
        """
        x = torch.cat([xa, xb], dim=1)
        return self.features(x)


class SupervisedHomographyModel(nn.Module):
    def __init__(self):
        super(SupervisedHomographyModel, self).__init__()
        self.model = SupervisedNet()

    def forward(self, a, b):
        return self.model(a, b)
    
    @staticmethod
    def compute_loss(pred_delta, gt_delta, normalizing_factor=1.0, corner_loss_weight=0.1):
        # smooth L1 on normalized predictions
        loss = F.smooth_l1_loss(pred_delta, gt_delta)
        
        # denormalize to pixel space for corner error
        pred_delta_px = pred_delta * normalizing_factor
        gt_delta_px = gt_delta * normalizing_factor
        
        # corner error: average L2 distance across all corners
        corner_err = (pred_delta_px - gt_delta_px).view(-1, 4, 2).norm(dim=2).mean()
        # add corner error to loss
        loss += corner_loss_weight * corner_err
        
        return loss, corner_err


class SupervisedNet(BaseHomographyNet):
    """supervised homography network; inherits from base architecture"""
    pass


class UnsupervisedHomographyModel(nn.Module):
    def __init__(self, hparams=None):
        super(UnsupervisedHomographyModel, self).__init__()
        self.hparams = hparams
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
        src_corners = torch.tensor(
            [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], 
            device=pred_delta.device, 
            dtype=pred_delta.dtype
        ).unsqueeze(0).repeat(B, 1, 1)
        
        # destination corners = source + predicted shifts
        dst_corners = src_corners + pred_delta_px.view(B, 4, 2)
        H_pixel = kornia.geometry.get_perspective_transform(src_corners, dst_corners)
        
        # Convert to normalized coordinates for warping
        # M maps normalized coords [-1,1] to pixel coords [0, W]
        M = torch.tensor([
            [w / 2.0, 0, w / 2.0],
            [0, h / 2.0, h / 2.0],
            [0, 0, 1]
        ], dtype=pred_delta.dtype, device=pred_delta.device).unsqueeze(0).repeat(B, 1, 1)
        
        M_inv = torch.inverse(M)
        
        # H_normalized = M^-1 @ H_pixel @ M
        # But for warping we need the inverse: H_norm_inv = M^-1 @ H_inv @ M
        H_pixel_inv = torch.inverse(H_pixel)
        H_normalized_inv = M_inv @ H_pixel_inv @ M
        
        # warp patch_a using normalized homography
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


class UnsupervisedNet(BaseHomographyNet):
    """unsupervised homography network; inherits from base architecture"""
    pass
