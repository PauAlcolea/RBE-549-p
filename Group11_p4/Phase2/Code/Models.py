import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualModel(nn.Module):
    """
    Siamese CNN model for visual odometry using adjacent frame pairs.
    No recurrent component -- each sample is an independent (t, t+1) image pair.

    Architecture: CNN (shared weights) -> bottleneck/pool -> concat -> FC -> 6D pose
    Input: image_t and image_tp1, each (batch_size, 3, H, W)
    Output: relative pose (batch_size, 6) = [dx, dy, qw, qx, qy, qz] (ground plane motion)
    """

    def __init__(
        self,
        feature_size=256,
        hidden_size=512,
        dropout=0.2,
        beta_translation=100.0,  # Weight for translation MSE
        beta_rotation=1.0,       # Weight for rotation angle (radians) - geodesic loss is already well-scaled
        use_geodesic_loss=True,  # Use angle-based loss instead of quaternion MSE
    ):
        super(VisualModel, self).__init__()

        self.beta_translation = beta_translation
        self.beta_rotation = beta_rotation
        self.use_geodesic_loss = use_geodesic_loss

        # Shared CNN backbone
        self.cnn = self._build_cnn_layers()

        # Compress spatial feature maps to a compact vector per image
        self.feature_bottleneck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(256, feature_size)
        self.feature_norm = nn.LayerNorm(feature_size)

        # Regression head: concatenated features from both frames -> 6D pose
        self.fc1 = nn.Linear(feature_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc_pose = nn.Linear(128, 6)  # [dx, dy, qw, qx, qy, qz]
        self.dropout = nn.Dropout(dropout)

    def _build_cnn_layers(self):
        """Build FlowNet-style CNN encoder."""
        layers = []
        layers.extend([nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def _extract_features(self, image):
        x = self.cnn(image)
        x = self.feature_bottleneck(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.feature_proj(x)
        x = self.feature_norm(x)
        return x

    def forward(self, batch):
        image_t = batch["image_t"]      # (B, 3, H, W)
        image_tp1 = batch["image_tp1"]  # (B, 3, H, W)

        feat_t = self._extract_features(image_t)
        feat_tp1 = self._extract_features(image_tp1)

        features = torch.cat([feat_t, feat_tp1], dim=-1)  # (B, feature_size*2)

        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        poses = self.fc_pose(x)  # (B, 6) = [dx, dy, qw, qx, qy, qz]

        poses = self._normalize_quaternions(poses)
        return poses

    def _normalize_quaternions(self, poses):
        """Normalize the quaternion part of the pose vector."""
        translation = poses[..., :2]  # dx, dy only
        quaternion = poses[..., 2:]   # qw, qx, qy, qz
        quat_norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-12)
        return torch.cat([translation, quaternion / quat_norm], dim=-1)

    def compute_loss(self, batch, return_components=False):
        """
        Compute weighted loss for pose prediction.

        Args:
            batch: Dictionary containing:
                - 'image_t': (B, 3, H, W) image at time t
                - 'image_tp1': (B, 3, H, W) image at time t+1
                - 'target_rel_pose': (B, 6) ground truth relative pose [dx, dy, qw, qx, qy, qz]
            return_components: If True, return dict with loss components

        Returns:
            loss: Scalar weighted loss, or dict if return_components=True
        """
        pred_poses = self.forward(batch)        # (B, 6)
        target_poses = batch["target_rel_pose"]  # (B, 6)

        # Translation loss (dx, dy for ground plane motion)
        pred_trans = pred_poses[..., :2]
        target_trans = target_poses[..., :2]
        loss_translation = F.mse_loss(pred_trans, target_trans)
        rmse_translation = torch.sqrt(loss_translation)
        
        # Rotation loss
        if self.use_geodesic_loss:
            # Geodesic distance: angle of rotation between quaternions
            # More meaningful than MSE on quaternion components
            pred_quat = F.normalize(pred_poses[..., 2:], dim=-1)
            target_quat = F.normalize(target_poses[..., 2:], dim=-1)
            
            # cos(angle/2) = |dot(q_pred, q_gt)| (absolute value handles q and -q symmetry)
            dot_product = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
            # Clamp more conservatively to avoid acos gradient instability
            dot_product = torch.clamp(dot_product, 0.0, 1.0 - 1e-7)
            angle_error_rad = 2.0 * torch.acos(dot_product)
            
            # Use mean of angle errors (not squared - more stable)
            loss_rotation = torch.mean(angle_error_rad)
            mean_angle_error_deg = torch.mean(angle_error_rad) * 180.0 / 3.14159265359
        else:
            # Standard quaternion MSE loss
            loss_rotation = F.mse_loss(pred_poses[..., 2:], target_poses[..., 2:])
            
            # Still compute angle error for logging
            pred_quat = F.normalize(pred_poses[..., 2:], dim=-1)
            target_quat = F.normalize(target_poses[..., 2:], dim=-1)
            dot_product = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
            dot_product = torch.clamp(dot_product, 0.0, 1.0 - 1e-7)
            angle_error_rad = 2.0 * torch.acos(dot_product)
            mean_angle_error_deg = torch.mean(angle_error_rad) * 180.0 / 3.14159265359

        total_loss = self.beta_translation * loss_translation + self.beta_rotation * loss_rotation
        
        if return_components:
            return {
                'loss': total_loss,
                'loss_translation': loss_translation,
                'loss_rotation': loss_rotation,
                'rmse_translation_m': rmse_translation,
                'mean_angle_error_deg': mean_angle_error_deg,
                'angle_error_rad': torch.mean(angle_error_rad),
            }
        
        return total_loss


class InertialModel(nn.Module):
    def __init__(self):
        super(InertialModel, self).__init__()


class VisualInertialModel(nn.Module):
    def __init__(self):
        super(VisualInertialModel, self).__init__()

