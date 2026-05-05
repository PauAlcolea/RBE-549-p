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
        beta_rotation=1.0,  # Weight for rotation angle (radians) - geodesic loss is already well-scaled
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

        # Conv1: 3 → 64
        layers.extend(
            [
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv2: 64 → 128
        layers.extend(
            [
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv3: 128 → 256
        layers.extend(
            [
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv4: 256 → 512
        layers.extend(
            [
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv5: 512 → 512
        layers.extend(
            [
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv6: 512 → 1024
        layers.extend(
            [
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ]
        )

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
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'images': (batch_size, seq_len, 3, H, W) image sequences

        Returns:
            poses: (batch_size, seq_len-1, 7) predicted relative poses
        """
        image_t = batch["image_t"]  # (B, 3, H, W)
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
        quaternion = poses[..., 2:]  # qw, qx, qy, qz
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
        pred_poses = self.forward(batch)  # (B, 6)
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

        total_loss = (
            self.beta_translation * loss_translation
            + self.beta_rotation * loss_rotation
        )

        if return_components:
            return {
                "loss": total_loss,
                "loss_translation": loss_translation,
                "loss_rotation": loss_rotation,
                "rmse_translation_m": rmse_translation,
                "mean_angle_error_deg": mean_angle_error_deg,
                "angle_error_rad": torch.mean(angle_error_rad),
            }

        return total_loss


class InertialModel(nn.Module):
    def __init__(
        self,
        imu_input_size=6,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.2,
        beta_translation=1.0,
        beta_rotation=1.0,
    ):
        super(InertialModel, self).__init__()

        self.imu_input_size = imu_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.beta_translation = beta_translation
        self.beta_rotation = beta_rotation

        # 2-layer LSTM over IMU sequence
        self.lstm = nn.LSTM(
            input_size=imu_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # 2 linear layers for pose regression
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc_pose = nn.Linear(128, 7)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'imu': (batch_size, seq_len, 6) IMU sequences [ax, ay, az, wx, wy, wz]

        Returns:
            poses: (batch_size, seq_len-1, 7) predicted relative poses
        """
        imu = batch["imu"]  # (B, T, 6)
        batch_size, seq_len, _ = imu.shape

        # Pass through LSTM: (B, T, hidden_size)
        lstm_out, _ = self.lstm(imu)

        # For pose prediction, we need relative poses between consecutive frames
        lstm_features = lstm_out[:, :-1, :]  # (B, T-1, hidden_size)

        # Pose regression
        fc1_out = F.relu(self.fc1(lstm_features))
        fc1_out = self.dropout(fc1_out)
        poses = self.fc_pose(fc1_out)

        # Normalize quaternion component (last 4 values)
        poses = self._normalize_quaternions(poses)

        return poses

    def _normalize_quaternions(self, poses):
        """Normalize the quaternion part of the pose vector."""
        # Split translation and quaternion
        translation = poses[..., :3]  # (B, T-1, 3)
        quaternion = poses[..., 3:]  # (B, T-1, 4)

        # Normalize quaternion
        quat_norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-12)
        quaternion_normalized = quaternion / quat_norm

        # Concatenate back
        return torch.cat([translation, quaternion_normalized], dim=-1)

    def compute_loss(self, batch):
        """
        Compute weighted MSE loss for pose prediction.

        Args:
            batch: Dictionary containing:
                - 'imu': (B, T, 6) IMU sequences
                - 'target_rel_pose': (B, T-1, 7) ground truth relative poses [dx, dy, dz, qw, qx, qy, qz]

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Forward pass
        pred_poses = self.forward(batch)  # (B, T-1, 7)
        target_poses = batch["target_rel_pose"]  # (B, T-1, 7)

        # Split into translation and rotation components
        pred_trans = pred_poses[..., :3]
        pred_quat = pred_poses[..., 3:]
        target_trans = target_poses[..., :3]
        target_quat = target_poses[..., 3:]

        # Compute MSE for each component
        loss_translation = F.mse_loss(pred_trans, target_trans)
        loss_rotation = F.mse_loss(pred_quat, target_quat)

        # Weighted combination
        loss = (
            self.beta_translation * loss_translation
            + self.beta_rotation * loss_rotation
        )

        return loss


class VisualInertialModel(nn.Module):
    """
    Visual-Inertial fusion model for odometry.
    
    Combines CNN features from image pairs with LSTM features from IMU sequences.
    Architecture:
    - Visual branch: Siamese CNN -> features (2*feature_size)
    - Inertial branch: LSTM -> hidden state (lstm_hidden_size)
    - Fusion: Concatenate visual + inertial features
    - Regression: FC layers -> 6D pose [dx, dy, qw, qx, qy, qz]
    """

    def __init__(
        self,
        feature_size=256,
        hidden_size=512,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.2,
        beta_translation=50.0,
        beta_rotation=10.0,
        use_geodesic_loss=True,
        modality_dropout=0.15,
        imu_feature_scale=1.5,
    ):
        super(VisualInertialModel, self).__init__()

        self.feature_size = feature_size
        self.lstm_hidden_size = lstm_hidden_size
        self.beta_translation = beta_translation
        self.beta_rotation = beta_rotation
        self.use_geodesic_loss = use_geodesic_loss
        self.modality_dropout = modality_dropout
        self.imu_feature_scale = imu_feature_scale

        # Visual branch: Shared CNN backbone (reuse from VisualModel)
        self.cnn = self._build_cnn_layers()
        self.feature_bottleneck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(256, feature_size)
        self.feature_norm = nn.LayerNorm(feature_size)

        # Inertial branch: LSTM for IMU sequence processing
        self.lstm = nn.LSTM(
            input_size=6,  # [ax, ay, az, wx, wy, wz]
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Fusion and regression head
        fusion_size = feature_size * 2 + lstm_hidden_size
        self.fc1 = nn.Linear(fusion_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc_pose = nn.Linear(128, 6)  # [dx, dy, qw, qx, qy, qz]
        self.dropout = nn.Dropout(dropout)

    def _build_cnn_layers(self):
        """Build FlowNet-style CNN encoder (same as VisualModel)."""
        layers = []

        # Conv1: 3 → 64
        layers.extend(
            [
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv2: 64 → 128
        layers.extend(
            [
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv3: 128 → 256
        layers.extend(
            [
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv4: 256 → 512
        layers.extend(
            [
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv5: 512 → 512
        layers.extend(
            [
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]
        )

        # Conv6: 512 → 1024
        layers.extend(
            [
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ]
        )

        return nn.Sequential(*layers)

    def _extract_visual_features(self, image):
        """Extract features from a single image."""
        x = self.cnn(image)
        x = self.feature_bottleneck(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.feature_proj(x)
        x = self.feature_norm(x)
        return x

    def _extract_imu_features(self, imu_seq):
        """
        Extract features from IMU sequence using LSTM.
        
        Args:
            imu_seq: (B, N, 6) IMU sequence
        
        Returns:
            features: (B, lstm_hidden_size) LSTM final hidden state
        """
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(imu_seq)
        # Use the last hidden state from the last layer
        features = h_n[-1]  # (B, lstm_hidden_size)
        return features

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'image_t': (B, 3, H, W) image at time t
                - 'image_tp1': (B, 3, H, W) image at time t+1
                - 'imu_seq': (B, N, 6) IMU sequence between frames

        Returns:
            poses: (B, 6) predicted relative poses [dx, dy, qw, qx, qy, qz]
        """
        image_t = batch["image_t"]
        image_tp1 = batch["image_tp1"]
        imu_seq = batch["imu_seq"]

        # Extract visual features
        feat_t = self._extract_visual_features(image_t)
        feat_tp1 = self._extract_visual_features(image_tp1)
        visual_features = torch.cat([feat_t, feat_tp1], dim=-1)  # (B, 2*feature_size)

        # Extract inertial features
        imu_features = self._extract_imu_features(imu_seq)  # (B, lstm_hidden_size)
        imu_features = imu_features * self.imu_feature_scale

        # Apply modality dropout during training to force model to use both modalities
        if self.training and self.modality_dropout > 0:
            batch_size = visual_features.shape[0]
            keep_visual = (
                torch.rand(batch_size, 1, device=visual_features.device)
                >= self.modality_dropout
            ).float()
            keep_imu = (
                torch.rand(batch_size, 1, device=imu_features.device)
                >= self.modality_dropout
            ).float()

            # Avoid dropping both modalities for the same sample.
            both_dropped = (keep_visual == 0.0) & (keep_imu == 0.0)
            keep_imu = torch.where(both_dropped, torch.ones_like(keep_imu), keep_imu)

            visual_features = visual_features * keep_visual
            imu_features = imu_features * keep_imu

        # Fuse features
        fused_features = torch.cat(
            [visual_features, imu_features], dim=-1
        )  # (B, 2*feature_size + lstm_hidden_size)

        # Regression head
        x = F.relu(self.fc1(fused_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        poses = self.fc_pose(x)  # (B, 6)

        # Normalize quaternions
        poses = self._normalize_quaternions(poses)
        return poses

    def _normalize_quaternions(self, poses):
        """Normalize the quaternion part of the pose vector."""
        translation = poses[..., :2]  # dx, dy only
        quaternion = poses[..., 2:]  # qw, qx, qy, qz
        quat_norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-12)
        return torch.cat([translation, quaternion / quat_norm], dim=-1)

    def compute_loss(self, batch, return_components=False):
        """
        Compute weighted loss for pose prediction.

        Args:
            batch: Dictionary containing:
                - 'image_t': (B, 3, H, W)
                - 'image_tp1': (B, 3, H, W)
                - 'imu_seq': (B, N, 6)
                - 'target_rel_pose': (B, 6)
            return_components: If True, return dict with loss components

        Returns:
            loss: Scalar weighted loss, or dict if return_components=True
        """
        pred_poses = self.forward(batch)  # (B, 6)
        target_poses = batch["target_rel_pose"]  # (B, 6)

        # Translation loss (dx, dy for ground plane motion)
        pred_trans = pred_poses[..., :2]
        target_trans = target_poses[..., :2]
        trans_mse = F.mse_loss(pred_trans, target_trans)
        trans_scale = torch.mean(torch.norm(target_trans, dim=-1)).detach()
        trans_scale = torch.clamp(trans_scale, min=1e-3)
        loss_translation = F.smooth_l1_loss(
            pred_trans / trans_scale,
            target_trans / trans_scale,
        )
        rmse_translation = torch.sqrt(trans_mse)

        # Rotation loss
        if self.use_geodesic_loss:
            # Geodesic distance: angle of rotation between quaternions
            pred_quat = F.normalize(pred_poses[..., 2:], dim=-1)
            target_quat = F.normalize(target_poses[..., 2:], dim=-1)

            # cos(angle/2) = |dot(q_pred, q_gt)|
            dot_product = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
            dot_product = torch.clamp(dot_product, 0.0, 1.0 - 1e-7)
            angle_error_rad = 2.0 * torch.acos(dot_product)

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

        total_loss = (
            self.beta_translation * loss_translation
            + self.beta_rotation * loss_rotation
        )

        if return_components:
            return {
                "loss": total_loss,
                "loss_translation": loss_translation,
                "loss_rotation": loss_rotation,
                "rmse_translation_m": rmse_translation,
                "mean_angle_error_deg": mean_angle_error_deg,
                "angle_error_rad": torch.mean(angle_error_rad),
            }

        return total_loss
