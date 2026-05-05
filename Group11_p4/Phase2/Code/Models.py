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


class LSTMVisualModel(nn.Module):
    """
    LSTM-based visual odometry model for temporal sequence processing.
    
    Processes sequences of frames to capture temporal motion patterns.
    Better generalization to different trajectory shapes vs independent frame pairs.
    
    Architecture:
    1. Siamese CNN extracts features from each frame
    2. Features from consecutive pairs are concatenated
    3. LSTM processes sequence of pair features
    4. FC layers predict 6D pose for each pair
    
    Input: Sequence of frames (B, seq_len, 3, H, W)
    Output: Sequence of relative poses (B, seq_len-1, 6)
    """
    
    def __init__(
        self,
        feature_size=512,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.2,
        beta_translation=100.0,
        beta_rotation=1.0,
        use_geodesic_loss=True,
    ):
        super(LSTMVisualModel, self).__init__()
        
        self.feature_size = feature_size
        self.lstm_hidden = lstm_hidden
        self.beta_translation = beta_translation
        self.beta_rotation = beta_rotation
        self.use_geodesic_loss = use_geodesic_loss
        
        # Shared CNN backbone for feature extraction
        self.cnn = self._build_cnn_layers()
        
        # Feature projection
        self.feature_bottleneck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(256, feature_size)
        self.feature_norm = nn.LayerNorm(feature_size)
        
        # Pairwise feature fusion (from two consecutive frames)
        self.pair_fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Pose regression head
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc_pose = nn.Linear(128, 6)
        self.dropout = nn.Dropout(dropout)
    
    def _build_cnn_layers(self):
        """Build FlowNet-style CNN encoder (same as VisualModel)."""
        layers = []
        layers.extend([nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)
    
    def _extract_features(self, image):
        """Extract features from a single image."""
        x = self.cnn(image)
        x = self.feature_bottleneck(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.feature_proj(x)
        x = self.feature_norm(x)
        return x
    
    def forward(self, batch):
        """
        Forward pass through LSTM model.
        
        Args:
            batch: Dictionary containing either:
                - 'image_seq': (B, seq_len, 3, H, W) sequence of frames (training mode)
                OR
                - 'image_t' and 'image_tp1': (B, 3, H, W) frame pairs (inference/test mode)
        
        Returns:
            poses: (B, seq_len-1, 6) or (B, 6) depending on input format
        """
        # Check if we have sequence input or frame pair input
        if "image_seq" in batch:
            # Sequence mode (training)
            image_seq = batch["image_seq"]  # (B, seq_len, 3, H, W)
            B, seq_len = image_seq.shape[:2]
            
            # Extract features for all frames in the sequence
            # Reshape to (B*seq_len, 3, H, W) for batch processing
            image_seq_flat = image_seq.view(B * seq_len, *image_seq.shape[2:])
            features_flat = self._extract_features(image_seq_flat)  # (B*seq_len, feature_size)
            
            # Reshape back to (B, seq_len, feature_size)
            features = features_flat.view(B, seq_len, self.feature_size)
            
            # Create pair features from consecutive frames
            pair_features = []
            for t in range(seq_len - 1):
                feat_t = features[:, t]      # (B, feature_size)
                feat_tp1 = features[:, t+1]  # (B, feature_size)
                
                # Concatenate and fuse
                pair_feat = torch.cat([feat_t, feat_tp1], dim=-1)  # (B, feature_size*2)
                pair_feat = self.pair_fusion(pair_feat)  # (B, feature_size)
                pair_features.append(pair_feat)
            
            # Stack into sequence: (B, seq_len-1, feature_size)
            pair_features = torch.stack(pair_features, dim=1)
            
            # Process with LSTM
            lstm_out, (h_n, c_n) = self.lstm(pair_features)  # (B, seq_len-1, lstm_hidden)
            
            # Predict pose for each timestep
            x = F.relu(self.fc1(lstm_out))  # (B, seq_len-1, 128)
            x = self.dropout(x)
            poses = self.fc_pose(x)  # (B, seq_len-1, 6)
            
            # Normalize quaternions
            poses = self._normalize_quaternions(poses)
            
            return poses
            
        elif "image_t" in batch and "image_tp1" in batch:
            # Frame pair mode (inference/testing compatibility)
            # Process as a sequence of length 2
            image_t = batch["image_t"]      # (B, 3, H, W)
            image_tp1 = batch["image_tp1"]  # (B, 3, H, W)
            
            B = image_t.shape[0]
            
            # Extract features from both frames
            feat_t = self._extract_features(image_t)      # (B, feature_size)
            feat_tp1 = self._extract_features(image_tp1)  # (B, feature_size)
            
            # Create pair feature
            pair_feat = torch.cat([feat_t, feat_tp1], dim=-1)  # (B, feature_size*2)
            pair_feat = self.pair_fusion(pair_feat)  # (B, feature_size)
            
            # Add sequence dimension: (B, 1, feature_size)
            pair_features = pair_feat.unsqueeze(1)
            
            # Process with LSTM (even though it's just one timestep)
            lstm_out, (h_n, c_n) = self.lstm(pair_features)  # (B, 1, lstm_hidden)
            
            # Predict pose
            x = F.relu(self.fc1(lstm_out))  # (B, 1, 128)
            x = self.dropout(x)
            poses = self.fc_pose(x)  # (B, 1, 6)
            
            # Remove sequence dimension and normalize
            poses = poses.squeeze(1)  # (B, 6)
            poses = self._normalize_quaternions(poses)
            
            return poses
        else:
            raise ValueError(
                "Batch must contain either 'image_seq' or both 'image_t' and 'image_tp1'"
            )
    
    def _normalize_quaternions(self, poses):
        """Normalize quaternion part of pose sequence or single pose."""
        # poses: (B, seq_len-1, 6) or (B, 6)
        translation = poses[..., :2]  # Extract translation
        quaternion = poses[..., 2:]   # Extract quaternion
        
        quat_norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-12)
        
        return torch.cat([translation, quaternion / quat_norm], dim=-1)
    
    def compute_loss(self, batch, return_components=False):
        """
        Compute weighted loss for sequence of pose predictions.
        
        Args:
            batch: Dictionary containing:
                - 'image_seq': (B, seq_len, 3, H, W)
                - 'target_poses': (B, seq_len-1, 6) ground truth poses
        
        Returns:
            loss: Scalar loss (or dict if return_components=True)
        """
        pred_poses = self.forward(batch)        # (B, seq_len-1, 6)
        target_poses = batch["target_poses"]    # (B, seq_len-1, 6)
        
        # Flatten sequences for loss computation
        pred_poses_flat = pred_poses.reshape(-1, 6)      # (B*(seq_len-1), 6)
        target_poses_flat = target_poses.reshape(-1, 6)  # (B*(seq_len-1), 6)
        
        # Translation loss (dx, dy)
        pred_trans = pred_poses_flat[..., :2]
        target_trans = target_poses_flat[..., :2]
        loss_translation = F.mse_loss(pred_trans, target_trans)
        rmse_translation = torch.sqrt(loss_translation)
        
        # Rotation loss
        if self.use_geodesic_loss:
            pred_quat = F.normalize(pred_poses_flat[..., 2:], dim=-1)
            target_quat = F.normalize(target_poses_flat[..., 2:], dim=-1)
            
            dot_product = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
            dot_product = torch.clamp(dot_product, 0.0, 1.0 - 1e-7)
            angle_error_rad = 2.0 * torch.acos(dot_product)
            
            loss_rotation = torch.mean(angle_error_rad)
            mean_angle_error_deg = torch.mean(angle_error_rad) * 180.0 / 3.14159265359
        else:
            loss_rotation = F.mse_loss(pred_poses_flat[..., 2:], target_poses_flat[..., 2:])
            
            pred_quat = F.normalize(pred_poses_flat[..., 2:], dim=-1)
            target_quat = F.normalize(target_poses_flat[..., 2:], dim=-1)
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
                - 'images': (B, T, 3, H, W) image sequences
                - 'target_rel_poses': (B, T-1, 7) ground truth relative poses

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Forward pass
        pred_poses = self.forward(batch)  # (B, T-1, 7)
        target_poses = batch["target_rel_poses"]  # (B, T-1, 7)

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
    def __init__(self):
        super(VisualInertialModel, self).__init__()

