import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualModel(nn.Module):
    """
    DeepVO-style visual odometry model.

    Architecture: CNN (FlowNet-style) → BiLSTM → FC layers → 7D pose
    Input: Image sequences (batch_size, seq_len, 3, H, W)
    Output: Relative poses (batch_size, seq_len-1, 7) where 7 = [dx, dy, dz, qw, qx, qy, qz]
    """

    def __init__(
        self,
        image_height=360,
        image_width=480,
        lstm_hidden_size=1000,
        lstm_num_layers=2,
        dropout=0.2,
        beta_translation=100.0,
        beta_rotation=1.0,
    ):
        super(VisualModel, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.beta_translation = beta_translation
        self.beta_rotation = beta_rotation

        # CNN feature extractor (FlowNet-style)
        self.cnn = self._build_cnn_layers()

        # Calculate CNN output feature size
        self.cnn_feature_size = self._calculate_cnn_output_size()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Pose regression head
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 128)  # *2 for bidirectional
        self.fc_pose = nn.Linear(128, 7)  # 7D output: [dx, dy, dz, qw, qx, qy, qz]
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

    def _calculate_cnn_output_size(self):
        """Calculate the flattened feature size after CNN."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.image_height, self.image_width)
            dummy_output = self.cnn(dummy_input)
            feature_size = dummy_output.view(1, -1).size(1)
        return feature_size

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'images': (batch_size, seq_len, 3, H, W) image sequences

        Returns:
            poses: (batch_size, seq_len-1, 7) predicted relative poses
        """
        images = batch["images"]  # (B, T, 3, H, W)
        batch_size, seq_len, C, H, W = images.shape

        # Reshape to process all images through CNN: (B*T, 3, H, W)
        images_flat = images.view(batch_size * seq_len, C, H, W)

        # Extract CNN features: (B*T, feature_size)
        cnn_features = self.cnn(images_flat)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)

        # Reshape back to sequences: (B, T, feature_size)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # Pass through BiLSTM: (B, T, hidden_size*2)
        lstm_out, _ = self.lstm(cnn_features)

        # For pose prediction, we need relative poses between consecutive frames
        # Use LSTM output at each timestep to predict pose from t to t+1
        # We'll use lstm_out[:, :-1, :] to predict poses (seq_len-1 poses)
        lstm_features = lstm_out[:, :-1, :]  # (B, T-1, hidden_size*2)

        # Pose regression
        fc1_out = F.relu(self.fc1(lstm_features))  # (B, T-1, 128)
        fc1_out = self.dropout(fc1_out)
        poses = self.fc_pose(fc1_out)  # (B, T-1, 7)

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
        # ignore dz loss
        pred_trans = pred_poses[..., :2]
        pred_quat = pred_poses[..., 3:]
        target_trans = target_poses[..., :2]
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


class InertialModel(nn.Module):
    def __init__(self):
        super(InertialModel, self).__init__()


class VisualInertialModel(nn.Module):
    def __init__(self):
        super(VisualInertialModel, self).__init__()
