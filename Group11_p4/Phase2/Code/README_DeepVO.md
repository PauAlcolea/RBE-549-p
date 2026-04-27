# DeepVO-Style Visual Odometry Implementation

## Overview
A deep learning model for visual odometry based on the DeepVO architecture. The model processes sequences of images through a CNN+BiLSTM network to regress relative camera poses.

## Architecture

**Input**: Image sequences `(batch_size, seq_len, 3, H, W)`  
**Output**: Relative poses `(batch_size, seq_len-1, 7)` where each pose is `[dx, dy, dz, qw, qx, qy, qz]`

### Components

1. **CNN Feature Extractor** (FlowNet-style)
   - 6 convolutional layers with batch normalization
   - Channel progression: 3 → 64 → 128 → 256 → 512 → 512 → 1024
   - Outputs spatial features that are flattened per timestep

2. **Bidirectional LSTM**
   - 2 layers (configurable)
   - Hidden size: 1000 per direction (configurable)
   - Processes temporal sequence of CNN features

3. **Pose Regression Head**
   - FC layers: LSTM output → 128 → 7
   - Outputs 7D pose with quaternion normalization

4. **Weighted MSE Loss**
   - Separate weights for translation (β_t=100) and rotation (β_r=1)
   - `Loss = β_t * MSE(translation) + β_r * MSE(quaternion)`

## Model Parameters

```python
VisualModel(
    image_height=192,        # Input image height
    image_width=384,         # Input image width
    lstm_hidden_size=1000,   # LSTM hidden units per direction
    lstm_num_layers=2,       # Number of LSTM layers
    dropout=0.2,             # Dropout rate
    beta_translation=100.0,  # Translation loss weight
    beta_rotation=1.0,       # Rotation loss weight
)
```

Total parameters: ~93M (with default settings)

## Dataset

The `VisualDataset` class supports two modes:

### Sequences Mode (for DeepVO)
```python
dataset = VisualDataset(
    data_dir='path/to/Generated',
    mode='sequences',
    sequence_length=10,      # Number of frames per sequence
    sequence_stride=1,       # Stride for sliding window
)
```

Returns:
```python
{
    'images': (seq_len, 3, H, W),           # Image sequence
    'target_rel_poses': (seq_len-1, 7),     # Relative poses
    'sequence_id': str,
    'frame_start': int,
    'frame_end': int,
}
```

### Pairs Mode (legacy)
```python
dataset = VisualDataset(
    data_dir='path/to/Generated',
    mode='pairs',
)
```

Returns individual frame pairs with single relative pose.

## Training

### Quick Start

```bash
# Make script executable
chmod +x train_visual_odometry.sh

# Run training
./train_visual_odometry.sh
```

### Manual Training

```bash
python Train.py \
    --train_data_dir ../Data/Generated/train \
    --val_data_dir ../Data/Generated/val \
    --num_epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --sequence_length 10 \
    --lstm_hidden 1000 \
    --num_workers 4 \
    -v
```

### Key Parameters

- `--sequence_length`: Number of frames per training sample (default: 10)
  - Longer sequences: Better temporal context, more memory
  - Shorter sequences: Faster training, less context
  
- `--lstm_hidden`: Hidden size per LSTM direction (default: 1000)
  - Larger values: More model capacity, slower training
  - Smaller values: Faster, may underfit
  
- `--batch_size`: Reduce if you run out of memory (default: 8)
  - Sequences use ~10x more memory than pairs

- `--lr`: Learning rate (default: 0.0001)
  - Start conservative, can increase if training is stable

## Testing

Run unit tests to verify the implementation:

```bash
python test_model.py
```

This tests:
- Forward pass with various sequence lengths
- Loss computation and backward pass
- Quaternion normalization
- Gradient flow

## Outputs

Training outputs are saved to:
- **Logs**: `../Output/Training/logs/VISUAL/`
  - View with TensorBoard: `tensorboard --logdir=../Output/Training/logs/VISUAL/`
  
- **Checkpoints**: `../Output/Training/checkpoints/VISUAL/`
  - Models saved periodically during training

## Next Steps

1. **Data Augmentation**: Add to dataset transform for better generalization
   ```python
   transform = transforms.Compose([
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       # Note: spatial transforms need pose adjustment
   ])
   ```

2. **Trajectory Visualization**: Implement plotting in Train.py (TODO marked)
   - Plot predicted vs. ground truth trajectories
   - Log to TensorBoard for visual validation

3. **Hyperparameter Tuning**:
   - Experiment with sequence_length (5-20)
   - Try different lstm_hidden values (512-2000)
   - Adjust loss weights (beta_translation/beta_rotation)

4. **Advanced Features**:
   - Implement checkpointing/resuming
   - Add learning rate scheduling
   - Consider geodesic loss for rotation if MSE insufficient

## Architecture Diagram

```
Input Images (B, T, 3, H, W)
    ↓
CNN Feature Extractor
    ├─ Conv1: 3→64  (stride 2)
    ├─ Conv2: 64→128 (stride 2)
    ├─ Conv3: 128→256 (stride 2)
    ├─ Conv4: 256→512 (stride 2)
    ├─ Conv5: 512→512 (stride 2)
    └─ Conv6: 512→1024 (stride 2)
    ↓
Flatten: (B, T, feature_size)
    ↓
BiLSTM (2 layers)
    ↓
LSTM Features: (B, T, hidden*2)
    ↓
Take [:, :-1, :] for pose prediction
    ↓
FC Layers: hidden*2 → 128 → 7
    ↓
Quaternion Normalization
    ↓
Output Poses (B, T-1, 7)
```

## File Structure

```
Code/
├── Models.py              # VisualModel implementation
├── Datasets.py            # VisualDataset with sequences support
├── Train.py              # Training loop and pipeline
├── test_model.py         # Unit tests
├── train_visual_odometry.sh  # Example training script
└── README_DeepVO.md      # This file
```

## References

- DeepVO: "DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks" (Wang et al., 2017)
- FlowNet: CNN architecture inspiration for optical flow
