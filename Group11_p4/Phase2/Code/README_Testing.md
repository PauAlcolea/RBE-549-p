# Phase 2 Testing Framework

Comprehensive evaluation framework for DL-based odometry models (visual, inertial, visual-inertial).

## Quick Start

### Test on a Single Sequence
```bash
cd Code
./run_test.sh seq_000041
```

### Test on All Test Sequences (Batch Mode)
```bash
./run_test.sh
```

## Usage

### Command Line

```bash
python Test.py \
    --checkpoint ../Output/Training/checkpoints/VISUAL/best_model.pth \
    --test-data-dir ../Data/Generated/test \
    --sequence-length 5 \
    --lstm-hidden 512 \
    --image-height 240 \
    --image-width 320 \
    -v
```

### Arguments

**Required:**
- `--checkpoint PATH` - Path to trained model checkpoint (.pth file)
- `--test-data-dir PATH` - Path to test data directory (contains test split or is test split)
- `-v` | `-i` | `-vi` - Model type (visual, inertial, or visual-inertial)

**Optional:**
- `--sequence-id ID` - Test specific sequence (e.g., seq_000041). Default: test all sequences
- `--output-dir PATH` - Output directory. Default: `../Output/Testing/{MODEL_TYPE}/`
- `--sequence-length N` - Sequence length used during training (default: 5)
- `--lstm-hidden N` - LSTM hidden size used during training (default: 512)
- `--image-height N` - Image height used during training (default: 240)
- `--image-width N` - Image width used during training (default: 320)
- `--show-plots` - Display plots interactively (default: only save)
- `--device DEVICE` - Device to run on (cuda/cpu/mps). Default: auto-detect

### Using run_test.sh Script

The `run_test.sh` script provides convenient defaults:

```bash
# Test all sequences
./run_test.sh

# Test specific sequence
./run_test.sh seq_000041

# Test inertial model
./run_test.sh -i

# Test visual-inertial model
./run_test.sh -vi

# Custom checkpoint
./run_test.sh --checkpoint /path/to/checkpoint.pth

# Help
./run_test.sh --help
```

## Output Structure

Results are saved to `../Output/Testing/{MODEL_TYPE}/{SEQUENCE_ID}/`:

```
Testing/
└── VISUAL/
    ├── seq_000041/
    │   ├── seq_000041_trajectory.png      # Visualization plots
    │   ├── seq_000041_metrics.json        # Computed metrics
    │   ├── seq_000041_trajectory.csv      # Full trajectory data
    │   ├── seq_000041_tum_gt.txt          # Ground truth in TUM format
    │   └── seq_000041_tum_est.txt         # Estimate in TUM format
    ├── seq_000042/
    │   └── ...
    └── summary.json                        # Aggregate statistics (batch mode)
```

## Output Files

### 1. Trajectory Visualization (`*_trajectory.png`)
Three-panel figure:
- **Left**: Top-down trajectory (X-Z view) with ground truth vs. estimate
- **Middle**: Position error norm over time
- **Right**: ATE histogram with mean/median markers

### 2. Metrics JSON (`*_metrics.json`)
Complete metrics including:
- **ATE (Absolute Trajectory Error)**: RMSE, mean, std, median, min, max
- **Per-axis RMSE**: X, Y, Z components
- **RPE (Relative Pose Error)**: Translation and rotation RMSE/mean/std
- Sequence metadata

Example:
```json
{
  "sequence_id": "seq_000041",
  "model_type": "VISUAL",
  "metrics": {
    "ate_rmse": 0.1234,
    "ate_mean": 0.0987,
    "rmse_x": 0.0654,
    "rmse_y": 0.0321,
    "rmse_z": 0.0123,
    "rpe_trans_rmse": 0.0156,
    "rpe_rot_rmse": 0.0234
  }
}
```

### 3. Trajectory CSV (`*_trajectory.csv`)
Full frame-by-frame data with ground truth and estimate poses:
```csv
frame,gt_tx,gt_ty,gt_tz,gt_qw,gt_qx,gt_qy,gt_qz,est_tx,est_ty,est_tz,est_qw,est_qx,est_qy,est_qz
1,1.0,0.0,1.5,1.0,0.0,0.0,0.0,1.0,0.0,1.5,1.0,0.0,0.0,0.0
...
```

### 4. TUM Format (`*_tum_gt.txt`, `*_tum_est.txt`)
Standard TUM format for EVO compatibility:
```
timestamp tx ty tz qx qy qz qw
1 1.000000 0.000000 1.500000 0.000000 0.000000 0.000000 1.000000
...
```

### 5. Summary JSON (`summary.json`, batch mode only)
Aggregate statistics across all test sequences:
```json
{
  "model_type": "VISUAL",
  "num_sequences": 5,
  "aggregate": {
    "ate_rmse_mean": 0.1234,
    "ate_rmse_std": 0.0567,
    "rpe_trans_rmse_mean": 0.0156
  }
}
```

## Metrics Explained

### Absolute Trajectory Error (ATE)
- Measures global consistency of the trajectory
- After rigid SE3 alignment (rotation + translation, no scale)
- **RMSE**: Root mean square error of position errors
- **Mean/Median**: Central tendency of errors
- **Per-axis RMSE**: X, Y, Z component errors

### Relative Pose Error (RPE)
- Measures local accuracy (frame-to-frame drift)
- **Translation RMSE**: Error in relative position changes
- **Rotation RMSE**: Error in relative orientation changes (radians)

## EVO Integration

The TUM format outputs are compatible with [evo](https://github.com/MichaelGrupp/evo) tools.

### Example EVO Commands

```bash
# Trajectory overlay
evo_traj tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \
    --ref seq_000041_tum_gt.txt \
    --align \
    --plot

# Absolute Pose Error
evo_ape tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \
    -va \
    --plot \
    --save_plot ate.pdf

# Relative Pose Error  
evo_rpe tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \
    -va \
    --delta 1 \
    --plot \
    --save_plot rpe.pdf
```

## Implementation Details

### Trajectory Reconstruction
- **Initial pose**: Both GT and estimate start from same initial pose (GT frame 1)
- **Accumulation**: Relative poses accumulated using quaternion composition
- **Normalization**: Quaternions normalized after each step to prevent drift
- **Sliding windows**: Non-overlapping consecutive windows matching training distribution

### Inference Strategy
- **Window size**: Uses same sequence_length as training
- **Overlap**: Non-overlapping windows (stride = sequence_length - 1)
- **Edge handling**: Last window may be shorter, includes all remaining predictions

### Alignment
- **SE3 rigid alignment**: Rotation + translation only (no scale correction)
- **SVD-based**: Uses Kabsch algorithm for optimal alignment
- **Applied for**: ATE computation and visualization only (RPE uses unaligned)

## Batch Testing

When testing multiple sequences, the framework:
1. Tests each sequence individually
2. Saves results to separate subdirectories
3. Prints summary table with all sequences
4. Saves aggregate statistics to `summary.json`

Example summary output:
```
================================================================================
SUMMARY ACROSS ALL SEQUENCES
================================================================================

Sequence        ATE RMSE     ATE Mean     RPE Trans    RPE Rot (deg)  
--------------------------------------------------------------------------------
seq_000041      0.1234       0.0987       0.0156       0.89           
seq_000042      0.1456       0.1123       0.0178       1.02           
seq_000043      0.1678       0.1345       0.0201       1.15           
--------------------------------------------------------------------------------
Mean            0.1456       0.1152       0.0178       1.02           
Std             0.0222       0.0179       0.0023       0.13           
================================================================================
```

## Model Support

### Currently Implemented
- ✅ **Visual Model**: Full support with image-based inference

### Coming Soon
- ⏳ **Inertial Model**: Pending trained checkpoint
- ⏳ **Visual-Inertial Model**: Pending trained checkpoint

The framework is designed to automatically support all model types once checkpoints are available.

## Troubleshooting

### Checkpoint not found
```bash
ERROR: Checkpoint not found: ../Output/Training/checkpoints/VISUAL/best_model.pth
```
**Solution**: Train a model first using `Train.py` or `run_train.sh`

### Test data not found
```bash
ERROR: Test data directory not found: ../Data/Generated/test
```
**Solution**: Generate test data using the Blender pipeline

### CUDA out of memory
**Solution**: 
- Use CPU: `--device cpu`
- Process sequences individually: `./run_test.sh seq_000041`

### Model architecture mismatch
```bash
ERROR: Error(s) in loading state_dict...
```
**Solution**: Ensure `--sequence-length`, `--lstm-hidden`, `--image-height`, `--image-width` match training configuration

## Comparison with Phase 1

Like Phase 1's EVO-based evaluation, Phase 2 provides:
- ✅ Trajectory overlay plots
- ✅ Error metrics (ATE, RPE)
- ✅ TUM format output
- ✅ Error histograms
- ✅ Multi-plane views (can be added to visualization)

Additional Phase 2 features:
- 🆕 Batch testing with summary statistics
- 🆕 JSON metrics output for programmatic analysis
- 🆕 Per-sequence detailed outputs
- 🆕 Model-agnostic framework (visual/inertial/visual-inertial)
