# Phase 2

## Data Generation

### To generate a single sequence:
```
Usage: ./run_blender.sh [--shape SHAPE] [--height METERS] [--texture NAME]
                        [--split SPLIT] [--seq-id ID] [--seed N]

Options:
        -s, --shape SHAPE   Trajectory shape: square | figure8 | circle | triangle
        --height METERS     Camera flight height in meters (e.g. 1.5)
        -t, --texture NAME  Texture image in textures/ (e.g. playrug, newyork, ispy, leaves, or toys)
        --split SPLIT       Dataset split label (default: train)
        --seq-id ID         Sequence id/name for output folder (e.g. seq_000123)
        -h, --help          Show this help message
```
### To generate a whole dataset (train/val/test):
```
Usage: ./run_dataset_sweep.sh [--imu] [--resume] [--jobs N] [--help]

Generate a dataset by sweeping combinations of:
  - trajectory shape
  - texture image
  - camera height

This script calls run_blender.sh once per combination.

Output:
  Generated data written to Data/Generated.

IMU mode:
  --imu generates trajectory-only sequences with IMU data using trajectory_imu.py

Resume mode:
  --resume will:
    - skip sequences with complete outputs and regenerate incomplete sequences

Parallel mode:
  --jobs N (or -j N) runs up to N Blender processes concurrently.
  Default is 1 (sequential).
```
## Train a Model
```
usage: Train.py [-h] --train_data_dir TRAIN_DATA_DIR --val_data_dir VAL_DATA_DIR [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--image_height IMAGE_HEIGHT]
                [--image_width IMAGE_WIDTH] [--use_augmentation] [--lstm_hidden LSTM_HIDDEN] [--lstm_layers LSTM_LAYERS] [--plot_every PLOT_EVERY] (-v | -i | -vi)

options:
  -h, --help            show this help message and exit
  --train_data_dir TRAIN_DATA_DIR
                        Path to training data directory
  --val_data_dir VAL_DATA_DIR
                        Path to validation data directory
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size for training and validation
  --lr LR               Learning rate for the optimizer
  --image_height IMAGE_HEIGHT
                        Image height for training (original: 480)
  --image_width IMAGE_WIDTH
                        Image width for training (original: 640)
  --use_augmentation    Enable data augmentation (brightness, contrast, noise) for training
  --lstm_hidden LSTM_HIDDEN
                        LSTM hidden size
  --lstm_layers LSTM_LAYERS
                        Number of LSTM layers
  --plot_every PLOT_EVERY
                        Save val GT-vs-predicted 3D trajectory plot every N epochs (<=0 disables)
  -v                    Use Visual Model
  -i                    Use Inertial Model
  -vi                   Use Visual-Inertial Model
```
# Test a Model
```
usage: Test.py [-h] --checkpoint CHECKPOINT --data-dir DATA_DIR [--split {train,val,test}] [--sequence-id SEQUENCE_ID] [--output-dir OUTPUT_DIR]
               [--image-height IMAGE_HEIGHT] [--image-width IMAGE_WIDTH] (-v | -i | -vi) [--show-plots] [--device DEVICE]

Test odometry model

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to model checkpoint (.pth file)
  --data-dir DATA_DIR   Path to data directory (contains train/val/test subdirectories or is a split directory)
  --split {train,val,test}
                        Which data split to evaluate on (default: test)
  --sequence-id SEQUENCE_ID
                        Specific sequence ID to evaluate (e.g., seq_000041). If not provided, evaluates all sequences in split.
  --output-dir OUTPUT_DIR
                        Output directory for results. Default: ../Output/Testing/{MODEL_TYPE}/
  --image-height IMAGE_HEIGHT
                        Image height used during training (default: 240)
  --image-width IMAGE_WIDTH
                        Image width used during training (default: 320)
  -v                    Test Visual Model
  -i                    Test Inertial Model
  -vi                   Test Visual-Inertial Model
  --show-plots          Display plots interactively
  --device DEVICE       Device to run on (cuda/cpu/mps). Default: auto-detect
```