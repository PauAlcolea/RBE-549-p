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
