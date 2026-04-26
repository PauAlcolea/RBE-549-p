# Phase 2

## Data Generation

### To generate a single sequence:
```
Usage: ./run_blender.sh [--shape SHAPE] [--height METERS] [--texture NAME]
                        [--split SPLIT] [--seq-id ID] [--seed N]

Options:
        -s, --shape SHAPE   Trajectory shape: square | figure8 | circle
        --height METERS     Camera flight height in meters (e.g. 1.5)
        -t, --texture NAME  Texture image in textures/ (e.g. playrug, newyork, ispy, leaves, or toys)
        --split SPLIT       Dataset split label (default: train)
        --seq-id ID         Sequence id/name for output folder (e.g. seq_000123)
        -h, --help          Show this help message

Environment:
        BLENDER_BIN         Blender executable path override
```
### To generate a whole dataset (train/val/test):
```
Usage: ./run_dataset_sweep.sh [--resume] [--help]

Generate a dataset by sweeping combinations of:
  - trajectory shape
  - texture image
  - camera height
Result is dataset of length SHAPES * TEXTURES * HEIGHTS sequences

This script calls run_blender.sh once per combination.

How to configure:
  Edit the sweep variables near the top of run_dataset_sweep.sh:
    SHAPES, TEXTURES, HEIGHTS, REPEATS, START_SEQ_NUM, SEED_BASE,
    TRAIN_PCT, VAL_PCT, TEST_PCT

Sequence naming:
  sequence_id = seq_XXXXXX, starting at START_SEQ_NUM and incrementing by 1.

Split assignment:
  Combinations are assigned train/val/test by percentile buckets in loop order
  using TRAIN_PCT / VAL_PCT / TEST_PCT.

Output:
  Generated data written under Phase2/Data/Generated.

Resume mode:
  --resume will scan expected split/sequence folders and:
    - skip sequences with complete outputs
    - delete and regenerate incomplete sequences
  This is useful if generation stopped mid-run.
```