#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_blender.sh"

print_usage() {
  cat <<'EOF'
Usage: ./run_dataset_sweep.sh [--help]

Generate a dataset by sweeping combinations of:
  - trajectory shape
  - texture image
  - camera height

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
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[dataset_sweep] ERROR: Unknown argument: $1" >&2
      print_usage
      exit 2
      ;;
  esac
fi

# Sweep configuration. Edit these lists for your dataset recipe.
SHAPES=(square figure8 circle)
TEXTURES=(playrug newyork ispy leaves toys)
HEIGHTS=(1.0 1.5 2.0)
REPEATS=1
START_SEQ_NUM=1
SEED_BASE=1000

# Split percentages (must sum to 100)
TRAIN_PCT=70
VAL_PCT=20
TEST_PCT=10

if (( TRAIN_PCT + VAL_PCT + TEST_PCT != 100 )); then
  echo "[dataset_sweep] ERROR: TRAIN_PCT + VAL_PCT + TEST_PCT must equal 100." >&2
  exit 2
fi

assign_split() {
  local idx="$1"
  local total="$2"
  # Map 1..total onto percentile buckets.
  local pct=$(( (idx * 100 + total - 1) / total ))
  if (( pct <= TRAIN_PCT )); then
    echo "train"
  elif (( pct <= TRAIN_PCT + VAL_PCT )); then
    echo "val"
  else
    echo "test"
  fi
}

total=$(( ${#SHAPES[@]} * ${#TEXTURES[@]} * ${#HEIGHTS[@]} * REPEATS ))
if (( total <= 0 )); then
  echo "[dataset_sweep] ERROR: No combinations to run." >&2
  exit 2
fi

echo "[dataset_sweep] Total sequences to generate: $total"

combo_idx=0
seq_num="$START_SEQ_NUM"
for (( rep=1; rep<=REPEATS; rep++ )); do
  for shape in "${SHAPES[@]}"; do
    for texture in "${TEXTURES[@]}"; do
      for height in "${HEIGHTS[@]}"; do
        combo_idx=$((combo_idx + 1))
        split="$(assign_split "$combo_idx" "$total")"
        seq_id="$(printf "seq_%06d" "$seq_num")"
        seed=$((SEED_BASE + combo_idx - 1))

        echo "[dataset_sweep] [$combo_idx/$total] seq=$seq_id split=$split shape=$shape texture=$texture height=$height seed=$seed"
        bash "$RUN_SCRIPT" \
          --shape "$shape" \
          --texture "$texture" \
          --height "$height" \
          --split "$split" \
          --seq-id "$seq_id" \
          --seed "$seed"

        seq_num=$((seq_num + 1))
      done
    done
  done
done

echo "[dataset_sweep] Done. Inspect Data/Generated/index.csv for the sequence manifest."
