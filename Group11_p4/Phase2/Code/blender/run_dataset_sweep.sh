#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_blender.sh"

print_usage() {
  cat <<'EOF'
Usage: ./run_dataset_sweep.sh [--resume] [--help]

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

Resume mode:
  --resume will scan expected split/sequence folders and:
    - skip sequences with complete outputs
    - delete and regenerate incomplete sequences
  This is useful if generation stopped mid-run.
EOF
}

RESUME_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE=1
      shift
      ;;
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
done

# Sweep configuration. Edit these lists for your dataset recipe.
SHAPES=(square figure8 circle)
TEXTURES=(playrug newyork ispy leaves toys)
HEIGHTS=(1.0 1.5 2.0)
REPEATS=1
START_SEQ_NUM=1
SEED_BASE=1000

# Expected location for generated outputs (matches generate.py OUTPUT_DIR).
DATASET_ROOT="$SCRIPT_DIR/../../Data/Generated"

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

is_sequence_complete() {
  local seq_dir="$1"
  local metadata_path="$seq_dir/metadata.json"
  local frames_dir="$seq_dir/frames"

  if [[ ! -f "$metadata_path" || ! -d "$frames_dir" ]]; then
    return 1
  fi

  local expected_frames
  expected_frames="$(python3 - "$metadata_path" <<'PY' 2>/dev/null || true
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get("num_frames")
print(int(value) if value is not None else "")
PY
)"

  if [[ -z "$expected_frames" ]]; then
    return 1
  fi

  local rendered_frames
  rendered_frames="$(find "$frames_dir" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d '[:space:]')"

  [[ "$rendered_frames" == "$expected_frames" ]]
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

        seq_dir="$DATASET_ROOT/$split/$seq_id"
        if (( RESUME_MODE )) && [[ -d "$seq_dir" ]]; then
          if is_sequence_complete "$seq_dir"; then
            echo "[dataset_sweep] [$combo_idx/$total] seq=$seq_id already complete; skipping."
            seq_num=$((seq_num + 1))
            continue
          fi

          echo "[dataset_sweep] [$combo_idx/$total] seq=$seq_id exists but is incomplete; regenerating."
          rm -rf "$seq_dir"
        fi

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
