#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_blender.sh"
IMU_SCRIPT="$SCRIPT_DIR/trajectory_imu.py"

print_usage() {
  cat <<'EOF'
Usage: ./run_dataset_sweep.sh [--resume] [--jobs N] [--imu] [--help]

Generate a dataset by sweeping combinations of:
  - trajectory shape
  - texture image
  - camera height

By default this script calls run_blender.sh once per combination.
With --imu it calls trajectory_imu.py directly (no Blender/rendering).

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

Parallel mode:
  --jobs N (or -j N) runs up to N generation processes concurrently.
  Default is 1 (sequential).
EOF
}

RESUME_MODE=0
JOBS=1
IMU_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_MODE=1
      shift
      ;;
    -j|--jobs)
      if [[ $# -lt 2 ]]; then
        echo "[dataset_sweep] ERROR: --jobs requires a value." >&2
        print_usage
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    --imu)
      IMU_MODE=1
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

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS < 1 )); then
  echo "[dataset_sweep] ERROR: --jobs must be an integer >= 1." >&2
  exit 2
fi

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
  if (( IMU_MODE )); then
    [[ -f "$seq_dir/poses.csv" && -f "$seq_dir/trajectory.txt" ]]
    return
  fi

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

active_pids=()
active_labels=()
failed_jobs=0

wait_for_oldest_job() {
  local pid="${active_pids[0]}"
  local label="${active_labels[0]}"

  if wait "$pid"; then
    echo "[dataset_sweep] Completed: $label"
  else
    echo "[dataset_sweep] ERROR: Failed: $label" >&2
    failed_jobs=$((failed_jobs + 1))
  fi

  active_pids=("${active_pids[@]:1}")
  active_labels=("${active_labels[@]:1}")
}

launch_sequence() {
  local combo_idx="$1"
  local total="$2"
  local seq_id="$3"
  local split="$4"
  local shape="$5"
  local texture="$6"
  local height="$7"
  local seed="$8"

  local label="[$combo_idx/$total] seq=$seq_id split=$split shape=$shape texture=$texture height=$height seed=$seed"
  echo "[dataset_sweep] $label"

  if (( JOBS == 1 )); then
    if (( IMU_MODE )); then
      python3 "$IMU_SCRIPT" \
        --data-root "$DATASET_ROOT" \
        --shape "$shape" \
        --height "$height" \
        --split "$split" \
        --seq-id "$seq_id"
    else
      bash "$RUN_SCRIPT" \
        --shape "$shape" \
        --texture "$texture" \
        --height "$height" \
        --split "$split" \
        --seq-id "$seq_id" \
        --seed "$seed"
    fi
    return
  fi

  if (( IMU_MODE )); then
    python3 "$IMU_SCRIPT" \
      --data-root "$DATASET_ROOT" \
      --shape "$shape" \
      --height "$height" \
      --split "$split" \
      --seq-id "$seq_id" &
  else
    bash "$RUN_SCRIPT" \
      --shape "$shape" \
      --texture "$texture" \
      --height "$height" \
      --split "$split" \
      --seq-id "$seq_id" \
      --seed "$seed" &
  fi
  local pid=$!

  active_pids+=("$pid")
  active_labels+=("$label")
  echo "[dataset_sweep] Launched PID $pid"

  while (( ${#active_pids[@]} >= JOBS )); do
    wait_for_oldest_job
  done
}

total=$(( ${#SHAPES[@]} * ${#TEXTURES[@]} * ${#HEIGHTS[@]} * REPEATS ))
if (( total <= 0 )); then
  echo "[dataset_sweep] ERROR: No combinations to run." >&2
  exit 2
fi

echo "[dataset_sweep] Total sequences to generate: $total"
echo "[dataset_sweep] Parallel jobs: $JOBS"
if (( IMU_MODE )); then
  echo "[dataset_sweep] Mode: IMU trajectory-only (trajectory_imu.py)"
else
  echo "[dataset_sweep] Mode: Blender visual generation (run_blender.sh)"
fi

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

        launch_sequence "$combo_idx" "$total" "$seq_id" "$split" "$shape" "$texture" "$height" "$seed"

        seq_num=$((seq_num + 1))
      done
    done
  done
done

while (( ${#active_pids[@]} > 0 )); do
  wait_for_oldest_job
done

if (( failed_jobs > 0 )); then
  echo "[dataset_sweep] ERROR: $failed_jobs sequence job(s) failed." >&2
  exit 1
fi

if (( IMU_MODE )); then
  echo "[dataset_sweep] Done. IMU trajectory poses written under Data/Generated/<split>/seq_xxxxxx."
else
  echo "[dataset_sweep] Done. Inspect Data/Generated/index.csv for the sequence manifest."
fi
