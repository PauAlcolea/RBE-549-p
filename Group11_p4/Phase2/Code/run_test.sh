#!/bin/bash

# =========================================================
# Test DL-based Odometry Models
# =========================================================
# Usage examples:
#   ./run_test.sh                    # Evaluate visual model on all test sequences
#   ./run_test.sh --split train      # Evaluate on training set
#   ./run_test.sh seq_000041         # Evaluate on specific sequence
#   ./run_test.sh --show-plots       # Display plots interactively
# =========================================================

# Default configuration
CHECKPOINT="../Output/Training/checkpoints/VISUAL/best_model.pth"
DATA_DIR="../Data/Generated"
SPLIT="test"
IMAGE_HEIGHT=240
IMAGE_WIDTH=320
MODEL_TYPE="-v"  # Visual model

# Parse command line arguments
SEQUENCE_ID=""
SHOW_PLOTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --show-plots)
            SHOW_PLOTS="--show-plots"
            shift
            ;;
        -i|--inertial)
            MODEL_TYPE="-i"
            CHECKPOINT="../Output/Training/checkpoints/INERTIAL/best_model.pth"
            shift
            ;;
        -vi|--visual-inertial)
            MODEL_TYPE="-vi"
            CHECKPOINT="../Output/Training/checkpoints/VISUAL_INERTIAL/best_model.pth"
            shift
            ;;
        seq_*)
            SEQUENCE_ID="--sequence-id $1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [SEQUENCE_ID]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH       Path to model checkpoint (default: auto-detect from model type)"
            echo "  --data-dir PATH         Path to data directory (default: ../Data/Generated)"
            echo "  --split SPLIT           Data split to evaluate: train, val, or test (default: test)"
            echo "  --show-plots            Display plots interactively"
            echo "  -i, --inertial          Evaluate inertial model"
            echo "  -vi, --visual-inertial  Evaluate visual-inertial model"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Arguments:"
            echo "  SEQUENCE_ID             Specific sequence to evaluate (e.g., seq_000041)"
            echo "                          If not provided, evaluates all sequences in split"
            echo ""
            echo "Examples:"
            echo "  $0                      # Evaluate visual model on test set"
            echo "  $0 --split train        # Evaluate on training set"
            echo "  $0 --split val          # Evaluate on validation set"
            echo "  $0 seq_000041           # Evaluate on specific sequence"
            echo "  $0 --show-plots         # Display plots interactively"
            echo "  $0 -i --split train     # Evaluate inertial model on training set"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Make sure you have trained a model first using Train.py"
    echo "Checkpoints are saved to: ../Output/Training/checkpoints/{MODEL_TYPE}/best_model.pth"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo ""
    echo "Make sure the data directory exists"
    exit 1
fi

echo "============================================"
echo "Evaluating DL-based Odometry Model"
echo "============================================"
echo "Checkpoint:        $CHECKPOINT"
echo "Data directory:    $DATA_DIR"
echo "Split:             $SPLIT"
echo "Image size:        ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "Model type:        $MODEL_TYPE"
if [ -n "$SEQUENCE_ID" ]; then
    echo "Sequence:          ${SEQUENCE_ID#--sequence-id }"
else
    echo "Mode:              Batch (all sequences in $SPLIT split)"
fi
echo "============================================"
echo ""

# Run evaluation
python Test.py \
    --checkpoint "$CHECKPOINT" \
    --data-dir "$DATA_DIR" \
    --split "$SPLIT" \
    --image-height $IMAGE_HEIGHT \
    --image-width $IMAGE_WIDTH \
    $MODEL_TYPE \
    $SEQUENCE_ID \
    $SHOW_PLOTS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: ../Output/Testing/"
    echo "============================================"
else
    echo ""
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
