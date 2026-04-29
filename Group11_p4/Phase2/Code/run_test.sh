#!/bin/bash

# =========================================================
# Test DL-based Odometry Models
# =========================================================
# Usage examples:
#   ./run_test.sh                    # Test visual model on all test sequences
#   ./run_test.sh seq_000041         # Test on specific sequence
#   ./run_test.sh --show-plots       # Display plots interactively
# =========================================================

# Default configuration
CHECKPOINT="../Output/Training/checkpoints/VISUAL/best_model.pth"
TEST_DATA_DIR="../Data/Generated/test"
SEQUENCE_LENGTH=5
LSTM_HIDDEN=512
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
        --test-data-dir)
            TEST_DATA_DIR="$2"
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
            echo "  --test-data-dir PATH    Path to test data directory (default: ../Data/Generated/test)"
            echo "  --show-plots            Display plots interactively"
            echo "  -i, --inertial          Test inertial model"
            echo "  -vi, --visual-inertial  Test visual-inertial model"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Arguments:"
            echo "  SEQUENCE_ID             Specific sequence to test (e.g., seq_000041)"
            echo "                          If not provided, tests all sequences in test split"
            echo ""
            echo "Examples:"
            echo "  $0                      # Test visual model on all test sequences"
            echo "  $0 seq_000041           # Test on specific sequence"
            echo "  $0 --show-plots         # Display plots interactively"
            echo "  $0 -i                   # Test inertial model"
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

# Check if test data directory exists
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "ERROR: Test data directory not found: $TEST_DATA_DIR"
    echo ""
    echo "Make sure you have generated test data"
    exit 1
fi

echo "============================================"
echo "Testing DL-based Odometry Model"
echo "============================================"
echo "Checkpoint:        $CHECKPOINT"
echo "Test data:         $TEST_DATA_DIR"
echo "Sequence length:   $SEQUENCE_LENGTH"
echo "LSTM hidden:       $LSTM_HIDDEN"
echo "Image size:        ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "Model type:        $MODEL_TYPE"
if [ -n "$SEQUENCE_ID" ]; then
    echo "Sequence:          ${SEQUENCE_ID#--sequence-id }"
else
    echo "Mode:              Batch (all sequences)"
fi
echo "============================================"
echo ""

# Run test
python Test.py \
    --checkpoint "$CHECKPOINT" \
    --test-data-dir "$TEST_DATA_DIR" \
    --sequence-length $SEQUENCE_LENGTH \
    --lstm-hidden $LSTM_HIDDEN \
    --image-height $IMAGE_HEIGHT \
    --image-width $IMAGE_WIDTH \
    $MODEL_TYPE \
    $SEQUENCE_ID \
    $SHOW_PLOTS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Testing completed successfully!"
    echo "Results saved to: ../Output/Testing/"
    echo "============================================"
else
    echo ""
    echo "ERROR: Testing failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
