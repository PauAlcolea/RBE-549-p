#!/bin/bash

# =========================================================
# Example: Quick Test Workflow
# =========================================================
# This script demonstrates how to test your trained models
# =========================================================

set -e  # Exit on error

echo "======================================"
echo "Quick Test Workflow Example"
echo "======================================"
echo ""

# Navigate to Code directory
cd "$(dirname "$0")"

# =========================================================
# Example 1: Test on a single sequence
# =========================================================
echo "Example 1: Testing on a single sequence (seq_000041)..."
echo "Command: ./run_test.sh seq_000041"
echo ""

# Uncomment to run:
# ./run_test.sh seq_000041

# =========================================================
# Example 2: Test on all test sequences (batch mode)
# =========================================================
echo "Example 2: Testing on all test sequences (batch mode)..."
echo "Command: ./run_test.sh"
echo ""

# Uncomment to run:
# ./run_test.sh

# =========================================================
# Example 3: Test with custom parameters
# =========================================================
echo "Example 3: Testing with custom checkpoint..."
echo "Command:"
echo "  python Test.py \\"
echo "      --checkpoint ../Output/Training/checkpoints/VISUAL/best_model.pth \\"
echo "      --test-data-dir ../Data/Generated/test \\"
echo "      --sequence-length 5 \\"
echo "      --lstm-hidden 512 \\"
echo "      --image-height 240 \\"
echo "      --image-width 320 \\"
echo "      -v"
echo ""

# Uncomment to run:
# python Test.py \
#     --checkpoint ../Output/Training/checkpoints/VISUAL/best_model.pth \
#     --test-data-dir ../Data/Generated/test \
#     --sequence-length 5 \
#     --lstm-hidden 512 \
#     --image-height 240 \
#     --image-width 320 \
#     -v

# =========================================================
# Example 4: Use EVO tools on outputs
# =========================================================
echo "Example 4: Using EVO tools on test outputs..."
echo "Commands (run after testing):"
echo ""
echo "  # Navigate to sequence output directory"
echo "  cd ../Output/Testing/VISUAL/seq_000041"
echo ""
echo "  # Plot trajectory overlay"
echo "  evo_traj tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \\"
echo "      --ref seq_000041_tum_gt.txt \\"
echo "      --align \\"
echo "      --plot"
echo ""
echo "  # Compute and plot ATE"
echo "  evo_ape tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \\"
echo "      -va \\"
echo "      --plot \\"
echo "      --save_plot ate.pdf"
echo ""
echo "  # Compute and plot RPE"
echo "  evo_rpe tum seq_000041_tum_gt.txt seq_000041_tum_est.txt \\"
echo "      -va \\"
echo "      --delta 1 \\"
echo "      --plot \\"
echo "      --save_plot rpe.pdf"
echo ""

# =========================================================
# Example 5: Batch analysis of results
# =========================================================
echo "Example 5: Analyzing batch test results..."
echo "Commands:"
echo ""
echo "  # View summary statistics"
echo "  cat ../Output/Testing/VISUAL/summary.json | python -m json.tool"
echo ""
echo "  # Extract ATE RMSE for all sequences"
echo "  python -c \\"
echo "      import json; \\"
echo "      data = json.load(open('../Output/Testing/VISUAL/summary.json')); \\"
echo "      for seq in data['sequences']: \\"
echo "          print(f\\\"{seq['sequence_id']}: {seq['ate_rmse']:.4f}m\\\")\\"
echo ""

echo "======================================"
echo "To actually run these examples:"
echo "1. Uncomment the desired example above"
echo "2. Run this script: ./examples_test.sh"
echo "======================================"
