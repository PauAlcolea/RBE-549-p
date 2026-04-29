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
# Example 1: Evaluate on test set (default)
# =========================================================
echo "Example 1: Evaluating on test set (all sequences)..."
echo "Command: ./run_test.sh"
echo ""

# Uncomment to run:
# ./run_test.sh

# =========================================================
# Example 2: Evaluate on training set
# =========================================================
echo "Example 2: Evaluating on training set..."
echo "Command: ./run_test.sh --split train"
echo ""

# Uncomment to run:
# ./run_test.sh --split train

# =========================================================
# Example 3: Evaluate on validation set
# =========================================================
echo "Example 3: Evaluating on validation set..."
echo "Command: ./run_test.sh --split val"
echo ""

# Uncomment to run:
# ./run_test.sh --split val

# =========================================================
# Example 4: Evaluate on a single sequence
# =========================================================
echo "Example 4: Evaluating on a single sequence (seq_000041)..."
echo "Command: ./run_test.sh seq_000041"
echo ""

# Uncomment to run:
# ./run_test.sh seq_000041

# =========================================================
# Example 5: Evaluate with custom parameters
# =========================================================
echo "Example 5: Evaluating with custom checkpoint on validation set..."
echo "Command:"
echo "  python Test.py \\"
echo "      --checkpoint ../Output/Training/checkpoints/VISUAL/best_model.pth \\"
echo "      --data-dir ../Data/Generated \\"
echo "      --split val \\"
echo "      --image-height 240 \\"
echo "      --image-width 320 \\"
echo "      -v"
echo ""

# Uncomment to run:
# python Test.py \
#     --checkpoint ../Output/Training/checkpoints/VISUAL/best_model.pth \
#     --data-dir ../Data/Generated \
#     --split val \
#     --image-height 240 \
#     --image-width 320 \
#     -v

# =========================================================
# Example 6: Use EVO tools on outputs
# =========================================================
echo "Example 6: Using EVO tools on evaluation outputs..."
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
# Example 7: Batch analysis of results
# =========================================================
echo "Example 7: Analyzing batch evaluation results..."
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
