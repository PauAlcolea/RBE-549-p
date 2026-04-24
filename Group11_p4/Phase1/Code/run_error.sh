#!/bin/bash

# =========================================================
# EVO Evaluation Script
# =========================================================

GT="../Output/traj_gt.txt"
EST="../Output/traj_est.txt"

echo "Running EVO trajectory evaluation..."

# ---------------------------------------------------------
# 1. Trajectory overlay (XY)
# ---------------------------------------------------------
echo "Generating trajectory overlay (XY)..."
evo_traj tum $GT $EST \
--ref $GT \
--align \
--plot_mode xy \
--plot \
--save_plot ../Output/traj_xy.pdf

# ---------------------------------------------------------
# 1. Trajectory overlay (XZ)
# ---------------------------------------------------------
echo "Generating trajectory overlay (XZ)..."
evo_traj tum $GT $EST \
--ref $GT \
--align \
--plot_mode xz \
--plot \
--save_plot ../Output/traj_xz.pdf

# ---------------------------------------------------------
# 1. Trajectory overlay (YZ)
# ---------------------------------------------------------
echo "Generating trajectory overlay (YZ)..."
evo_traj tum $GT $EST \
--ref $GT \
--align \
--plot_mode yz \
--plot \
--save_plot ../Output/traj_yz.pdf

# ---------------------------------------------------------
# 2. Trajectory overlay (3D)
# ---------------------------------------------------------
echo "Generating trajectory overlay (3D)..."
evo_traj tum $GT $EST \
--ref $GT \
--align \
--plot_mode xyz \
--plot \
--save_plot ../Output/traj_3d.pdf

# ---------------------------------------------------------
# 3. Absolute Trajectory Error (color trajectory)
# ---------------------------------------------------------
echo "Generating ATE visualization..."
evo_ape tum $GT $EST \
-va \
--plot \
--plot_mode xy \
--save_plot ../Output/ate_colormap.pdf

# ---------------------------------------------------------
# 4. ATE histogram
# ---------------------------------------------------------
echo "Generating ATE histogram..."
evo_ape tum $GT $EST \
-va \
--plot \
--plot_histogram \
--save_plot ../Output/ate_histogram.pdf

# ---------------------------------------------------------
# 5. Relative Pose Error
# ---------------------------------------------------------
echo "Generating RPE visualization..."
evo_rpe tum $GT $EST \
-va \
--delta 1 \
--plot \
--save_plot ../Output/rpe.pdf

echo ""
echo "All evaluation plots generated:"
echo "  ../Output/traj_xy.pdf"
echo "  ../Output/traj_xz.pdf"
echo "  ../Output/traj_yz.pdf"
echo "  ../Output/traj_3d.pdf"
echo "  ../Output/ate_colormap.pdf"
echo "  ../Output/ate_histogram.pdf"
echo "  ../Output/rpe.pdf"