#!/usr/bin/env bash
# Evaluate a KITTI trajectory and save BOTH error and map plots.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <gt_poses.txt> <estimated_traj.txt>" >&2
    exit 1
fi

GT="$1"
EST="$2"
PLOT_ERROR="${EST%.txt}_error.pdf"
PLOT_TRAJ="${EST%.txt}_traj_xz.pdf"

if ! command -v evo_ape &>/dev/null; then
    echo "error: evo not found — install with: pip install evo" >&2
    exit 1
fi

echo "=== 1. Generating Error Statistics Plot (APE) ==="
# We use 'xyz' here to see the error over the whole 3D path
MPLBACKEND=Agg evo_ape kitti "$GT" "$EST" \
    --align \
    --correct_scale \
    --plot \
    --plot_mode xyz \
    --save_plot "$PLOT_ERROR"

echo "=== 2. Generating XZ Trajectory Comparison (Top-Down View) ==="
# Changed plot_mode to 'xz' for KITTI's ground plane
MPLBACKEND=Agg evo_traj kitti "$EST" \
    --ref="$GT" \
    --align \
    --correct_scale \
    --plot \
    --plot_mode xz \
    --save_plot "$PLOT_TRAJ"

echo ""
echo "Success! Results saved to:"
echo "  - $PLOT_ERROR (Error Statistics)"
echo "  - $PLOT_TRAJ (Bird's Eye Map View)"