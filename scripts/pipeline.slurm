#!/bin/bash
#SBATCH --job-name=brats_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/pipeline_%j.log

# ─────────────────────────────────────────────────────────────────────────
# Complete Pipeline: Explore → Train → Compare
#
# Runs the full workflow:
#   1. Explore dataset (quick check)
#   2. Train Gaussian Splatting
#   3. Compare with baseline methods
#
# Usage:
#   sbatch scripts/pipeline.slurm
#   sbatch --export=CASE_ID=BraTS-GLI-00005-100 scripts/pipeline.slurm
# ─────────────────────────────────────────────────────────────────────────

set -e
cd /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/MedVision3D
mkdir -p logs outputs/metrics outputs/comparisons

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medvision3d

CASE_ID="${CASE_ID:-BraTS-GLI-00005-100}"
N_GAUSS="${N_GAUSS:-100000}"
ITERS="${ITERS:-30000}"

echo "========================================"
echo "BraTS-GLI Complete Pipeline"
echo "  Case:       ${CASE_ID}"
echo "  Gaussians:  ${N_GAUSS}"
echo "  Iterations: ${ITERS}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"
echo ""

# Step 1: Explore
echo "[1/3] Exploring dataset..."
python -u explore.py --mode dataset
sleep 2

# Step 2: Train
echo "[2/3] Training Gaussian Splatting on ${CASE_ID}..."
python -u train.py \
    "case_id=${CASE_ID}" \
    "num_gaussians=${N_GAUSS}" \
    "iterations=${ITERS}"
sleep 2

# Step 3: Compare
echo "[3/3] Comparing with baseline methods..."
python -u compare.py \
    "case_id=${CASE_ID}" \
    "methods=['gs','spline','unet','baseline_voxel','dvr','mip']" \
    "num_gaussians=${N_GAUSS}"

echo ""
echo "========================================"
echo "Pipeline complete! Outputs:"
echo "  - Training logs: logs/"
echo "  - Checkpoints: checkpoints/"
echo "  - Comparison results: outputs/comparisons/"
echo "========================================"
