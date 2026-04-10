#!/bin/bash
#SBATCH --job-name=brats_3dgs
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log

# ─────────────────────────────────────────────────────────────
# Train a model on a BraTS-GLI MRI case.
#
# MODEL_TYPE=gs   → Gaussian Splatting (default, ITERS=30000)
# MODEL_TYPE=unet → U-Net slice interpolation baseline (ITERS=5000)
#
# Usage:
#   sbatch scripts/train.slurm
#   sbatch --export=CASE_ID=BraTS-GLI-00005-100,ITERS=30000 scripts/train.slurm
#   sbatch --export=MODEL_TYPE=unet,SPARSE=5 scripts/train.slurm
# ─────────────────────────────────────────────────────────────

set -e
cd /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/MedVision3D
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medvision3d

CASE_ID="${CASE_ID:-BraTS-GLI-00005-100}"
MODEL_TYPE="${MODEL_TYPE:-gs}"
ITERS="${ITERS:-30000}"
SPARSE="${SPARSE:-0}"

echo "========================================"
echo "BraTS-GLI Training"
echo "  Case:          ${CASE_ID}"
echo "  Model:         ${MODEL_TYPE}"
echo "  Iterations:    ${ITERS}"
echo "  Sparse factor: ${SPARSE}"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

if [ "${MODEL_TYPE}" = "unet" ]; then
    python -u train.py \
        "case_id=${CASE_ID}" \
        "model_type=unet" \
        "unet_iterations=${ITERS}" \
        "sparse_factor=${SPARSE}"
else
    python -u train.py \
        "case_id=${CASE_ID}" \
        "iterations=${ITERS}" \
        "sparse_factor=${SPARSE}"
fi

echo "Training complete."
