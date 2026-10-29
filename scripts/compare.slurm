#!/bin/bash
#SBATCH --job-name=brats_compare
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/compare_%j.log

# ─────────────────────────────────────────────────────────────────────────
# Compare Gaussian Splatting vs Traditional Reconstruction Methods
#
# Benchmarks: Gaussian Splatting, Voxel Grid, DVR, MIP, Cubic Spline, U-Net
# Metrics: PSNR, memory usage, inference speed (FPS)
#
# Usage:
#   sbatch scripts/compare.slurm
#   sbatch --export=CASE_ID=BraTS-GLI-00005-100,METHODS="gs,baseline_voxel,dvr,mip,spline,unet" scripts/compare.slurm
# ─────────────────────────────────────────────────────────────────────────

set -e
: "${MEDVISION3D_DIR:?Set MEDVISION3D_DIR to the repository root, e.g. sbatch --export=ALL,MEDVISION3D_DIR=/path/to/repo}"
cd "${MEDVISION3D_DIR}"
mkdir -p logs outputs/comparisons

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medvision3d

CASE_ID="${CASE_ID:-BraTS-GLI-00005-100}"
METHODS="${METHODS:-gs,spline,unet,baseline_voxel,dvr,mip}"
N_GAUSS="${N_GAUSS:-100000}"
SPARSE="${SPARSE:-5}"

echo "========================================"
echo "BraTS-GLI Method Comparison"
echo "  Case:          ${CASE_ID}"
echo "  Methods:       ${METHODS}"
echo "  Gaussians:     ${N_GAUSS}"
echo "  Sparse factor: ${SPARSE}"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

python -u compare.py \
    "case_id=${CASE_ID}" \
    "methods=${METHODS}" \
    "num_gaussians=${N_GAUSS}" \
    "sparse_factor=${SPARSE}"

echo "Comparison complete. Results in outputs/comparisons/"
