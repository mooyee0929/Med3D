#!/bin/bash
#SBATCH --job-name=brats_explore
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/explore_%j.log

# ─────────────────────────────────────────────────────────────
# Data exploration and pipeline testing.
#
# Usage:
#   sbatch scripts/explore.slurm
#   sbatch --export=MODE=inspect,CASE=BraTS-GLI-00005-100 scripts/explore.slurm
#   sbatch --export=MODE=pipeline,CASE=BraTS-GLI-00005-100,N_GAUSS=1000 scripts/explore.slurm
# ─────────────────────────────────────────────────────────────

set -e
cd /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/MedVision3D
mkdir -p logs outputs/explore

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medvision3d

MODE="${MODE:-dataset}"
CASE="${CASE:-BraTS-GLI-00005-100}"
N_GAUSS="${N_GAUSS:-1000}"

echo "Mode: ${MODE} | Case: ${CASE}"

if [ "${MODE}" = "dataset" ]; then
    python explore.py --mode dataset

elif [ "${MODE}" = "inspect" ]; then
    python explore.py --mode inspect --case "${CASE}"

elif [ "${MODE}" = "pipeline" ]; then
    python explore.py --mode pipeline --case "${CASE}" --n_gaussians "${N_GAUSS}"

else
    echo "Unknown MODE=${MODE}"
    exit 1
fi
