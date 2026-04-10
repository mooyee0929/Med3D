#!/bin/bash
#SBATCH --job-name=setup_medvision
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.log

set -e
cd /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/MedVision3D

echo "=== Creating conda environment: medvision3d ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate medvision3d

echo "=== Installing diff-gaussian-rasterization (3DGS CUDA kernels) ==="
cd /tmp
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

echo "=== Verifying CUDA availability ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('diff-gaussian-rasterization OK')"

echo "=== Done! Activate with: conda activate medvision3d ==="
