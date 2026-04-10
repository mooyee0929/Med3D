#!/bin/bash
# Setup script for MedVision3D conda environment
set -e

echo "=== Creating conda environment: medvision3d ==="
conda env create -f environment.yml

echo "=== Activating environment ==="
conda activate medvision3d

echo "=== Installing diff-gaussian-rasterization (3DGS CUDA kernels) ==="
cd /tmp
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

echo "=== Done! Activate with: conda activate medvision3d ==="
