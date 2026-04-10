# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedVision3D adapts **3D Gaussian Splatting (3DGS)** for MRI volume reconstruction. The core innovation replaces RGB color with scalar tissue intensity, uses volumetric aggregation instead of 2D splatting, and renders fixed slice planes instead of arbitrary novel views. Performance targets: PSNR > 30 dB, SSIM > 0.90, ≥ 30 FPS at 480×480.

## Setup

```bash
conda env create -f environment.yml
conda activate medvision3d
bash scripts/setup_env.sh   # Installs diff-gaussian-rasterization CUDA kernels
```

Dataset: BraTS-GLI 2023 at `../Data/BraTS-GLI/train/` (modalities: t1c, t1n, t2f, t2w, seg as `.nii.gz`).

## Commands

```bash
# Data exploration
python explore.py --mode dataset
python explore.py --mode inspect --case BraTS-GLI-00005-100
python explore.py --mode pipeline --case BraTS-GLI-00005-100 --n_gaussians 1000  # smoke test

# Training (Hydra-style CLI args)
python train.py case_id=BraTS-GLI-00005-100
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 iterations=50000
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 model_type=unet  # train U-Net interp baseline

# Benchmarking
python compare.py case_id=BraTS-GLI-00005-100
python compare.py case_id=BraTS-GLI-00005-100 methods=['gs','baseline_voxel','dvr','mip','spline','unet']

# SLURM (HPC)
sbatch scripts/train.slurm
sbatch scripts/compare.slurm
sbatch scripts/pipeline.slurm   # full workflow: explore → train → compare
sbatch --export=CASE_ID=BraTS-GLI-00005-100,ITERS=50000 scripts/train.slurm
```

## Architecture

**Data flow**: NIfTI volume → `src/preprocessing/mri_loader.py` (load + normalize to [0,1]) → slice extraction → `src/gaussian/gaussian_model.py` (init Gaussians) → training loop in `train.py` → `src/rendering/volume_renderer.py` → `src/utils/metrics.py` (PSNR/SSIM/LPIPS).

**Gaussian representation** (key difference from standard 3DGS):
- `_xyz (N,3)` — position, `_scaling (N,3)` — log-scale axes, `_rotation (N,4)` — quaternion
- `_intensity (N,1)` — scalar tissue intensity (logit-encoded), **not** RGB
- `_opacity (N,1)` — opacity (logit-encoded)

**Rendering pipeline** in `src/rendering/volume_renderer.py`:
1. Cull Gaussians outside 3σ of target slice plane
2. Project 3D Gaussians to 2D in-plane coordinates
3. Depth-sort along slice axis
4. Alpha-composite: `I(x,y) = Σ α_i · intensity_i · Π_{j<i}(1 - α_j)`
5. CUDA kernel via `diff_gaussian_rasterization` with PyTorch fallback

**Baseline methods** in `src/reconstruction/`:
- `BaselineVoxelModel` — learned dense voxel grid (~30 MB, direct lookup)
- `TraditionalRenderer` — non-learned DVR (ray-casting, 1-5 FPS) and MIP (100+ FPS)

## Configuration

`train.py` uses a class-based `Config` with kwargs overrides (not a YAML file), passed via Hydra-style CLI. Key defaults: `num_gaussians=100000`, `iterations=30000`, `densify_from_iter=500`, `densify_until_iter=15000`, loss = MSE + 0.2 × DSSIM.

Densification during training: Gaussians are added/duplicated based on gradients; opacity is reset periodically. LR uses exponential scheduling.

## Outputs

```
checkpoints/   # Model checkpoints per case
outputs/
  reconstructions/
  metrics/       # PSNR/SSIM/LPIPS JSON
  comparisons/   # Benchmarking results
logs/            # TensorBoard (default) or WandB (optional)
```

## Code Style

- Python 3.10, type hints on all class attributes and function parameters
- Class-based Config with kwargs overrides (see `train.py:35-74`)
- Imports: stdlib → third-party → local (with `sys.path.insert(0, str(Path(__file__).parent))`)
- `snake_case` variables/functions, `PascalCase` classes, `UPPER_CASE` constants
