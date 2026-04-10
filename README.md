# MedVision3D — Interactive 3D Visualization from MRI Slices Using Gaussian Splatting

EECS 504 Computer Vision, Winter 2026 | Team: MedVision3D

## Project Structure

```
MedVision3D/
├── src/
│   ├── preprocessing/      ← MRI loading, slice extraction (mri_loader.py)
│   ├── gaussian/           ← 3DGS model adapted for MRI (gaussian_model.py)
│   ├── rendering/          ← volumetric slice renderer (volume_renderer.py)
│   ├── reconstruction/     ← baseline voxel model + traditional renderer
│   └── utils/              ← metrics (PSNR, SSIM, LPIPS)
├── scripts/
│   ├── setup_env.sh/slurm  ← conda env + CUDA kernel install
│   ├── train.slurm         ← SLURM training
│   ├── compare.slurm       ← SLURM benchmarking
│   ├── explore.slurm       ← SLURM data exploration
│   └── pipeline.slurm      ← full workflow: explore → train → compare
├── train.py                ← GS training entry point
├── compare.py              ← benchmark GS vs baselines
├── explore.py              ← dataset inspection
├── visualize.py            ← visualize .ply and .nii.gz outputs
├── checkpoints/            ← saved model checkpoints
├── outputs/
│   ├── comparisons/        ← benchmark results (.ply, .nii.gz, .json)
│   ├── reconstructions/
│   └── metrics/
└── logs/                   ← TensorBoard + SLURM job logs
```

Dataset: BraTS-GLI 2023 at `../Data/BraTS-GLI/train/` (modalities: `t1c`, `t1n`, `t2f`, `t2w`, `seg` as `.nii.gz`).

## Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate medvision3d
```

### 2. Install 3DGS CUDA Kernels
```bash
bash scripts/setup_env.sh          # manual
sbatch scripts/setup_env.slurm     # SLURM (HPC)
```

### 3. Get Dataset

**BraTS 2023 (requires registration)**
```bash
bash scripts/download_brats.sh   # prints download instructions
```
Register at: https://www.synapse.org/Synapse:syn51156910

## Usage

### Explore Dataset
```bash
python explore.py --mode dataset                                   # audit all cases
python explore.py --mode inspect --case BraTS-GLI-00005-100        # single case details
python explore.py --mode pipeline --case BraTS-GLI-00005-100 --n_gaussians 1000  # smoke test
```

### Train
```bash
python train.py case_id=BraTS-GLI-00005-100                        # default: 100k Gaussians, 30k iters
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 iterations=50000

# Train the U-Net slice-interpolation baseline (compare.py auto-loads it)
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 model_type=unet
```

### Benchmark
Compares GS against three baselines: learned dense voxel grid, direct volume rendering, and MIP.
```bash
python compare.py case_id=BraTS-GLI-00005-100
python compare.py case_id=BraTS-GLI-00005-100 methods=gs,dvr,mip
```

Outputs saved to `outputs/comparisons/`:
- `{case_id}_gs.ply` — Gaussian splat point cloud
- `{case_id}_gs_volume.nii.gz` — GS-rendered 3D volume
- `{case_id}_baseline_volume.nii.gz` — learned voxel grid volume
- `{case_id}_dvr_volume.nii.gz` / `{case_id}_mip_volume.nii.gz` — traditional method volumes
- `{case_id}_comparison.json` — PSNR, time, memory, parameter count per method

### Visualize Outputs
```bash
# Interactive point cloud viewer (Open3D)
python visualize.py ply outputs/comparisons/BraTS-GLI-00005-100_gs.ply

# NIfTI volume — 25-slice montage
python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz

# Single slice
python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_baseline_volume.nii.gz --axis 2 --slice 90

# Compare GT vs GS at held-out test positions
python visualize.py compare outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz \
    --gt /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/Data/BraTS-GLI/train/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t1c.nii.gz \
    --save

# Compare GT vs GS + Baseline + DVR + MIP (all methods side by side)
python visualize.py compare outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz \
    --gt /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/Data/BraTS-GLI/train/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t1c.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_baseline_volume.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_dvr_volume.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_mip_volume.nii.gz \
    --save
```

`compare` mode shows each held-out test slice as a row with columns: **GT | Method | \|GT−Method\|** for every method. Test positions are the same interleaved split used by `compare.py`. Images saved to `outputs/images/`.

| Option | Description |
|--------|-------------|
| `--gt PATH` | Ground-truth NIfTI (required) |
| `--also PATH` | Additional reconstruction to compare; repeatable |
| `--test-slices N` | Number of test positions to show (default: 10) |
| `--axis INT` | Slice axis: 0=sagittal 1=coronal 2=axial (default: 2) |
| `--save [file]` | Save PNG; omit filename to auto-save to `outputs/images/` |

### Full Pipeline (SLURM)
```bash
sbatch scripts/pipeline.slurm
sbatch --export=CASE_ID=BraTS-GLI-00005-100,ITERS=50000 scripts/pipeline.slurm
```

## Architecture

**Data flow:**
```
NIfTI volume (.nii.gz)
    │
    ▼
src/preprocessing/mri_loader.py   ← load + normalize to [0,1], extract slices
    │
    ▼
src/gaussian/gaussian_model.py    ← initialize Gaussians weighted by intensity
    │
    ▼
train.py training loop            ← 30k iters, MSE + 0.2×DSSIM loss, densification
    │
    ▼
src/rendering/volume_renderer.py  ← cull 3σ, project, depth-sort, alpha-composite
    │
    ▼
src/utils/metrics.py              ← PSNR / SSIM / LPIPS
```

**Gaussian representation** (key difference from standard 3DGS):
- `_intensity (N,1)` — scalar tissue intensity (logit-encoded), replaces RGB
- `_xyz (N,3)`, `_scaling (N,3)`, `_rotation (N,4)`, `_opacity (N,1)` — standard 3DGS parameters

**Baseline methods** (`src/reconstruction/`):
| Method | Description | Speed |
|--------|-------------|-------|
| `baseline_voxel` | Learned dense voxel grid (~30 MB) | fast |
| `dvr` | Direct slice extraction from volume (nearest-neighbour) | very fast |
| `mip` | Maximum Intensity Projection | very fast |
| `spline` | Cubic spline interpolation along slice axis | very fast |
| `unet` | 2-level U-Net slice interpolation (~430K params) | medium |

## Evaluation Targets

- PSNR > 30 dB on held-out BraTS slices
- SSIM > 0.90
- Rendering speed ≥ 30 FPS at 480×480
