# MRI Gaussian Splatting (MedVision3D)

MedVision3D adapts 3D Gaussian Splatting (3DGS) to reconstruct full 3D brain MRI volumes from a sparse stack of 2D slices. Instead of RGB Gaussians rendered from arbitrary camera poses, each Gaussian carries a single scalar tissue intensity and the renderer produces axis-aligned MRI slices. The model is trained on every Nth axial slice of a BraTS-GLI 2023 T1c volume and evaluated only on the slices it never saw, then benchmarked against cubic-spline interpolation, a learned dense voxel grid, a small 3D U-Net, nearest-slice lookup (DVR) and maximum intensity projection (MIP).

![GT vs. all methods, BraTS-GLI-00005-100, sparse-20](outputs/images/BraTS-GLI-00005-100_sparse20_all_compare.png)

*BraTS-GLI-00005-100 trained on every 20th slice. Rows are ten evenly spaced axial positions; columns are ground truth, then each reconstruction (GS, spline, voxel grid, DVR, U-Net, MIP) followed by its absolute-error map (hot colormap clipped at 0.3).*

## Method

**Gaussian representation** (`src/gaussian/gaussian_model.py`). Each Gaussian has a position `_xyz (N,3)`, log-scales `_scaling (N,3)`, a quaternion `_rotation (N,4)`, a logit-encoded opacity `_opacity (N,1)`, and a logit-encoded scalar `_intensity (N,1)` that replaces the RGB / spherical-harmonics color of standard 3DGS. Gaussians are initialized by sampling voxels with intensity above 0.01, weighted by intensity, with small positional jitter and half-voxel scales. The model saves and loads PLY files that store intensity and opacity per Gaussian.

**Slice renderer** (`src/rendering/volume_renderer.py`). To render slice `k` along an axis, the renderer keeps Gaussians within 3 sigma of the slice plane, orders them by distance to the plane, evaluates their 2D footprints on the slice pixel grid, attenuates opacity by the out-of-plane distance, and alpha-composites front to back:

```
I(x, y) = sum_i T_i * a_i(x, y) * intensity_i,   T_i = prod_{j<i} (1 - a_j(x, y))
```

The pure PyTorch path is fully vectorized and differentiable. If the `diff_gaussian_rasterization` CUDA extension is installed, the renderer uses it instead.

**Training** (`train.py`). With `sparse_factor=N`, axial slices `0, N, 2N, ...` are used for training and every other slice is held out. Each iteration renders one random training slice and minimizes `(1 - 0.2) * L1 + 0.2 * DSSIM` with Adam, using per-parameter learning rates (position 1.6e-4 with exponential decay to 1% of the initial value, intensity 2.5e-3, opacity 0.05, scaling 5e-3, rotation 1e-3). Adaptive density control runs every 100 iterations between iterations 500 and 15,000: high-gradient small Gaussians are cloned, high-gradient large ones are split, and low-opacity or oversized ones are pruned. Opacity is reset every 3,000 iterations during that window. Defaults are 100,000 initial Gaussians and 30,000 iterations. With `sparse_factor=0` the model trains on slices from all three axes.

**Baselines** (`compare.py`, `src/reconstruction/`):

| Method | Key | Type | Description |
|---|---|---|---|
| Cubic spline | `spline` | non-learned | Per-pixel `scipy` cubic spline through the training slices along the slice axis |
| Voxel grid | `baseline_voxel` | learned | Dense logit-space voxel grid (7,221,032 parameters) initialized from training slices and nearest-neighbour filled, trained with L1 + MSE for 100k iterations |
| 3D U-Net | `unet` | learned | `TinyUNet3D` (339,569 parameters) that refines a linearly interpolated volume; trained by masking about 1/6 of the training slices each step and supervising only those |
| DVR | `dvr` | non-learned | Returns the nearest training slice for each held-out position |
| MIP | `mip` | non-learned | Maximum intensity projection of the training slices, compared against every held-out slice |

## Results

All numbers below come from the JSON files in [`outputs/comparisons/`](outputs/comparisons/). Each run uses the T1c modality of one BraTS-GLI case with 182 axial slices. PSNR (dB) is averaged over held-out slices only. The GS and U-Net models were trained with `train.py` and loaded from their checkpoints by `compare.py`.

**Sparse-5** (37 training slices, 145 held out)

| Case | GS | Spline | Voxel grid | 3D U-Net | DVR | MIP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BraTS-GLI-00005-100 | **44.34** | 44.12 | 33.25 | 32.65 | 31.92 | 12.53 |
| BraTS-GLI-00027-100 | **43.37** | 41.40 | 32.47 | 30.55 | 30.78 | 11.96 |
| BraTS-GLI-00060-100 | 38.09 | **40.48** | 32.10 | 31.57 | 30.26 | 11.39 |
| Mean | 41.93 | **42.00** | 32.61 | 31.59 | 30.99 | 11.96 |

**Sparse-20** (10 training slices, 172 held out)

| Case | GS | Spline | Voxel grid | 3D U-Net | DVR | MIP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BraTS-GLI-00005-100 | **35.86** | 30.69 | 29.90 | 28.54 | 28.04 | 12.99 |
| BraTS-GLI-00027-100 | **32.78** | 28.03 | 28.17 | 19.48 | 24.75 | 12.31 |
| BraTS-GLI-00060-100 | **29.83** | 27.53 | 27.69 | 24.04 | 24.57 | 11.90 |
| Mean | **32.82** | 28.75 | 28.58 | 24.02 | 25.79 | 12.40 |

**All six runs**

| Method | Mean PSNR (dB) | Parameters |
|---|:---:|:---:|
| Gaussian Splatting | **37.38** | 96,643 to 99,609 Gaussians after training |
| Cubic spline | 35.37 | 0 |
| Voxel grid | 30.60 | 7,221,032 |
| DVR | 28.39 | 0 |
| 3D U-Net | 27.81 | 339,569 |
| MIP | 12.18 | 0 |

Observations from these runs:

- GS gives the best held-out PSNR in five of the six runs. Cubic spline wins BraTS-GLI-00060-100 at sparse-5, and the two are effectively tied on the sparse-5 mean (42.00 vs. 41.93 dB).
- The GS advantage appears when slices are far apart: at sparse-20 it leads the next best method by about 4 dB on average.
- GS reaches higher PSNR than the dense voxel grid in every run while using more than 70 times fewer parameters.
- The 3D U-Net is competitive at sparse-5 but degrades at sparse-20 (19.48 dB on BraTS-GLI-00027-100), where only 10 slices are available for masked training.
- PSNR is averaged over all held-out slices, including near-empty slices outside the brain, which inflates absolute values. The comparison is most meaningful between methods rather than as absolute numbers.
- `compare.py` also records a throughput figure, but its timer covers different work for each method (for example, it includes checkpoint loading and full-volume rendering for GS and training for the voxel grid), so it is not a like-for-like rendering-speed measurement. GS recorded 5 to 10 slices per second, below the original 30 FPS goal. SSIM is used as a training loss but is not recorded in the comparison files.

**Gaussian centres after training** (sparse-20, BraTS-GLI-00005-100), produced by `visualize.py ply --save`:

![Gaussian point cloud](outputs/images/BraTS-GLI-00005-100_sparse20_gs_pointcloud.png)

More figures for every case and sparsity level are in [`outputs/images/`](outputs/images/): `*_all_compare.png` (GT vs. all methods), `*_gs_pointcloud.png` (three orthogonal views of the Gaussian centres, colored by intensity times opacity) and `*_gs_montage.png` (slice montage of the GS-rendered volume).

## Repository structure

```
.
├── train.py                 GS or 3D U-Net training (key=value overrides)
├── compare.py               benchmark GS against the baselines on held-out slices
├── explore.py               dataset audit, case inspection, smoke test, sparse evaluation
├── visualize.py             point clouds, NIfTI montages, GT-vs-reconstruction grids
├── environment.yml          conda environment (Python 3.10, PyTorch 2.1, CUDA 11.8)
├── configs/default.yaml     reference hyperparameters (train.py uses its in-code Config)
├── src/
│   ├── preprocessing/       mri_loader.py: NIfTI loading, [0, 1] normalization, slice extraction
│   ├── gaussian/            gaussian_model.py: intensity Gaussians, densify/prune, PLY I/O
│   ├── rendering/           volume_renderer.py: differentiable slice renderer
│   ├── reconstruction/      voxel grid, DVR/MIP renderer, 3D U-Net
│   └── utils/               PSNR, differentiable SSIM, plotting helpers
├── scripts/                 environment setup, dataset instructions, SLURM jobs
├── outputs/
│   ├── comparisons/         per-run comparison JSON and GS metadata
│   └── images/              comparison grids, point clouds, montages
├── docs/, ADAPTATION.md, WORKFLOW.md, SLURM_GUIDE.md
│                            design notes (some describe earlier iterations of the pipeline)
```

## Setup

```bash
conda env create -f environment.yml
conda activate medvision3d
```

The optional CUDA rasterizer comes from the official 3DGS repository. `scripts/setup_env.sh` creates the environment and then builds `diff-gaussian-rasterization` and `simple-knn` from `graphdeco-inria/gaussian-splatting`:

```bash
bash scripts/setup_env.sh
```

Without the extension, `VolumeRenderer` falls back to its pure PyTorch implementation.

## Data

The code expects BraTS-GLI 2023 training cases as NIfTI files. Access requires registration on Synapse; `bash scripts/download_brats.sh` prints the download steps (project `syn51156910`).

By default, `train.py`, `compare.py` and `explore.py` read from a `Data` folder next to the repository:

```
../Data/BraTS-GLI/train/
└── BraTS-GLI-00005-100/
    ├── BraTS-GLI-00005-100-t1c.nii.gz
    ├── BraTS-GLI-00005-100-t1n.nii.gz
    ├── BraTS-GLI-00005-100-t2f.nii.gz
    ├── BraTS-GLI-00005-100-t2w.nii.gz
    └── BraTS-GLI-00005-100-seg.nii.gz
```

`train.py` and `compare.py` accept `data_root=/path/to/train` to point elsewhere, and `modality=` to switch from the default `t1c`. `explore.py` always uses the default location. Volumes are min-max normalized to [0, 1] on load.

## Usage

### Explore the data

```bash
python explore.py --mode dataset                                    # audit all cases
python explore.py --mode inspect --case BraTS-GLI-00005-100         # shapes, stats, label counts, slice grid
python explore.py --mode pipeline --case BraTS-GLI-00005-100 --n_gaussians 1000   # end-to-end smoke test
python explore.py --mode sparse_eval --case BraTS-GLI-00005-100 --sparse_factor 5 # after training
```

Figures are written to `outputs/explore/`.

### Train

`train.py` takes `key=value` overrides for any field of its `Config` class.

```bash
# Gaussian Splatting on every 5th / 20th axial slice
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=20

# More iterations or a different Gaussian budget
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 iterations=50000 num_gaussians=50000

# 3D U-Net baseline (compare.py loads this checkpoint automatically)
python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5 model_type=unet
```

GS checkpoints are saved to `checkpoints/<case_id>_sparse<N>/iteration_XXXXXX.ply` (at 7,000, 15,000 and the last iteration) together with `slice_split.json` and evaluation grids. Training resumes from the latest PLY in that folder. TensorBoard logs go to `logs/<case_id>_sparse<N>/`. The U-Net checkpoint is `checkpoints/<case_id>_sparse<N>_unet3d.pt`.

### Benchmark

```bash
python compare.py case_id=BraTS-GLI-00005-100 sparse_factor=5
python compare.py case_id=BraTS-GLI-00005-100 sparse_factor=20 methods=gs,spline,dvr
```

`methods` defaults to `gs,spline,unet,baseline_voxel,dvr,mip`. `sparse_factor` must match the value used in `train.py` so the latest GS and U-Net checkpoints are found; a specific GS checkpoint can be given with `gs_checkpoint=path/to/iteration_030000.ply`. If no checkpoint exists, GS and the U-Net are trained inside `compare.py`. With `sparse_factor=0`, ten evenly spaced slices are held out instead.

Outputs in `outputs/comparisons/`, prefixed `<case_id>_sparse<N>`:

- `_comparison.json`: PSNR, training time, memory, parameter count and throughput per method
- `_gs.ply`: the Gaussian model; `_gs_volume.nii.gz`: GS-rendered volume
- `_baseline_volume.nii.gz`, `_spline_volume.nii.gz`, `_unet3d_volume.nii.gz`, `_dvr_volume.nii.gz`, `_mip_volume.nii.gz`: reconstructed volumes
- `_baseline.pt`, `_unet3d.pt`: learned baseline weights

### Visualize

```bash
# Gaussian centres: interactive Open3D window, or a 3-view PNG with --save
python visualize.py ply outputs/comparisons/BraTS-GLI-00005-100_sparse5_gs.ply --save

# Slice montage or a single slice of a volume
python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_sparse5_gs_volume.nii.gz --save
python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_sparse5_gs_volume.nii.gz --axis 2 --slice 90

# Ground truth vs. several reconstructions with error maps
python visualize.py compare outputs/comparisons/BraTS-GLI-00005-100_sparse5_gs_volume.nii.gz \
    --gt ../Data/BraTS-GLI/train/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t1c.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_sparse5_spline_volume.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_sparse5_baseline_volume.nii.gz \
    --also outputs/comparisons/BraTS-GLI-00005-100_sparse5_unet3d_volume.nii.gz \
    --save
```

| Option | Description |
|---|---|
| `--gt PATH` | Ground-truth NIfTI (required for `compare`) |
| `--also PATH` | Additional reconstruction to show; repeatable |
| `--test-slices N` | Number of evenly spaced positions to show (default 10) |
| `--axis INT` | 0 = sagittal, 1 = coronal, 2 = axial (default 2) |
| `--slice INT` | Single slice index for `nii` mode |
| `--save [file]` | Save a PNG instead of opening a window; with no filename it is written to `outputs/images/` |

### SLURM

`scripts/` contains batch jobs for an HPC cluster: `explore.slurm`, `train.slurm`, `compare.slurm`, `pipeline.slurm` (explore, train, compare) and `setup_env.slurm`. Every script changes into the directory given by `MEDVISION3D_DIR` and stops with an error if it is not set. The jobs are configured with environment variables, for example:

```bash
export MEDVISION3D_DIR="$PWD"
sbatch --export=ALL,CASE_ID=BraTS-GLI-00005-100,SPARSE=5 scripts/train.slurm
sbatch --export=ALL,MODEL_TYPE=unet,SPARSE=5 scripts/train.slurm
sbatch --export=ALL,CASE_ID=BraTS-GLI-00005-100,SPARSE=20 scripts/compare.slurm
```

See `SLURM_GUIDE.md` for the other variables each script reads.

## Acknowledgements

- Developed by the MedVision3D team.
- The method builds on 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023). The optional CUDA rasterizer (`diff-gaussian-rasterization`, `simple-knn`) is installed from [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and is subject to its license.
- Data: BraTS 2023 Glioma (BraTS-GLI) challenge dataset, distributed through [Synapse](https://www.synapse.org/Synapse:syn51156910) under the challenge's data use terms. No imaging data is included in this repository.
