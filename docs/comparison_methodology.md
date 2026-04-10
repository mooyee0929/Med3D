# MRI Reconstruction Method Comparison

## Overview

This document describes the experimental setup used to benchmark **3D Gaussian Splatting (GS)** against three baseline methods for MRI volume reconstruction on the BraTS-GLI 2023 dataset.

## Dataset

- **Source**: BraTS-GLI 2023 Training Set
- **Case**: `BraTS-GLI-00005-100`
- **Modality**: T1-weighted contrast-enhanced (t1c)
- **Volume size**: 240 x 240 x 182 (H x W x D axial slices)
- **Preprocessing**: Min-max normalization to [0, 1]

## Train / Test Split

Slices are split along the axial axis using an **interleaved strategy** to measure interpolation quality rather than boundary extrapolation:

- **Total slices**: 182
- **Test slices**: 10, evenly spaced at positions [9, 27, 45, 63, 81, 99, 117, 135, 153, 171]
- **Training slices**: 172 (all remaining slices)

This ensures test slices are surrounded by training data on both sides, testing each method's ability to reconstruct unseen intermediate slices.

## Methods

### 1. Gaussian Splatting (GS)

A 3D Gaussian Splatting model adapted for volumetric MRI reconstruction. Each Gaussian is parameterized by:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `_xyz` | (N, 3) | 3D position |
| `_scaling` | (N, 3) | Log-scale axes |
| `_rotation` | (N, 4) | Quaternion orientation |
| `_intensity` | (N, 1) | Scalar tissue intensity (logit-encoded) |
| `_opacity` | (N, 1) | Opacity (logit-encoded) |

**Key difference from standard 3DGS**: Uses scalar intensity instead of RGB color, with volumetric aggregation via alpha compositing along the slice normal axis.

**Training configuration**:
- Initial Gaussians: 100,000 (final: 98,316 after densification/pruning)
- Iterations: 30,000
- Loss: (1 - 0.2) x L1 + 0.2 x DSSIM
- Optimizer: Adam with per-parameter learning rates
  - xyz: 1.6e-4 -> 1.6e-6 (exponential decay)
  - intensity: 2.5e-3
  - opacity: 0.05
  - scaling: 5e-3
  - rotation: 1e-3
- Densification: every 100 iters from iter 500 to 15,000 (gradient threshold: 0.0002)
- Opacity reset: every 3,000 iters up to iter 15,000

**Rendering pipeline**:
1. Cull Gaussians outside 3 sigma of target slice plane
2. Project 3D Gaussians to 2D in-plane coordinates
3. Depth-sort along slice axis
4. Alpha-composite: `I(x,y) = sum_i alpha_i * intensity_i * prod_{j<i}(1 - alpha_j)`

**Evaluation**: Pre-trained checkpoint loaded (`iteration_030000.ply`), no re-training during comparison.

### 2. Baseline Voxel Grid

A learnable dense voxel grid with sigmoid activation. Each voxel stores a logit-space intensity value, optimized to minimize reconstruction loss.

**Training configuration**:
- Parameters: 7,221,032 (dense grid matching volume dimensions)
- Iterations: 100,000
- Learning rate: 0.01 with cosine annealing (eta_min = 1e-4)
- Loss: L1 + MSE
- Initialization: Training slices set to logit(GT), held-out slices interpolated from nearest training neighbor

**Inference**: Direct voxel lookup with sigmoid activation — `output = sigmoid(voxels[:, :, slice_idx])`.

### 3. Direct Volume Rendering (DVR)

A non-learned nearest-neighbor baseline. For each test slice, the method returns the closest training slice by index distance.

- **No training** or optimization
- **No parameters** to learn
- Tests pure spatial interpolation via nearest-neighbor lookup

### 4. Maximum Intensity Projection (MIP)

A non-learned projection baseline. Computes the maximum intensity across the depth axis from training slices, producing a single 2D projection that is compared against each test slice.

- **No training** or optimization
- **No parameters** to learn
- Collapses 3D information into a single 2D image — expected to perform poorly on per-slice reconstruction

## Evaluation Metric

**PSNR (Peak Signal-to-Noise Ratio)** in dB, computed on 10 held-out test slices:

```
PSNR = 10 * log10(1.0 / MSE)
```

where MSE is the mean squared error between the predicted and ground-truth slice (both normalized to [0, 1]).

## Results

| Method | PSNR (dB) | Training Time (s) | Memory (MB) | Parameters |
|--------|-----------|-------------------|-------------|------------|
| **Gaussian Splatting** | **44.92** | 0 (pre-trained) | 5.25 | 98,316 |
| Baseline Voxel Grid | 34.31 | 152.88 | ~0 | 7,221,032 |
| Direct Volume Rendering | 32.41 | 0 | 26.03 | 0 |
| Max Intensity Projection | 12.30 | 0 | 26.03 | 0 |

### Key Observations

1. **GS achieves the highest PSNR (44.92 dB)**, outperforming the second-best method (Baseline Voxel) by +10.6 dB, demonstrating the effectiveness of the 3D Gaussian representation for volumetric MRI data.

2. **GS is the most parameter-efficient**: 98K Gaussians vs 7.2M voxel parameters (73x fewer), while achieving significantly better reconstruction quality.

3. **GS uses less memory (5.25 MB)** compared to the voxel-based methods (26 MB for storing the raw volume), showing that a sparse Gaussian representation is more storage-efficient than dense grids.

4. **Baseline Voxel Grid (34.31 dB)** benefits from direct supervision at training positions but struggles with interpolation at held-out slices, limited by nearest-neighbor initialization for unseen positions.

5. **DVR (32.41 dB)** provides a reasonable non-learned baseline through nearest-neighbor lookup, indicating that adjacent slices share substantial structural similarity.

6. **MIP (12.30 dB)** performs poorly as expected — collapsing 3D information into a single maximum projection fundamentally cannot represent individual slice content.

## Visualization

Two visualization outputs are generated by `visualize_comparison.py`:

- **Method Comparison** (`BraTS-GLI-00005-100_method_comparison.png`): Side-by-side axial slices (positions 45, 91, 136) showing Ground Truth, GS, Baseline Voxel, DVR, and MIP reconstructions.
- **Error Maps** (`BraTS-GLI-00005-100_error_maps.png`): Absolute error `|GT - Recon|` heatmaps for each method, highlighting spatial distribution of reconstruction errors.

## Hardware

- **GPU**: NVIDIA A40
- **Platform**: Great Lakes HPC (SLURM)

## Reproduction

```bash
# Run comparison
sbatch scripts/compare.slurm
# or directly:
python compare.py case_id=BraTS-GLI-00005-100 methods="['gs','baseline_voxel','dvr','mip']"

# Generate visualization
python visualize_comparison.py
```
