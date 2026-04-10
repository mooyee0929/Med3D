# MedVision3D Workflow

## Overview: Gaussian Splatting for MRI Reconstruction

**For a detailed technical explanation of how Gaussian Splatting was adapted from RGB images to MRI, see [ADAPTATION.md](ADAPTATION.md).**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ONE CONTINUOUS WORKFLOW (5 Sequential Steps)                            │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: DATA LOADING          Step 2: PREPROCESSING    Step 3: TRAINING
       ↓                              ↓                      ↓
Load NIfTI volumes       →    Extract 2D slices    →   Optimize Gaussians
(3D brain MRI file)           Normalize [0,1]         (30k iterations)
                              Train/test split        
                                                      Densification:
                                                      • Add Gaussians
                              Step 4: EVALUATION      • Remove Gaussians
                                     ↓                • Render slices
                              Render test slices  
                              Compute PSNR, SSIM
                              Compare predictions
                              vs ground truth
                              
                              Step 5: COMPARISON
                                     ↓
                              GS vs Baseline Voxel
                              GS vs DVR
                              GS vs MIP
                              Benchmark metrics
```

### Summary: The 5 Steps (One Continuous Workflow)

| Step | What | Input | Output | Time |
|------|------|-------|--------|------|
| **1. DATA LOADING** | Load MRI from disk | `t1c.nii.gz` file | 3D volume array | <1s |
| **2. PREPROCESSING** | Extract & normalize slices | 3D volume | Train/test slices [0,1] | <1s |
| **3. TRAINING** | Optimize Gaussian Splatting model | Train slices | Trained checkpoint | 30-60 min |
| **4. EVALUATION** | Test model & compute metrics | Trained model + test slices | PSNR/SSIM/LPIPS values | 5-10 min |
| **5. COMPARISON** | Compare GS vs traditional methods | Training/test data | Results table + JSON | 20-40 min |

**Total duration: 1-2 hours**

---

## Step-by-Step Workflow

### **Step 1: Data Loading** (`src/preprocessing/mri_loader.py`)
**Input:** NIfTI volume file (`.nii.gz`)  
**Output:** 3D numpy array `shape=(D, H, W)`

```python
# Load volume
volume = load_nifti("../Data/BraTS-GLI-00005-100/t1c.nii.gz")
# shape: (155, 182, 218) for axial slices

# Extract slices
slices = extract_slices(volume, axis=2)  # axis=2 for axial
# Returns list of 155 slices, each (182, 218)
```

**Key utilities:**
- `load_nifti()`: Loads compressed `.nii.gz` files
- `extract_slices()`: Extracts 2D slices along chosen axis
- Handles NIfTI header information (spacing, orientation)

---

### **Step 2: Preprocessing** 
**Input:** Raw intensity values  
**Output:** Normalized slices in [0, 1] range

```python
# Normalize to [0, 1]
slices_norm = (slices - slices.min()) / (slices.max() - slices.min())

# Create train/test split
train_slices = slices_norm[:n_train]  # Most slices for training
test_slices = slices_norm[n_train:]   # Held-out slices for evaluation
```

**Why normalize?**
- Gaussian model outputs intensities in [0, 1] via sigmoid
- Ground truth must match output range for loss computation

---

### **Step 3A: Gaussian Splatting Training** (`train.py`)
**Input:** Training slices  
**Output:** Trained Gaussian model checkpoint

```python
# Initialize Gaussian model
gs_model = GaussianModel(
    image_size=(182, 218),
    num_gaussians=100_000
)

# Optimization loop (30,000 iterations)
for iteration in range(30_000):
    # 1. Select random training slice
    slice_idx = random.randint(0, len(train_slices)-1)
    gt_slice = train_slices[slice_idx]
    
    # 2. Render that slice from current Gaussian scene
    rendered = renderer(gs_model, axis=2, slice_idx=slice_idx)
    pred = rendered["render"]  # (H, W)
    
    # 3. Compute loss (MSE + DSSIM)
    loss = mse_loss(pred, gt_slice) + λ * dssim_loss(pred, gt_slice)
    
    # 4. Backprop & update Gaussian parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 5. Densification (add/remove Gaussians based on gradients)
    if iteration % 100 == 0 and densify_from ≤ iteration < densify_until:
        add_gaussians_with_high_gradients()
        remove_gaussians_with_zero_opacity()
```

**Key hyperparameters:**
- `num_gaussians=100_000`: Number of 3D Gaussian primitives
- `iterations=30_000`: Total training steps
- `densify_from_iter=500, densify_until_iter=15_000`: When to add/remove Gaussians
- `lambda_dssim=0.2`: Weight for DSSIM loss

**What happens in each Gaussian:**
- Position (x, y, z): Where the Gaussian is located in 3D space
- Covariance matrix: Size and orientation (parametrized by scale + rotation)
- Intensity: Grayscale value [0, 1] for tissue intensity
- Opacity: How opaque/transparent the Gaussian is

---

### **Step 3B: Rendering** (`src/rendering/volume_renderer.py`)
**Input:** Gaussian model + slice axis + slice index  
**Output:** Rendered 2D slice

```python
# Rendering a single slice:
1. Cull Gaussians outside 3σ of slice plane
   (only Gaussians near slice z=z_slice are rendered)

2. Project surviving Gaussians onto slice plane (H, W)
   (3D Gaussians → 2D in-plane ellipses)

3. Sort by depth (z-coordinate)

4. Alpha-composite front-to-back
   I(x,y) = Σ_i ( α_i * color_i * Π_j(1 - α_j) for j < i )
   
   Where:
   - α_i = opacity of Gaussian i
   - color_i = intensity of Gaussian i
   - Π = cumulative transmittance
```

**Key difference from RGB splatting:**
- Standard 3DGS: Each Gaussian has RGB color + SH coefficients
- MRI Splatting: Each Gaussian has single scalar intensity (grayscale)
- Renders volumetric slices, not novel camera viewpoints

---

### **Step 4: Evaluation on Held-Out Slices** (`src/utils/metrics.py`)
**Input:** Trained model + test slices  
**Output:** Quality metrics (PSNR, SSIM, LPIPS)

```python
# Evaluate on held-out slices
psnrs = []
for test_idx in range(len(test_slices)):
    # Render slice using trained Gaussians
    rendered = renderer(gs_model, axis=2, slice_idx=test_idx)
    pred = rendered["render"]
    
    gt = test_slices[test_idx]
    
    # Compute metrics
    psnr_val = psnr(pred, gt)           # Peak Signal-to-Noise Ratio (dB)
    ssim_val = ssim(pred, gt)           # Structural Similarity Index
    lpips_val = lpips(pred, gt)         # Learned Perceptual Image Patch Similarity
    
    psnrs.append(psnr_val)

mean_psnr = np.mean(psnrs)
print(f"Mean PSNR: {mean_psnr:.2f} dB")
```

**Metrics explained:**
- **PSNR** (higher=better): Peak signal-to-noise ratio, standard image quality metric
- **SSIM** (higher=better): Structural similarity (luminance, contrast, structure)
- **LPIPS** (lower=better): Perceptual distance learned by neural network

---

### **Step 5: Comparison with Traditional Methods** (`compare.py`)
**Input:** Same training/test data  
**Output:** Benchmark comparison

```
Methods compared:
┌──────────────────────────────────────────────────────────────┐
│ 1. Gaussian Splatting (Learned, Sparse)                      │
│    → Train Gaussians on training slices                       │
│    → Memory: ~100k params × 8 floats = 3.2 MB                │
│    → Speed: Real-time rendering (30+ FPS)                    │
│                                                               │
│ 2. Baseline Voxel Grid (Learned, Dense)                       │
│    → Learn 3D voxel intensities directly                      │
│    → Memory: D×H×W × 4 bytes = 30+ MB                        │
│    → Speed: Fast inference (direct slice lookup)             │
│                                                               │
│ 3. Direct Volume Rendering - DVR (Non-learned)               │
│    → Ray-casting with alpha compositing                      │
│    → Memory: Store input volume only                         │
│    → Speed: ~1-5 FPS (depends on ray steps)                  │
│                                                               │
│ 4. Max Intensity Projection - MIP (Non-learned)               │
│    → Simple max along viewing axis                           │
│    → Memory: Store input volume only                         │
│    → Speed: Fastest (~100+ FPS)                              │
└──────────────────────────────────────────────────────────────┘

Result: Compare PSNR, speed, memory for each method
```

---

## Usage: Complete Pipeline

### **Option A: Train Gaussian Splatting Only (Steps 1-4)**
```bash
python train.py case_id=BraTS-GLI-00005-100

# This runs: DATA LOADING → PREPROCESSING → TRAINING → EVALUATION
# Output:
# - Checkpoints: checkpoints/BraTS-GLI-00005-100/
# - Logs: logs/BraTS-GLI-00005-100/ (TensorBoard)
# - Metrics: outputs/metrics/
```

### **Option B: Full Pipeline (All 5 Steps)**
```bash
sbatch scripts/pipeline.slurm case_id=BraTS-GLI-00005-100

# This runs all steps:
# Step 1: DATA LOADING
# Step 2: PREPROCESSING  
# Step 3: TRAINING (30-60 min)
# Step 4: EVALUATION
# Step 5: COMPARISON with baselines (20-40 min)

# Output: All checkpoints + comparison table
```

### **Option C: Compare All Methods (Steps 1-2, then 5)**
```bash
python compare.py case_id=BraTS-GLI-00005-100 \
  methods=['gs','baseline_voxel','dvr','mip']

# This runs: DATA LOADING → PREPROCESSING → COMPARISON
# Assumes training is already done (checkpoints exist)
# Output:
# - Comparison results: outputs/comparisons/BraTS-GLI-00005-100_comparison.json
# - Summary table with PSNR, memory, FPS for each method
```

### **Option D: Explore Data Only (Steps 1-2)**
```bash
# Inspect single case
python explore.py --mode inspect --case BraTS-GLI-00005-100

# Audit all cases
python explore.py --mode dataset

# Pipeline smoke-test (fast small test)
python explore.py --mode pipeline --case BraTS-GLI-00005-100 --n_gaussians 1000
```

---

### Visual: How Steps Connect

```
┌─────────────────────┐
│ Step 1: Load        │ Load NIfTI file
│ DATA LOADING        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Step 2: Prepare     │ Extract slices, normalize, split
│ PREPROCESSING       │
└──────────┬──────────┘
           ↓
    ╔════════════════════════════════════════╗
    ║ TRAINING & EVALUATION PAIR             ║
    ║ (Steps 3 & 4 - happens together)      ║
    ╠════════════════════════════════════════╣
    ║ Step 3: TRAINING                       ║
    ║ • Optimize Gaussians on training      ║
    ║   slices (30,000 iterations)          ║
    ║ • Sample slices randomly               ║
    ║ • Render → compare → backprop         ║
    ║ • Densify (add/remove Gaussians)      ║
    ║ • Save checkpoints                     ║
    ║                                        ║
    ║ Step 4: EVALUATION                     ║
    ║ • Render held-out test slices         ║
    ║ • Compute PSNR, SSIM, LPIPS           ║
    ║ • Log metrics to TensorBoard           ║
    ╚════════════════════════════════════════╝
           ↓
┌─────────────────────────────────────────────┐
│ Step 5: COMPARISON                          │
│ • Compare GS with baselines                 │
│ • Run: Baseline Voxel, DVR, MIP            │
│ • Save results: outputs/comparisons/        │
└─────────────────────────────────────────────┘
```

---

## Architecture: Where Each Component Fits

```
src/
├── preprocessing/
│   └── mri_loader.py          ← Step 1-2: Load & extract slices
│
├── gaussian/
│   └── gaussian_model.py      ← Step 3A: Gaussian scene representation
│
├── rendering/
│   └── volume_renderer.py     ← Step 3B: Render slices from Gaussians
│
├── reconstruction/
│   ├── baseline_model.py      ← Step 5: Dense voxel baseline
│   └── traditional_renderer.py ← Step 5: DVR, MIP non-learned baselines
│
├── evaluation/                ← Step 4: Metrics computation
│
└── utils/
    ├── metrics.py             ← PSNR, SSIM, LPIPS
    └── viz.py                 ← Visualization helpers

train.py      ← Main training script (Steps 1-4)
compare.py    ← Comparison script (Steps 1,5)
explore.py    ← Data exploration & debugging
```

---

## Key Design Decisions

### Why Gaussian Splatting?
- **Memory-efficient**: Sparse Gaussian primitives vs dense voxel grids
- **Real-time**: CUDA kernels render at 30+ FPS
- **Learnable**: Optimize positions, covariances, intensities end-to-end
- **1D Intensity**: Adapted from RGB splatting for grayscale MRI

### Why Compare with Traditional Methods?
- **Baseline Voxel Grid**: Shows memory/quality tradeoff of dense learning
- **DVR/MIP**: Classical non-learned volume rendering baseline
- **Evaluate real benefit** of learned Gaussian representation

### Training on Slices (Not 3D)
- MRI slices are 2D ground truth (standard clinical visualization)
- Render slices at train-time, evaluate on held-out slices
- More memory-efficient than 3D volume rendering
- Matches clinical use case (slice review)

---

## Performance Expectations

| Method | PSNR (dB) | Memory | Training Time | Inference |
|--------|-----------|--------|---|---|
| GS | 28-32 | ~3 MB | 30-60 min | Real-time |
| Baseline Voxel | 26-30 | ~30 MB | 20-40 min | Direct |
| DVR | 24-28 | ~50 MB | 0 min | 1-5 FPS |
| MIP | 20-24 | ~50 MB | 0 min | 100+ FPS |

*Typical numbers for BraTS case at 100k Gaussians*

---

## Next Steps

1. **Train on full dataset**: Run `train.py` on multiple cases
2. **Visualize results**: Check checkpoints with rendering pipeline
3. **Optimize hyperparameters**: Tune `num_gaussians`, `lambda_dssim`
4. **Extend to 3D rendering**: Adapt for arbitrary camera poses (not just slices)
5. **Real-time viewer**: Build interactive viewer with WebGL rendering
