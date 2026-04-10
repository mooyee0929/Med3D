# SLURM Scripts Guide

Quick reference for submitting jobs to SLURM clusters.

## Script Overview: What Gets Executed

```
5-Step Workflow:  Step 1      Step 2         Step 3       Step 4       Step 5
                  ↓           ↓              ↓            ↓            ↓
                  DATA    →  PREPROC    →  TRAINING  →  EVAL    →  COMPARISON
                  LOAD

Scripts in order:
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. explore.slurm   [Step1-2]      ← Check data first (5-30 min)
│ 2. train.slurm            [Step3-4]        ← Train model (30-60 min)
│ 3. compare.slurm                  [Step5]   ← Benchmark (20-40 min)
│ 4. pipeline.slurm  [All steps]    ← OR run everything at once (1-2 hrs)
└─────────────────────────────────────────────────────────────────────────┘
```

## Available Scripts

### 1. **explore.slurm** — Data Exploration (Start Here)
Inspect and verify MRI data before training. **Run this first!**

```bash
# Audit all cases in dataset
sbatch scripts/explore.slurm

# Inspect single case (detailed analysis)
sbatch --export=MODE=inspect,CASE=BraTS-GLI-00005-100 scripts/explore.slurm

# Pipeline smoke-test (small 1k Gaussian test)
sbatch --export=MODE=pipeline,CASE=BraTS-GLI-00005-100,N_GAUSS=1000 scripts/explore.slurm
```

**Modes:**
- `dataset`: Audit all available cases
- `inspect`: Detailed analysis of single case (shape, stats, label distribution)
- `pipeline`: Quick end-to-end test with small model

**Output:**
- Visualizations: `outputs/explore/`
- Logs: `logs/explore_<jobid>.log`

**Duration:** 5-30 min depending on mode

---

### 2. **train.slurm** — Train Gaussian Splatting
Train the 3D Gaussian Splatting model on a specified MRI case.

```bash
# Default: BraTS-GLI-00005-100, 30,000 iterations
sbatch scripts/train.slurm

# Custom case and iterations
sbatch --export=CASE_ID=BraTS-GLI-00005-100,ITERS=50000 scripts/train.slurm
```

**Output:**
- Checkpoints: `checkpoints/BraTS-GLI-00005-100/`
- Logs: `logs/train_<jobid>.log` (TensorBoard in `logs/`)

**Duration:** ~30-60 min on GPU

---

### 3. **compare.slurm** — Benchmark Methods
Compare Gaussian Splatting with traditional reconstruction methods.

```bash
# Default: all methods (gs, baseline_voxel, dvr, mip)
sbatch scripts/compare.slurm

# Custom case and methods
sbatch --export=CASE_ID=BraTS-GLI-00005-100,METHODS="gs,dvr,mip" scripts/compare.slurm

# Adjusted Gaussian count
sbatch --export=CASE_ID=BraTS-GLI-00005-100,N_GAUSS=50000 scripts/compare.slurm
```

**Output:**
- Results table: `outputs/comparisons/BraTS-GLI-00005-100_comparison.json`
- Logs: `logs/compare_<jobid>.log`

**Metrics compared:**
- PSNR (Peak Signal-to-Noise Ratio) in dB
- Memory usage (MB)
- Training time (seconds)
- Inference speed (FPS)

**Duration:** ~20-40 min depending on methods

---

### 4. **pipeline.slurm** — Full Workflow (All 5 Steps in One)
Run the complete 5-step pipeline in one job: data loading → preprocessing → training → evaluation → comparison.

**Use this for convenience** (instead of running explore, train, compare separately).

```bash
# Default: Full pipeline (all 5 steps)
sbatch scripts/pipeline.slurm

# Custom case with extended training
sbatch --export=CASE_ID=BraTS-GLI-00005-100,N_GAUSS=150000,ITERS=50000 scripts/pipeline.slurm
```

**What it runs (in order):**
1. **Data Loading** — Load NIfTI file from `../Data/`
2. **Preprocessing** — Extract slices, normalize to [0,1], create train/test split
3. **Training** — Optimize 100k Gaussians for 30,000 iterations (~30-60 min)
4. **Evaluation** — Render test slices, compute PSNR/SSIM/LPIPS
5. **Comparison** — Benchmark GS vs Baseline Voxel vs DVR vs MIP

**Output:**
- Checkpoints: `checkpoints/BraTS-GLI-00005-100/`
- Logs: `logs/train_<jobid>.log`, `logs/compare_<jobid>.log`
- Comparison results: `outputs/comparisons/BraTS-GLI-00005-100_comparison.json`

**Duration:** ~1-2 hours total (single job)

---

## Common Usage Patterns

### Pattern 1: Recommended Workflow (Step-by-Step)
Run scripts in logical order for best control and debugging:

```bash
# Step 1: Verify data exists (5-30 min)
sbatch scripts/explore.slurm

# Wait for completion, then:
# Step 2: Train Gaussian Splatting (30-60 min)
sbatch --export=CASE_ID=BraTS-GLI-00005-100 scripts/train.slurm

# Wait for completion, then:
# Step 3: Benchmark vs traditional methods (20-40 min)
sbatch --export=CASE_ID=BraTS-GLI-00005-100 scripts/compare.slurm

# Total time: 1-2 hours
# Advantages:
# • Can catch data issues early
# • Can stop between steps if needed
# • Can debug problems at each stage
```

### Pattern 2: All-in-One (For convenience)
If you know data is good, run everything in one command:

```bash
# Single command: runs all 5 steps automatically
# Step 1: Data Loading ✓
# Step 2: Preprocessing ✓
# Step 3: Training (30-60 min) ✓
# Step 4: Evaluation ✓
# Step 5: Comparison (20-40 min) ✓
# Total: 1-2 hours

sbatch scripts/pipeline.slurm

# Check status
squeue -u $USER

# View logs in real-time (will show all steps)
tail -f logs/train_<jobid>.log
tail -f logs/compare_<jobid>.log
```

### Pattern 3: Multiple Cases (Parallel)
Train multiple cases in parallel, then compare each:

```bash
# Step 1: Check data once
sbatch scripts/explore.slurm

# Step 2: Train multiple cases in parallel
for CASE in BraTS-GLI-00005-100 BraTS-GLI-00005-101 BraTS-GLI-00005-102; do
  sbatch --export=CASE_ID=$CASE scripts/train.slurm
done

# Wait for all to finish, then:
# Step 3: Compare each case
for CASE in BraTS-GLI-00005-100 BraTS-GLI-00005-101 BraTS-GLI-00005-102; do
  sbatch --export=CASE_ID=$CASE scripts/compare.slurm
done
```

### Pattern 4: Hyperparameter Sweep
Test different Gaussian counts on same case:

```bash
# Step 1: Check data
sbatch scripts/explore.slurm

# Step 2: Train with different parameters (parallel)
for N_GAUSS in 50000 100000 150000; do
  sbatch --export=CASE_ID=BraTS-GLI-00005-100,N_GAUSS=$N_GAUSS scripts/train.slurm
done

# Step 3: Compare all runs
sbatch --export=CASE_ID=BraTS-GLI-00005-100 scripts/compare.slurm
```

### Quick Decision Tree

```
Do you want to verify data first?
├─ YES (Recommended)
│  → sbatch scripts/explore.slurm
│  → sbatch scripts/train.slurm
│  → sbatch scripts/compare.slurm
└─ NO (Trust data is good)
   → sbatch scripts/pipeline.slurm  (all in one)
```

---

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER              # All your jobs
squeue -j <jobid>            # Specific job
```

### View logs in real-time
```bash
tail -f logs/train_<jobid>.log

# View TensorBoard (if using local)
tensorboard --logdir logs/
```

### Cancel job
```bash
scancel <jobid>
```

---

## SLURM Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--partition` | `gpu` | Request GPU node |
| `--gres=gpu:1` | 1 GPU | Request 1 GPU |
| `--cpus-per-task` | 8 | 8 CPU cores |
| `--mem` | 32G | 32 GB RAM |
| `--time` | 08:00:00 | 8 hour time limit |
| `--output` | `logs/...` | Log file location |

---

## Troubleshooting

**Job fails immediately:**
- Check logs: `cat logs/train_<jobid>.log`
- Verify data path: `ls ../Data/BraTS-GLI/`
- Ensure conda env is activated

**Out of memory errors:**
- Reduce `N_GAUSS` parameter
- Increase `--mem` allocation
- Reduce `--cpus-per-task`

**GPU not found:**
- Ensure `--partition=gpu` and `--gres=gpu:1`
- Check nvcc: `which nvcc`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Long queue times:**
- Try different `--partition` (check with `sinfo`)
- Reduce `--time` estimate
- Run during off-peak hours

---

## Environment Variables Reference

Export custom values with `-–export`:

```bash
sbatch --export=CASE_ID=MyCase,N_GAUSS=50000,ITERS=50000 scripts/train.slurm
```

### train.slurm
- `CASE_ID`: MRI case identifier (default: BraTS-GLI-00005-100)
- `ITERS`: Number of training iterations (default: 30000)

### compare.slurm
- `CASE_ID`: MRI case identifier (default: BraTS-GLI-00005-100)
- `METHODS`: Comma-separated methods: `gs,baseline_voxel,dvr,mip` (default: all)
- `N_GAUSS`: Number of Gaussians (default: 100000)

### explore.slurm
- `MODE`: `dataset`, `inspect`, or `pipeline` (default: dataset)
- `CASE`: MRI case identifier (default: BraTS-GLI-00005-100)
- `N_GAUSS`: Gaussians for pipeline mode (default: 1000)

### pipeline.slurm
- `CASE_ID`: MRI case identifier (default: BraTS-GLI-00005-100)
- `N_GAUSS`: Number of Gaussians (default: 100000)
- `ITERS`: Training iterations (default: 30000)

---

## See Also
- [WORKFLOW.md](../WORKFLOW.md) — Detailed pipeline documentation
- [train.py](../train.py) — Training script internals
- [compare.py](../compare.py) — Comparison script internals
