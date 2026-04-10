"""
Data exploration and pipeline smoke-test for BraTS-GLI + 3DGS.

Modes:
  inspect      -- Inspect a single case: shape, stats, label distribution, slice grid
  dataset      -- Audit all cases: count, shape distribution, split sizes
  pipeline     -- End-to-end smoke test: load → Gaussians → render → PSNR → backward
  sparse_eval  -- Visualise sparse vs dense slice quality after training

Usage:
  python explore.py --mode inspect --case BraTS-GLI-00005-100
  python explore.py --mode dataset
  python explore.py --mode pipeline --case BraTS-GLI-00005-100 --n_gaussians 1000
  python explore.py --mode sparse_eval --case BraTS-GLI-00005-100 --sparse_factor 5
  python explore.py --mode sparse_eval --case BraTS-GLI-00005-100 --sparse_factor 10
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data" / "BraTS-GLI" / "train"
OUTPUT_DIR = Path("outputs/explore")
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


# ---------------------------------------------------------------------------
# Mode: inspect
# ---------------------------------------------------------------------------

def mode_inspect(case_id: str) -> None:
    import nibabel as nib
    from src.utils.viz import plot_slice_grid, overlay_seg_on_slice

    case_dir = DATA_ROOT / case_id
    print(f"\n{'='*60}")
    print(f"Case: {case_id}")
    print(f"{'='*60}")

    slices, titles = [], []

    for mod in MODALITIES + ["seg"]:
        path = case_dir / f"{case_id}-{mod}.nii.gz"
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        vox = img.header.get_zooms()[:3]

        print(f"\n[{mod.upper()}]")
        print(f"  Shape:      {data.shape}")
        print(f"  Voxel size: {vox[0]:.2f} x {vox[1]:.2f} x {vox[2]:.2f} mm")

        if mod == "seg":
            labels, counts = np.unique(data.astype(int), return_counts=True)
            total = data.size
            print(f"  Labels:")
            label_names = {0: "Background", 1: "NCR", 2: "SNFH/Edema", 3: "ET", 4: "Tumor Core"}
            for lbl, cnt in zip(labels, counts):
                name = label_names.get(lbl, f"Label {lbl}")
                print(f"    {lbl} ({name}): {cnt:,} voxels ({cnt/total*100:.1f}%)")
        else:
            nonzero = data[data > 0]
            print(f"  Nonzero:    {len(nonzero):,} / {data.size:,} voxels ({len(nonzero)/data.size*100:.1f}%)")
            print(f"  Intensity:  min={data.min():.1f}, max={data.max():.1f}, "
                  f"mean={nonzero.mean():.1f}, std={nonzero.std():.1f}")
            # Normalize for visualization
            data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
            mid_ax = data.shape[2] // 2
            mid_cor = data.shape[1] // 2
            mid_sag = data.shape[0] // 2
            slices += [data_norm[:, :, mid_ax], data_norm[:, mid_cor, :], data_norm[mid_sag, :, :]]
            titles += [f"{mod} axial", f"{mod} coronal", f"{mod} sagittal"]

    # Save slice grid
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUTPUT_DIR / f"{case_id}_overview.png"
    plot_slice_grid(slices, titles, grid_path, cols=3)

    # Save segmentation overlay (axial midslice)
    t1c_path = case_dir / f"{case_id}-t1c.nii.gz"
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    t1c = nib.load(str(t1c_path)).get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
    mid = t1c.shape[2] // 2
    t1c_norm = (t1c[:, :, mid] - t1c.min()) / (t1c.max() - t1c.min() + 1e-8)
    overlay_path = OUTPUT_DIR / f"{case_id}_seg_overlay.png"
    overlay_seg_on_slice(t1c_norm, seg[:, :, mid].astype(int), overlay_path)

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Mode: dataset
# ---------------------------------------------------------------------------

def mode_dataset() -> None:
    import nibabel as nib

    print(f"\nScanning: {DATA_ROOT}")
    cases = sorted(DATA_ROOT.iterdir())
    valid, shapes, missing = [], [], []

    for case_dir in cases:
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        has_all = all((case_dir / f"{cid}-{m}.nii.gz").exists() for m in MODALITIES)
        if not has_all:
            missing.append(cid)
            continue
        valid.append(cid)

    print(f"\n{'='*60}")
    print(f"Total directories:  {len(cases)}")
    print(f"Valid cases:        {len(valid)} (all 4 modalities present)")
    print(f"Incomplete cases:   {len(missing)}")

    # Shape audit on first 10 cases
    print(f"\nShape audit (first 10 cases):")
    shape_counts: dict = {}
    for cid in valid[:10]:
        path = DATA_ROOT / cid / f"{cid}-t1c.nii.gz"
        shape = nib.load(str(path)).shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        print(f"  {cid}: {shape}")

    # Proposed splits
    n = len(valid)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    print(f"\nProposed splits (70/15/15):")
    print(f"  Train: {n_train}")
    print(f"  Val:   {n_val}")
    print(f"  Test:  {n_test}")
    print(f"  Total: {n}")


# ---------------------------------------------------------------------------
# Mode: pipeline
# ---------------------------------------------------------------------------

def mode_pipeline(case_id: str, n_gaussians: int) -> None:
    import torch
    from src.preprocessing.mri_loader import load_nifti, extract_slices
    from src.gaussian.gaussian_model import GaussianModel
    from src.rendering.volume_renderer import VolumeRenderer
    from src.utils.metrics import psnr, ssim_loss
    from src.utils.viz import plot_slice_grid, plot_gaussian_stats

    print(f"\n{'='*60}")
    print(f"Pipeline smoke test | case={case_id} | n_gaussians={n_gaussians}")
    print(f"{'='*60}")

    # 1. Load volume
    t0 = time.time()
    path = DATA_ROOT / case_id / f"{case_id}-t1c.nii.gz"
    volume = load_nifti(path)
    print(f"\n[1] Loaded volume: shape={volume.data.shape}, voxel={volume.voxel_size} mm "
          f"({time.time()-t0:.2f}s)")

    # 2. Initialize Gaussians
    t0 = time.time()
    gaussians = GaussianModel(max_scale=32.0)
    gaussians.initialize_from_volume(volume, num_gaussians=n_gaussians)
    print(f"[2] Initialized {gaussians.num_gaussians:,} Gaussians ({time.time()-t0:.2f}s)")

    # 3. Render 3 slices (axial, coronal, sagittal midplanes)
    H, W, D = volume.data.shape

    # Renderer with volume shape for dynamic axis-based sizing
    renderer = VolumeRenderer(volume_shape=(H, W, D))

    rendered_slices, gt_slices, titles = [], [], []
    psnr_vals = []

    for axis, label, mid in [
        (2, "axial",     D // 2),
        (1, "coronal",   W // 2),
        (0, "sagittal",  H // 2),
    ]:
        t0 = time.time()
        with torch.no_grad():
            out = renderer(gaussians, axis=axis, slice_idx=mid)
        rendered = out["render"].squeeze().numpy()

        gt = extract_slices(volume, axis)[mid]
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

        p = psnr(
            torch.from_numpy(rendered).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0),
        )
        psnr_vals.append(p)
        print(f"[3] Render {label} slice {mid}: PSNR={p:.2f} dB ({time.time()-t0:.2f}s)")

        rendered_slices.append(rendered)
        gt_slices.append(gt_norm)
        titles += [f"Render {label}", f"GT {label}"]

    # 4. Backward pass (test autograd on Gaussian parameters directly)
    t0 = time.time()
    gaussians.train()
    cfg_obj = type("cfg", (), {"gaussian": type("g", (), {
        "position_lr": 1.6e-4, "feature_lr": 2.5e-3, "opacity_lr": 0.05,
        "scaling_lr": 5e-3, "rotation_lr": 1e-3,
    })()})()
    optimizer = torch.optim.Adam(gaussians.get_optimizer_params(cfg_obj))
    # Simple parameter-level loss to verify autograd graph is intact
    dummy_loss = (gaussians.get_intensity - 0.5).pow(2).mean()
    dummy_loss.backward()
    grad_norm = gaussians._intensity.grad.norm().item()
    optimizer.step()
    print(f"[4] Backward pass OK, grad_norm={grad_norm:.4f} ({time.time()-t0:.2f}s)")

    # 5. Save comparison figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    interleaved = []
    for r, g in zip(rendered_slices, gt_slices):
        interleaved += [r, g]
    plot_slice_grid(interleaved, titles, OUTPUT_DIR / "pipeline_test.png", cols=2)
    plot_gaussian_stats(gaussians, OUTPUT_DIR / "gaussian_stats.png")

    print(f"\nAll checks passed!")
    print(f"  Mean PSNR: {np.mean(psnr_vals):.2f} dB (expected low at init, ~10-15 dB)")
    print(f"  Outputs:   {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mode: sparse_eval
# ---------------------------------------------------------------------------

def mode_sparse_eval(case_id: str, sparse_factor: int) -> None:
    """
    After training with sparse_factor N, load the checkpoint and produce:
      1. Quantitative table: seen PSNR vs unseen PSNR across all axial slices
      2. Visual grid: seen slice (render vs GT) and unseen slice (render vs GT)
      3. PSNR-per-slice plot to show where interpolation fails
    """
    import torch
    from src.preprocessing.mri_loader import load_nifti, extract_slices
    from src.gaussian.gaussian_model import GaussianModel
    from src.rendering.volume_renderer import VolumeRenderer
    from src.utils.metrics import psnr
    from src.utils.viz import plot_slice_grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train import build_slice_sets

    suffix = f"_sparse{sparse_factor}" if sparse_factor > 0 else "_dense"
    ckpt_dir = Path("checkpoints") / (case_id + suffix)

    ckpts = sorted(ckpt_dir.glob("iteration_*.ply"))
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        print("Run training first:  python train.py case_id=... sparse_factor=...")
        return

    ckpt = ckpts[-1]
    print(f"\n{'='*60}")
    print(f"Sparse Eval | case={case_id} | sparse_factor={sparse_factor}")
    print(f"Checkpoint:  {ckpt}")
    print(f"{'='*60}")

    # Load volume
    nifti_path = DATA_ROOT / case_id / f"{case_id}-t1c.nii.gz"
    volume = load_nifti(nifti_path)
    H, W, D = volume.data.shape
    all_slices = extract_slices(volume, axis=2)
    gt_min, gt_max = volume.data.min(), volume.data.max()

    # Load model
    gaussians = GaussianModel(max_scale=float(max(H, W, D)) * 0.1)
    gaussians.load_ply(ckpt)
    gaussians.eval()

    # Renderer with volume shape for dynamic axis-based sizing
    renderer = VolumeRenderer(volume_shape=(H, W, D))
    training_axial, holdout_axial = build_slice_sets(D, sparse_factor)

    # ── Per-slice PSNR ───────────────────────────────────────────────────
    seen_psnrs, unseen_psnrs = [], []
    all_psnrs = []

    print("\nEvaluating all axial slices...")
    with torch.no_grad():
        for idx in range(D):
            out = renderer(gaussians, axis=2, slice_idx=idx)
            gt = all_slices[idx]
            gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)
            gt_t = torch.from_numpy(gt_norm[None, None].astype(np.float32))
            p = psnr(out["render"].cpu().unsqueeze(0), gt_t)
            p = p if not math.isinf(p) else 0.0
            all_psnrs.append(p)
            if idx in set(training_axial):
                seen_psnrs.append(p)
            else:
                unseen_psnrs.append(p)

    print(f"\n{'─'*40}")
    print(f"{'Metric':<30} {'Value':>8}")
    print(f"{'─'*40}")
    print(f"{'Seen slices mean PSNR':<30} {np.mean(seen_psnrs):>7.2f} dB")
    print(f"{'Unseen slices mean PSNR':<30} {np.mean(unseen_psnrs):>7.2f} dB  ← key")
    print(f"{'Gap (seen - unseen)':<30} {np.mean(seen_psnrs)-np.mean(unseen_psnrs):>7.2f} dB")
    print(f"{'─'*40}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = OUTPUT_DIR / f"{case_id}_sparse{sparse_factor}"

    # ── PSNR-per-slice plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(D), all_psnrs, color="steelblue", linewidth=0.8, label="PSNR per slice")
    for idx in training_axial:
        ax.axvline(x=idx, color="red", alpha=0.3, linewidth=0.5)
    ax.axhline(y=np.mean(seen_psnrs),   color="red",   linestyle="--", label=f"Seen mean {np.mean(seen_psnrs):.1f} dB")
    ax.axhline(y=np.mean(unseen_psnrs), color="green", linestyle="--", label=f"Unseen mean {np.mean(unseen_psnrs):.1f} dB")
    ax.set_xlabel("Axial slice index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{case_id} | sparse_factor={sparse_factor} | Red lines = training slices")
    ax.legend()
    plt.tight_layout()
    psnr_plot_path = f"{out_prefix}_psnr_per_slice.png"
    plt.savefig(psnr_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {psnr_plot_path}")

    # ── Visual comparison grid ────────────────────────────────────────────
    # Pick 4 seen + 4 unseen slices spread across the volume
    sample_seen    = training_axial[::max(1, len(training_axial) // 4)][:4]
    sample_unseen  = holdout_axial[::max(1, len(holdout_axial) // 4)][:4]

    grid_slices, grid_titles = [], []
    with torch.no_grad():
        for label, indices in [("SEEN", sample_seen), ("UNSEEN", sample_unseen)]:
            for idx in indices:
                out = renderer(gaussians, axis=2, slice_idx=idx)
                rendered = out["render"].squeeze().numpy()
                gt = all_slices[idx]
                gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)
                p = all_psnrs[idx]
                grid_slices += [rendered, gt_norm]
                grid_titles += [f"{label} s{idx} Render ({p:.1f}dB)", f"{label} s{idx} GT"]

    grid_path = f"{out_prefix}_visual_comparison.png"
    plot_slice_grid(grid_slices, grid_titles, grid_path, cols=4)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"\nInterpretation:")
    print(f"  Gap < 2 dB  → 3DGS interpolates well between sparse slices")
    print(f"  Gap > 5 dB  → 3DGS struggles, may need more training or denser input")


def main():
    parser = argparse.ArgumentParser(description="BraTS-GLI exploration tool")
    parser.add_argument("--mode", choices=["inspect", "dataset", "pipeline", "sparse_eval"], required=True)
    parser.add_argument("--case", default="BraTS-GLI-00005-100",
                        help="Case ID (for inspect/pipeline/sparse_eval modes)")
    parser.add_argument("--n_gaussians", type=int, default=1000,
                        help="Number of Gaussians for pipeline smoke test")
    parser.add_argument("--sparse_factor", type=int, default=5,
                        help="Sparse factor used during training (for sparse_eval mode)")
    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent))

    if args.mode == "inspect":
        mode_inspect(args.case)
    elif args.mode == "dataset":
        mode_dataset()
    elif args.mode == "pipeline":
        mode_pipeline(args.case, args.n_gaussians)
    elif args.mode == "sparse_eval":
        mode_sparse_eval(args.case, args.sparse_factor)


if __name__ == "__main__":
    main()
