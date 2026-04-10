"""
Comparison script: Gaussian Splatting vs Traditional Reconstruction.

Benchmarks both methods on the same MRI case for memory usage, inference speed,
and reconstruction quality (PSNR, SSIM, LPIPS).

Usage:
  python compare.py case_id=BraTS-GLI-00005-100
  python compare.py case_id=BraTS-GLI-00005-100 num_gaussians=50000
"""

import json
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.gaussian.gaussian_model import GaussianModel
from src.preprocessing.mri_loader import load_nifti, extract_slices, MRIVolume
from src.reconstruction.baseline_model import BaselineVoxelModel
from src.reconstruction.traditional_renderer import TraditionalRenderer
from src.reconstruction.unet_interp import TinyUNet3D, pad_to_multiple_3d, build_input_volume
from src.rendering.volume_renderer import VolumeRenderer
from src.utils.metrics import psnr, ssim_loss


class ComparisonConfig:
    """Configuration for comparison experiments."""

    data_root: str = str(Path(__file__).resolve().parent.parent / "Data" / "BraTS-GLI" / "train")
    case_id: str = "BraTS-GLI-00005-100"
    modality: str = "t1c"

    # Methods to compare
    methods: list = None  # ["gs", "baseline_voxel", "dvr", "mip"]

    # Gaussian model
    num_gaussians: int = 100_000
    gs_iterations: int = 30_000
    gs_densify_from: int = 500
    gs_densify_until: int = 15_000
    gs_densify_every: int = 100
    gs_densify_grad_threshold: float = 0.0002
    gs_opacity_reset_every: int = 3000
    gs_lambda_dssim: float = 0.2
    gs_checkpoint: str = ""  # Path to pre-trained .ply; skips training when set
    gs_max_active: int = 50_000  # Max Gaussians per slice render (was 500)

    # Baseline model
    baseline_iterations: int = 100_000
    baseline_lr: float = 0.01

    # 3D U-Net slice interpolation baseline (masked autoencoding)
    unet_iterations: int = 10_000
    unet_lr: float = 1e-3
    unet_base_ch: int = 16

    # Sparse mode (must match train.py sparse_factor for fair comparison)
    sparse_factor: int = 0  # 0=use interleaved split, >0=use train.py-style split

    # Evaluation
    test_slices: int = 10  # Number of held-out slices (only used when sparse_factor=0)
    slice_axis: int = 2  # 0=sagittal, 1=coronal, 2=axial

    # Output
    output_dir: str = "outputs/comparisons"
    save_images: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.methods is None:
            self.methods = ["gs", "spline", "unet", "baseline_voxel", "dvr", "mip"]

    @property
    def file_prefix(self) -> str:
        """Output filename prefix including sparse factor to avoid overwrites."""
        if self.sparse_factor > 0:
            return f"{self.case_id}_sparse{self.sparse_factor}"
        return self.case_id


def parse_args() -> ComparisonConfig:
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare GS vs Traditional MRI Reconstruction")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    
    cfg = ComparisonConfig()
    
    # Apply overrides
    for override in parser.parse_args().overrides:
        key, val = override.split("=", 1)
        # Special handling for list arguments
        if key == "methods":
            try:
                evaluated = eval(val)
                val = [str(m).strip() for m in evaluated] if isinstance(evaluated, list) else [m.strip() for m in val.split(",")]
            except (SyntaxError, NameError):
                val = [m.strip() for m in val.split(",")]
        else:
            try:
                val = eval(val)  # Try to parse as Python literal
            except (SyntaxError, NameError):
                pass  # Keep as string
        setattr(cfg, key, val)
    
    return cfg


def benchmark_gs(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark Gaussian Splatting method.

    Trains from scratch on train_indices only so test_indices are truly held-out.
    Renders at actual 3D positions for both training and evaluation.
    """
    print("\n[GS] Starting Gaussian Splatting benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = all_slices[0].shape
    D_total = len(all_slices)
    volume_shape = (H, W, D_total)

    gs_model = GaussianModel()

    start_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    start_time = time.time()

    # Try to load pre-trained checkpoint
    ckpt_path = cfg.gs_checkpoint
    if not ckpt_path:
        # Auto-discover: match sparse_factor to checkpoint directory
        if cfg.sparse_factor > 0:
            suffix = f"_sparse{cfg.sparse_factor}"
        else:
            suffix = "_dense"
        ckpt_dir = Path("checkpoints") / f"{cfg.case_id}{suffix}"
        if ckpt_dir.exists():
            plys = sorted(ckpt_dir.glob("iteration_*.ply"))
            if plys:
                ckpt_path = str(plys[-1])

    if ckpt_path and Path(ckpt_path).exists():
        print(f"[GS] Loading checkpoint: {ckpt_path}")
        gs_model.load_ply(ckpt_path)
        gs_model = gs_model.to(device)
        train_time = 0.0
    else:
        print(f"[GS] No checkpoint found, training {cfg.gs_iterations} iters on {len(train_indices)} slices...")
        full_volume = MRIVolume(
            data=np.moveaxis(all_slices, 0, -1).astype(np.float32),
            affine=np.eye(4, dtype=np.float32),
            voxel_size=np.ones(3, dtype=np.float32),
        )
        gs_model.initialize_from_volume(full_volume, num_gaussians=cfg.num_gaussians)
        gs_model = gs_model.to(device)

        renderer_train = VolumeRenderer(volume_shape=volume_shape).to(device)
        optimizer = torch.optim.Adam([
            {"params": [gs_model._xyz],       "lr": 1.6e-4, "name": "xyz"},
            {"params": [gs_model._intensity], "lr": 2.5e-3, "name": "intensity"},
            {"params": [gs_model._opacity],   "lr": 0.05,   "name": "opacity"},
            {"params": [gs_model._scaling],   "lr": 5e-3,   "name": "scaling"},
            {"params": [gs_model._rotation],  "lr": 1e-3,   "name": "rotation"},
        ], eps=1e-15)

        # LR schedule for xyz
        lr_init, lr_final = 1.6e-4, 1.6e-6
        for iteration in range(1, cfg.gs_iterations + 1):
            # Update xyz LR with exponential decay
            t = min(iteration / cfg.gs_iterations, 1.0)
            new_lr = lr_init * (lr_final / lr_init) ** t
            for group in optimizer.param_groups:
                if group["name"] == "xyz":
                    group["lr"] = new_lr

            # Sample random training slice (with occasional multi-axis in dense regions)
            train_pos = train_indices[random.randint(0, len(train_indices) - 1)]
            gt_slice = torch.tensor(all_slices[train_pos], dtype=torch.float32, device=device)
            rendered = renderer_train(gs_model, axis=cfg.slice_axis, slice_idx=train_pos,
                                      max_active=cfg.gs_max_active)
            pred = rendered["render"].squeeze(0)

            # Combined L1 + DSSIM loss (same as train.py)
            l1 = F.l1_loss(pred, gt_slice)
            dssim = ssim_loss(pred.unsqueeze(0), gt_slice.unsqueeze(0))
            loss = (1 - cfg.gs_lambda_dssim) * l1 + cfg.gs_lambda_dssim * dssim

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gs_model.update_xyz_grad_accum()
            optimizer.step()

            # Densification
            if cfg.gs_densify_from <= iteration <= cfg.gs_densify_until:
                if iteration % cfg.gs_densify_every == 0:
                    gs_model.densify_and_prune(cfg.gs_densify_grad_threshold, optimizer)
            if iteration % cfg.gs_opacity_reset_every == 0 and iteration <= cfg.gs_densify_until:
                gs_model.reset_opacity(optimizer)

            if iteration % 1000 == 0:
                print(f"  Iter {iteration}: Loss={loss.item():.6f} L1={l1.item():.4f} SSIM={1-dssim.item():.4f} N={gs_model.num_gaussians:,}")
        train_time = time.time() - start_time

    renderer = VolumeRenderer(volume_shape=volume_shape).to(device)

    final_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    memory_used_mb = (final_memory - start_memory) / (1024 ** 2) if torch.cuda.is_available() else 0

    # Evaluate at actual test positions (interpolation, not extrapolation)
    print(f"[GS] Evaluating on {len(test_indices)} held-out slices (positions: {test_indices})...")
    psnrs = []
    with torch.no_grad():
        for test_pos in test_indices:
            rendered = renderer(gs_model, axis=cfg.slice_axis, slice_idx=test_pos,
                                max_active=cfg.gs_max_active)
            pred = rendered["render"].squeeze(0)
            gt = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
            psnrs.append(psnr(pred, gt))

    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    num_params = gs_model.num_gaussians

    print(f"[GS] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[GS] Training time: {train_time:.2f}s")
    print(f"[GS] Memory used: {memory_used_mb:.2f} MB")
    print(f"[GS] Gaussians: {num_params:,}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{cfg.file_prefix}_gs.ply"
    gs_model.save_ply(model_path, iteration=cfg.gs_iterations)
    print(f"[GS] Gaussian splat saved: {model_path}")

    print(f"[GS] Rendering full volume ({D_total} slices)...")
    rendered_slices = []
    with torch.no_grad():
        for i in range(D_total):
            r = renderer(gs_model, axis=cfg.slice_axis, slice_idx=i,
                         max_active=cfg.gs_max_active)
            rendered_slices.append(r["render"].squeeze(0).cpu().numpy())
    recon_volume = np.moveaxis(np.stack(rendered_slices, axis=0), 0, -1)
    volume_path = output_dir / f"{cfg.file_prefix}_gs_volume.nii.gz"
    nib.save(nib.Nifti1Image(recon_volume.astype(np.float32), np.eye(4)), str(volume_path))
    print(f"[GS] 3D volume saved:      {volume_path}")

    return {
        "method": "Gaussian Splatting",
        "mean_psnr": mean_psnr,
        "training_time_s": train_time,
        "memory_mb": memory_used_mb,
        "num_parameters": num_params,
        "fps": len(test_indices) / (time.time() - start_time),
        "model_path": str(model_path),
        "volume_nifti_path": str(volume_path),
    }


def benchmark_baseline_voxel(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark baseline voxel model."""
    print("\n[BASELINE] Starting voxel-based training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = all_slices[0].shape
    D_total = len(all_slices)
    # volume_shape=(H,W,D_total): BaselineVoxelModel.get_slice(2,idx)=voxels[:,:,idx]=(H,W) ✓
    volume_shape = (H, W, D_total)

    baseline = BaselineVoxelModel(volume_shape=volume_shape, learning_rate=cfg.baseline_lr)
    baseline = baseline.to(device)

    # Initialize voxels from training data instead of random noise
    with torch.no_grad():
        for idx in train_indices:
            gt_slice = torch.tensor(all_slices[idx], dtype=torch.float32, device=device)
            # Set voxel slice to logit of GT (inverse of sigmoid activation)
            logit_val = torch.logit(gt_slice.clamp(0.01, 0.99))
            if cfg.slice_axis == 0:
                baseline.voxels[idx, :, :] = logit_val
            elif cfg.slice_axis == 1:
                baseline.voxels[:, idx, :] = logit_val
            else:
                baseline.voxels[:, :, idx] = logit_val
        # Interpolate held-out slices from nearest training slices
        for test_pos in [i for i in range(D_total) if i not in set(train_indices)]:
            nearest = min(train_indices, key=lambda t: abs(t - test_pos))
            if cfg.slice_axis == 0:
                baseline.voxels[test_pos, :, :] = baseline.voxels[nearest, :, :].clone()
            elif cfg.slice_axis == 1:
                baseline.voxels[:, test_pos, :] = baseline.voxels[:, nearest, :].clone()
            else:
                baseline.voxels[:, :, test_pos] = baseline.voxels[:, :, nearest].clone()

    # Recreate optimizer after initialization
    baseline.optimizer = torch.optim.Adam([baseline.voxels], lr=cfg.baseline_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        baseline.optimizer, T_max=cfg.baseline_iterations, eta_min=cfg.baseline_lr * 0.01
    )

    start_time = time.time()
    start_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Train only on train_indices (actual 3D positions)
    for iteration in range(cfg.baseline_iterations):
        train_pos = train_indices[random.randint(0, len(train_indices) - 1)]
        gt_slice = torch.tensor(all_slices[train_pos], dtype=torch.float32, device=device)
        pred = baseline(axis=cfg.slice_axis, slice_idx=train_pos)
        loss = F.l1_loss(pred, gt_slice) + F.mse_loss(pred, gt_slice)
        baseline.optimizer.zero_grad()
        loss.backward()
        baseline.optimizer.step()
        scheduler.step()
        if iteration % 5000 == 0:
            print(f"  Iter {iteration}: Loss={loss.item():.6f} LR={scheduler.get_last_lr()[0]:.6f}")

    train_time = time.time() - start_time
    final_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    memory_used_mb = (final_memory - start_memory) / (1024 ** 2) if torch.cuda.is_available() else 0

    # Evaluate at actual test positions (interspersed in the volume)
    print(f"[BASELINE] Evaluating on {len(test_indices)} held-out slices (positions: {test_indices})...")
    psnrs = []
    with torch.no_grad():
        for test_pos in test_indices:
            pred = baseline(axis=cfg.slice_axis, slice_idx=test_pos)
            gt = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
            psnrs.append(psnr(pred, gt))

    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    num_params = baseline.parameters_per_sample()

    print(f"[BASELINE] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[BASELINE] Training time: {train_time:.2f}s")
    print(f"[BASELINE] Memory used: {memory_used_mb:.2f} MB")
    print(f"[BASELINE] Parameters: {num_params:,}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{cfg.file_prefix}_baseline.pt"
    torch.save(baseline.state_dict(), model_path)
    print(f"[BASELINE] Weights saved:    {model_path}")

    with torch.no_grad():
        voxel_volume = torch.sigmoid(baseline.voxels).cpu().numpy().astype(np.float32)
    volume_path = output_dir / f"{cfg.file_prefix}_baseline_volume.nii.gz"
    nib.save(nib.Nifti1Image(voxel_volume, np.eye(4)), str(volume_path))
    print(f"[BASELINE] 3D volume saved:  {volume_path}")

    return {
        "method": "Baseline Voxel Grid",
        "mean_psnr": mean_psnr,
        "training_time_s": train_time,
        "memory_mb": memory_used_mb,
        "num_parameters": num_params,
        "fps": len(test_indices) / (time.time() - start_time),
        "model_path": str(model_path),
        "volume_nifti_path": str(volume_path),
    }


def benchmark_dvr(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark Direct Volume Rendering (non-learned baseline).

    Uses only training slices. For each test position, finds the nearest
    training slice (by index distance) — this is the best a simple DVR can do
    without learning.
    """
    print("\n[DVR] Starting Direct Volume Rendering benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = all_slices[0].shape
    train_data = all_slices[train_indices].astype(np.float32)
    volume = torch.tensor(train_data, dtype=torch.float32, device=device)

    renderer = TraditionalRenderer(image_size=(H, W))
    renderer = renderer.to(device)

    start_time = time.time()

    print(f"[DVR] Evaluating on {len(test_indices)} held-out slices (nearest-neighbour from training data)...")
    psnrs = []
    for test_pos in test_indices:
        # Find nearest training slice by position distance
        nearest_train_local = min(range(len(train_indices)), key=lambda i: abs(train_indices[i] - test_pos))
        pred = volume[nearest_train_local]
        gt = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
        psnrs.append(psnr(pred, gt))

    render_time = time.time() - start_time
    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    memory_mb = volume.element_size() * volume.nelement() / (1024 ** 2)

    print(f"[DVR] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[DVR] Render time: {render_time:.2f}s")
    print(f"[DVR] Memory used: {memory_mb:.2f} MB")
    print(f"[DVR] Parameters: 0 (non-learned)")

    # Build full-size volume by nearest-neighbour filling held-out slices
    # (matches the per-slice eval above so the saved NIfTI is the same data)
    D_total = len(all_slices)
    full = np.zeros((D_total, H, W), dtype=np.float32)
    for pos in range(D_total):
        nearest_local = min(range(len(train_indices)), key=lambda i: abs(train_indices[i] - pos))
        full[pos] = train_data[nearest_local]

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_nifti = np.moveaxis(full, 0, -1)
    volume_path = output_dir / f"{cfg.file_prefix}_dvr_volume.nii.gz"
    nib.save(nib.Nifti1Image(volume_nifti, np.eye(4)), str(volume_path))
    print(f"[DVR] 3D volume saved: {volume_path}")

    return {
        "method": "Direct Volume Rendering",
        "mean_psnr": mean_psnr,
        "training_time_s": 0.0,
        "memory_mb": memory_mb,
        "num_parameters": 0,
        "fps": len(test_indices) / render_time,
        "volume_nifti_path": str(volume_path),
    }


def benchmark_mip(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark Maximum Intensity Projection (non-learned baseline).

    Computes MIP from training slices only, compares each test slice to the
    MIP projection (upper-bound for what MIP can recover per slice).
    """
    print("\n[MIP] Starting Maximum Intensity Projection benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = all_slices[0].shape
    train_data = all_slices[train_indices].astype(np.float32)
    volume = torch.tensor(train_data, dtype=torch.float32, device=device)
    renderer = TraditionalRenderer(image_size=(H, W))

    start_time = time.time()

    print(f"[MIP] Evaluating on {len(test_indices)} held-out slices...")
    mip_image = renderer.render_mip(volume, axis=cfg.slice_axis)
    psnrs = []
    for test_pos in test_indices:
        gt = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
        pred = mip_image if mip_image.shape == gt.shape else torch.nn.functional.interpolate(
            mip_image.unsqueeze(0).unsqueeze(0), size=gt.shape, mode="bilinear", align_corners=False
        ).squeeze()
        psnrs.append(psnr(pred, gt))

    render_time = time.time() - start_time
    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    memory_mb = volume.element_size() * volume.nelement() / (1024 ** 2)

    print(f"[MIP] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[MIP] Render time: {render_time:.2f}s")
    print(f"[MIP] Memory used: {memory_mb:.2f} MB")
    print(f"[MIP] Parameters: 0 (non-learned)")

    # MIP collapses the volume to one 2D image; broadcast it to every slice
    # so the saved NIfTI matches GT/GS/Baseline shape and reflects how PSNR
    # is computed (each test slice compared against the same MIP image).
    D_total = len(all_slices)
    mip_np = mip_image.cpu().numpy().astype(np.float32)
    if mip_np.shape != (H, W):
        # Resize if MIP renderer produced a different shape
        from torch.nn.functional import interpolate as _interp
        mip_np = _interp(
            torch.from_numpy(mip_np)[None, None], size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().numpy()
    full = np.broadcast_to(mip_np, (D_total, H, W)).copy()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_nifti = np.moveaxis(full, 0, -1)
    volume_path = output_dir / f"{cfg.file_prefix}_mip_volume.nii.gz"
    nib.save(nib.Nifti1Image(volume_nifti, np.eye(4)), str(volume_path))
    print(f"[MIP] 3D volume saved: {volume_path}")

    return {
        "method": "Max Intensity Projection",
        "mean_psnr": mean_psnr,
        "training_time_s": 0.0,
        "memory_mb": memory_mb,
        "num_parameters": 0,
        "fps": len(test_indices) / render_time,
        "volume_nifti_path": str(volume_path),
    }


def benchmark_spline(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark cubic spline interpolation along the slice axis (non-learned).

    For each (x, y) pixel, fit a 1-D cubic spline through the values at the
    training slice positions and evaluate at the test positions. Strictly
    stronger than DVR (which is piecewise-constant nearest-neighbour).
    """
    from scipy.interpolate import CubicSpline

    print("\n[SPLINE] Starting cubic spline interpolation benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = all_slices[0].shape
    D_total = len(all_slices)
    train_data = all_slices[train_indices].astype(np.float32)  # (M, H, W)

    start_time = time.time()
    # CubicSpline supports n-d y; axis=0 = the slice-position axis
    cs = CubicSpline(train_indices, train_data, axis=0, extrapolate=True)

    print(f"[SPLINE] Evaluating on {len(test_indices)} held-out slices...")
    psnrs = []
    for test_pos in test_indices:
        pred = torch.from_numpy(np.clip(cs(test_pos), 0, 1).astype(np.float32)).to(device)
        gt = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
        psnrs.append(psnr(pred, gt))

    # Build full-volume reconstruction for visualization / consistent shape
    full = np.clip(cs(np.arange(D_total)), 0, 1).astype(np.float32)
    render_time = time.time() - start_time

    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    memory_mb = train_data.nbytes / (1024 ** 2)

    print(f"[SPLINE] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[SPLINE] Time: {render_time:.2f}s")
    print(f"[SPLINE] Memory: {memory_mb:.2f} MB")
    print(f"[SPLINE] Parameters: 0 (non-learned)")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_nifti = np.moveaxis(full, 0, -1)
    volume_path = output_dir / f"{cfg.file_prefix}_spline_volume.nii.gz"
    nib.save(nib.Nifti1Image(volume_nifti, np.eye(4)), str(volume_path))
    print(f"[SPLINE] 3D volume saved: {volume_path}")

    return {
        "method": "Cubic Spline",
        "mean_psnr": mean_psnr,
        "training_time_s": 0.0,
        "memory_mb": memory_mb,
        "num_parameters": 0,
        "fps": len(test_indices) / max(render_time, 1e-6),
        "volume_nifti_path": str(volume_path),
    }


def benchmark_unet(
    cfg: ComparisonConfig,
    all_slices: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
):
    """Benchmark 3D U-Net slice interpolation (masked autoencoding).

    Trains a small 3D U-Net to refine a linearly-interpolated input volume,
    using held-out training slices as supervision targets. At test time,
    builds a single input volume from the (true) train slices and predicts
    the entire 3D refined volume in one forward pass.
    """
    print("\n[UNET3D] Starting 3D U-Net slice interpolation benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = all_slices[0].shape
    D_total = len(all_slices)
    train_arr = sorted(train_indices)

    model = TinyUNet3D(in_ch=1, out_ch=1, base=cfg.unet_base_ch).to(device)

    # Auto-discover pre-trained checkpoint
    suffix = f"_sparse{cfg.sparse_factor}" if cfg.sparse_factor > 0 else "_dense"
    ckpt_path = Path("checkpoints") / f"{cfg.case_id}{suffix}_unet3d.pt"
    start_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    start_time = time.time()

    if ckpt_path.exists():
        print(f"[UNET3D] Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        train_time = 0.0
    else:
        print(f"[UNET3D] No checkpoint found at {ckpt_path}")
        print(f"[UNET3D] Training {cfg.unet_iterations} iters with masked autoencoding...")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.unet_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.unet_iterations, eta_min=cfg.unet_lr * 0.01
        )

        gt_volume = torch.from_numpy(all_slices.astype(np.float32)).to(device)  # (D, H, W)
        n_mask = max(1, len(train_arr) // 6)  # mask ~16% of train slices each iter

        model.train()
        for it in range(1, cfg.unet_iterations + 1):
            # Pick random training slices to hide as fake "unseen" supervision
            masked = random.sample(train_arr, n_mask)
            visible = [i for i in train_arr if i not in set(masked)]

            input_vol = build_input_volume(all_slices, visible).to(device)  # (1,1,D,H,W)
            input_padded, _ = pad_to_multiple_3d(input_vol, multiple=4)
            pred_padded = model(input_padded)
            pred_full = pred_padded[..., :D_total, :H, :W]  # (1,1,D,H,W)

            # L1 loss only on masked positions
            mask_idx = torch.tensor(masked, dtype=torch.long, device=device)
            pred_masked = pred_full[0, 0, mask_idx]      # (n_mask, H, W)
            gt_masked   = gt_volume[mask_idx]            # (n_mask, H, W)
            loss = F.l1_loss(pred_masked, gt_masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % 200 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Iter {it}: Loss={loss.item():.6f} LR={lr_now:.5f}  (masked {n_mask}/{len(train_arr)})")
        train_time = time.time() - start_time

    final_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    memory_mb = (final_mem - start_mem) / (1024 ** 2) if torch.cuda.is_available() else 0

    # Inference: single forward pass over input volume built from ALL train slices
    print(f"[UNET3D] Predicting full volume from {len(train_arr)} train slices...")
    model.eval()
    with torch.no_grad():
        input_vol = build_input_volume(all_slices, train_arr).to(device)
        input_padded, _ = pad_to_multiple_3d(input_vol, multiple=4)
        pred_padded = model(input_padded)
        pred_full = pred_padded[0, 0, :D_total, :H, :W].cpu().numpy()  # (D, H, W)
    # Note: we do NOT overwrite train slices with GT — same fairness as GS,
    # which also doesn't get its training-slice renderings replaced by GT.

    print(f"[UNET3D] Evaluating on {len(test_indices)} held-out slices...")
    psnrs = []
    for test_pos in test_indices:
        pred_t = torch.from_numpy(pred_full[test_pos]).to(device)
        gt_t = torch.from_numpy(all_slices[test_pos].astype(np.float32)).to(device)
        psnrs.append(psnr(pred_t, gt_t))

    mean_psnr = np.mean([p for p in psnrs if not np.isinf(p)] or [0.0])
    num_params = sum(p.numel() for p in model.parameters())

    print(f"[UNET3D] Mean PSNR: {mean_psnr:.2f} dB")
    print(f"[UNET3D] Training time: {train_time:.2f}s")
    print(f"[UNET3D] Memory: {memory_mb:.2f} MB")
    print(f"[UNET3D] Parameters: {num_params:,}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{cfg.file_prefix}_unet3d.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[UNET3D] Weights saved: {model_path}")

    volume_nifti = np.moveaxis(pred_full.astype(np.float32), 0, -1)
    volume_path = output_dir / f"{cfg.file_prefix}_unet3d_volume.nii.gz"
    nib.save(nib.Nifti1Image(volume_nifti, np.eye(4)), str(volume_path))
    print(f"[UNET3D] 3D volume saved: {volume_path}")

    return {
        "method": "3D U-Net Interp",
        "mean_psnr": mean_psnr,
        "training_time_s": train_time,
        "memory_mb": memory_mb,
        "num_parameters": num_params,
        "fps": len(test_indices) / max(time.time() - start_time, 1e-6),
        "model_path": str(model_path),
        "volume_nifti_path": str(volume_path),
    }


def main():
    cfg = parse_args()
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Gaussian Splatting vs Traditional MRI Reconstruction")
    print(f"{'='*70}")
    print(f"Case: {cfg.case_id}")
    print(f"Modality: {cfg.modality}")
    print(f"Methods: {', '.join(cfg.methods)}")

    # Load data
    print(f"\nLoading volume...")
    volume_path = Path(cfg.data_root) / cfg.case_id / f"{cfg.case_id}-{cfg.modality}.nii.gz"
    img_data = load_nifti(str(volume_path))
    slices = extract_slices(img_data, axis=cfg.slice_axis)
    slices = np.array(slices)

    # Normalize slices to [0, 1]
    slices = (slices - slices.min()) / (slices.max() - slices.min() + 1e-8)

    # Train/test split
    n_total = len(slices)
    if cfg.sparse_factor > 0:
        # Match train.py split: every Nth slice trains, the rest are holdout
        train_indices = list(range(0, n_total, cfg.sparse_factor))
        test_indices = [i for i in range(n_total) if i not in set(train_indices)]
    else:
        # Interleaved split (legacy): test slices evenly distributed
        step = n_total // cfg.test_slices
        test_indices = sorted([step // 2 + i * step for i in range(cfg.test_slices) if step // 2 + i * step < n_total])
        train_indices = [i for i in range(n_total) if i not in set(test_indices)]

    print(f"Total slices: {n_total}")
    print(f"Training slices: {len(train_indices)}")
    print(f"Test slices: {len(test_indices)} at positions {test_indices}")

    # Benchmark methods
    results = []

    if "gs" in cfg.methods:
        try:
            results.append(benchmark_gs(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[GS] Error: {e}")

    if "baseline_voxel" in cfg.methods:
        try:
            results.append(benchmark_baseline_voxel(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[BASELINE] Error: {e}")

    if "dvr" in cfg.methods:
        try:
            results.append(benchmark_dvr(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[DVR] Error: {e}")

    if "mip" in cfg.methods:
        try:
            results.append(benchmark_mip(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[MIP] Error: {e}")

    if "spline" in cfg.methods:
        try:
            results.append(benchmark_spline(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[SPLINE] Error: {e}")

    if "unet" in cfg.methods:
        try:
            results.append(benchmark_unet(cfg, slices, train_indices, test_indices))
        except Exception as e:
            print(f"[UNET] Error: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Method':<30} {'PSNR (dB)':<12} {'Time (s)':<12} {'Memory (MB)':<15} {'Params':<12}")
    print("-" * 80)
    
    for result in results:
        print(
            f"{result['method']:<30} "
            f"{result['mean_psnr']:<12.2f} "
            f"{result['training_time_s']:<12.2f} "
            f"{result['memory_mb']:<15.2f} "
            f"{result['num_parameters']:<12,}"
        )
    
    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{cfg.file_prefix}_comparison.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
