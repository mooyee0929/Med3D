"""
Training entry point for 3D Gaussian Splatting on BraTS-GLI MRI.

Supports two modes:
  dense  -- Train on all slices (baseline)
  sparse -- Train on every Nth axial slice, evaluate on held-out slices (2D→3D)

Usage:
  python train.py case_id=BraTS-GLI-00005-100
  python train.py case_id=BraTS-GLI-00005-100 sparse_factor=5   # easy
  python train.py case_id=BraTS-GLI-00005-100 sparse_factor=10  # medium
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.gaussian.gaussian_model import GaussianModel
from src.preprocessing.mri_loader import load_nifti, extract_slices
from src.reconstruction.unet_interp import TinyUNet3D, pad_to_multiple_3d, build_input_volume
from src.rendering.volume_renderer import VolumeRenderer
from src.utils.metrics import psnr, ssim_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    # Data
    data_root: str = str(Path(__file__).resolve().parent.parent / "Data" / "BraTS-GLI" / "train")
    case_id: str = "BraTS-GLI-00005-100"
    modality: str = "t1c"

    # Which model to train: gs (Gaussian Splatting) or unet (U-Net slice interp)
    model_type: str = "gs"

    # 3D U-Net (only used when model_type=unet)
    unet_iterations: int = 10_000
    unet_lr: float = 1e-3
    unet_base_ch: int = 16

    # Sparse input (0 = dense, use all slices)
    sparse_factor: int = 0   # 5 = easy, 10 = medium

    # Gaussian
    num_gaussians: int = 100_000
    position_lr: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5.0e-3
    rotation_lr: float = 1.0e-3
    lambda_dssim: float = 0.2

    # Training
    iterations: int = 30_000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    densify_every: int = 100
    opacity_reset_every: int = 3000
    test_iterations: list = None
    save_iterations: list = None

    # Output
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    image_size: tuple = (182, 218)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.test_iterations is None:
            self.test_iterations = [7000, 15000, self.iterations]
        if self.save_iterations is None:
            self.save_iterations = [7000, 15000, self.iterations]


def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()
    kwargs = {}
    for override in args.overrides:
        key, _, val = override.partition("=")
        key = key.split(".")[-1]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        kwargs[key] = val
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# Sparse slice index builder
# ---------------------------------------------------------------------------

def build_slice_sets(n_slices: int, sparse_factor: int) -> tuple[list[int], list[int]]:
    """
    Split axial slices into training (seen) and holdout (unseen).

    sparse_factor=0  → all slices are training, holdout is empty
    sparse_factor=5  → every 5th slice trains, the rest are holdout
    sparse_factor=10 → every 10th slice trains, the rest are holdout

    Returns: (training_indices, holdout_indices)
    """
    if sparse_factor == 0:
        return list(range(n_slices)), []

    training = list(range(0, n_slices, sparse_factor))
    holdout = [i for i in range(n_slices) if i not in set(training)]
    return training, holdout


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_expon_lr(lr_init: float, lr_final: float, max_steps: int):
    def scheduler(step: int) -> float:
        t = min(step / max_steps, 1.0)
        return lr_init * (lr_final / lr_init) ** t
    return scheduler


def update_lr(optimizer, name: str, new_lr: float) -> None:
    for group in optimizer.param_groups:
        if group["name"] == name:
            group["lr"] = new_lr
            break


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    gaussians: GaussianModel,
    renderer: VolumeRenderer,
    volume,
    iteration: int,
    training_axial: list[int],
    holdout_axial: list[int],
    writer=None,
    save_dir: Path = None,
) -> dict:
    """
    Evaluate on both seen (training) and unseen (holdout) axial slices.
    The key metric is holdout PSNR — measures true 2D→3D interpolation quality.
    """
    gaussians.eval()
    H, W, D = volume.data.shape
    all_slices = extract_slices(volume, axis=2)  # (D, H, W)
    gt_min, gt_max = volume.data.min(), volume.data.max()

    results = {"seen": [], "unseen": []}

    eval_seen    = training_axial[::max(1, len(training_axial) // 10)]   # sample up to 10
    eval_holdout = holdout_axial[::max(1, len(holdout_axial) // 20)]     # sample up to 20

    rendered_imgs, gt_imgs, labels = [], [], []

    for group, indices in [("seen", eval_seen), ("unseen", eval_holdout)]:
        for idx in indices:
            out = renderer(gaussians, axis=2, slice_idx=idx)
            rendered = out["render"].cpu()  # (1, H, W)

            gt = all_slices[idx]
            gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)
            gt_t = torch.from_numpy(gt_norm[None].astype(np.float32))  # (1, H, W)

            p = psnr(rendered.unsqueeze(0), gt_t.unsqueeze(0))
            if not math.isinf(p):
                results[group].append(p)

            rendered_imgs.append(rendered.squeeze().numpy())
            gt_imgs.append(gt_norm)
            labels.append(f"{group} slice {idx}")

    mean_seen    = float(np.mean(results["seen"]))    if results["seen"]    else 0.0
    mean_unseen  = float(np.mean(results["unseen"]))  if results["unseen"]  else 0.0

    print(f"  [Eval iter={iteration}]")
    print(f"    Seen    slices PSNR: {mean_seen:.2f} dB  ({len(results['seen'])} slices)")
    print(f"    Unseen  slices PSNR: {mean_unseen:.2f} dB  ({len(results['unseen'])} slices)  ← key metric")

    if writer:
        writer.add_scalar("Eval/PSNR_seen",   mean_seen,   iteration)
        writer.add_scalar("Eval/PSNR_unseen", mean_unseen, iteration)

    # Save comparison grid
    if save_dir is not None and len(rendered_imgs) > 0:
        try:
            from src.utils.viz import plot_slice_grid
            save_dir.mkdir(parents=True, exist_ok=True)
            interleaved, interleaved_titles = [], []
            for r, g, lbl in zip(rendered_imgs, gt_imgs, labels):
                interleaved += [r, g]
                interleaved_titles += [f"Render {lbl}", f"GT {lbl}"]
            plot_slice_grid(
                interleaved, interleaved_titles,
                save_dir / f"eval_iter{iteration:06d}.png",
                cols=4,
            )
        except Exception as e:
            print(f"  (viz skipped: {e})")

    gaussians.train()
    return {"seen": mean_seen, "unseen": mean_unseen}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_unet(cfg: Config) -> None:
    """Train the 3D TinyUNet slice-interpolation model via masked autoencoding.

    Saves a single checkpoint to checkpoints/{case_id}{suffix}_unet3d.pt
    where suffix is "_dense" or "_sparseN" matching train.py's GS layout.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Case:    {cfg.case_id}")
    print(f"Model:   3D U-Net (slice interpolation, masked autoencoding)")
    print(f"Sparse:  {cfg.sparse_factor}")
    print(f"Iters:   {cfg.unet_iterations}")
    print(f"Base ch: {cfg.unet_base_ch}")

    nifti_path = Path(cfg.data_root) / cfg.case_id / f"{cfg.case_id}-{cfg.modality}.nii.gz"
    print(f"\nLoading: {nifti_path}")
    volume = load_nifti(nifti_path)
    H, W, D = volume.data.shape

    all_slices = np.array(extract_slices(volume, axis=2))
    gt_min, gt_max = all_slices.min(), all_slices.max()
    all_slices = (all_slices - gt_min) / (gt_max - gt_min + 1e-8)

    training_axial, holdout_axial = build_slice_sets(D, cfg.sparse_factor)
    train_arr = sorted(training_axial)
    print(f"Training slices: {len(train_arr)}, holdout: {len(holdout_axial)}")

    if len(train_arr) < 4:
        raise RuntimeError("Need at least 4 training slices for masked autoencoding")

    model = TinyUNet3D(in_ch=1, out_ch=1, base=cfg.unet_base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.unet_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.unet_iterations, eta_min=cfg.unet_lr * 0.01
    )
    gt_volume = torch.from_numpy(all_slices.astype(np.float32)).to(device)
    n_mask = max(1, len(train_arr) // 6)

    print(f"\nTraining 3D U-Net for {cfg.unet_iterations} iters (mask {n_mask}/{len(train_arr)} per step)...\n")
    model.train()
    for it in range(1, cfg.unet_iterations + 1):
        masked = random.sample(train_arr, n_mask)
        visible = [i for i in train_arr if i not in set(masked)]

        input_vol = build_input_volume(all_slices, visible).to(device)
        input_padded, _ = pad_to_multiple_3d(input_vol, multiple=4)
        pred_padded = model(input_padded)
        pred_full = pred_padded[..., :D, :H, :W]

        mask_idx = torch.tensor(masked, dtype=torch.long, device=device)
        loss = F.l1_loss(pred_full[0, 0, mask_idx], gt_volume[mask_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 200 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Iter {it:6d} | L1={loss.item():.6f} | LR={lr_now:.5f}")

    # Held-out evaluation
    if holdout_axial:
        model.eval()
        with torch.no_grad():
            input_vol = build_input_volume(all_slices, train_arr).to(device)
            input_padded, _ = pad_to_multiple_3d(input_vol, multiple=4)
            pred_padded = model(input_padded)
            pred_full = pred_padded[0, 0, :D, :H, :W]
            psnrs = []
            for test_pos in holdout_axial:
                psnrs.append(psnr(pred_full[test_pos], gt_volume[test_pos]))
        if psnrs:
            finite = [p for p in psnrs if not math.isinf(p)]
            mean = float(np.mean(finite)) if finite else 0.0
            print(f"\nHoldout PSNR: {mean:.2f} dB ({len(finite)} slices)")

    # Save checkpoint
    suffix = f"_sparse{cfg.sparse_factor}" if cfg.sparse_factor > 0 else "_dense"
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.case_id}{suffix}_unet3d.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved {n_params:,} 3D U-Net params → {ckpt_path}")


def main() -> None:
    cfg = parse_args()

    if cfg.model_type == "unet":
        train_unet(cfg)
        return
    if cfg.model_type == "spline":
        print("Cubic spline has no learnable parameters — nothing to train.")
        print("compare.py runs spline directly from training slices at eval time.")
        return
    if cfg.model_type != "gs":
        raise ValueError(f"Unknown model_type={cfg.model_type!r}; expected 'gs', 'unet', or 'spline'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_str = f"sparse (every {cfg.sparse_factor} slices)" if cfg.sparse_factor > 0 else "dense (all slices)"
    print(f"Device:  {device}")
    print(f"Case:    {cfg.case_id}")
    print(f"Mode:    {mode_str}")
    print(f"Iters:   {cfg.iterations}")

    # Checkpoint dir includes sparse factor so experiments don't overwrite each other
    suffix = f"_sparse{cfg.sparse_factor}" if cfg.sparse_factor > 0 else "_dense"
    ckpt_dir = Path(cfg.checkpoint_dir) / (cfg.case_id + suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = ckpt_dir / "eval_images"

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_path = Path(cfg.log_dir) / (cfg.case_id + suffix)
        log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_path))
        print(f"TensorBoard: {log_path}")
    except ImportError:
        print("TensorBoard not available.")

    # Load volume
    nifti_path = Path(cfg.data_root) / cfg.case_id / f"{cfg.case_id}-{cfg.modality}.nii.gz"
    print(f"\nLoading: {nifti_path}")
    volume = load_nifti(nifti_path)
    H, W, D = volume.data.shape
    print(f"Volume: {volume.data.shape}, voxel: {volume.voxel_size} mm")

    # Build sparse slice sets
    training_axial, holdout_axial = build_slice_sets(D, cfg.sparse_factor)
    print(f"\nSlice split (axial axis, total={D}):")
    print(f"  Training (seen):   {len(training_axial)} slices  {training_axial[:5]}{'...' if len(training_axial)>5 else ''}")
    print(f"  Holdout (unseen):  {len(holdout_axial)} slices  {holdout_axial[:5]}{'...' if len(holdout_axial)>5 else ''}")

    # Save split info
    with open(ckpt_dir / "slice_split.json", "w") as f:
        json.dump({
            "sparse_factor": cfg.sparse_factor,
            "total_axial": D,
            "training_axial": training_axial,
            "holdout_axial": holdout_axial,
        }, f, indent=2)

    # Gaussian model
    gaussians = GaussianModel(max_scale=float(max(H, W, D)) * 0.1)

    start_iter = 1
    existing_ckpts = sorted(ckpt_dir.glob("iteration_*.ply"))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        gaussians.load_ply(latest)
        meta_path = latest.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            start_iter = meta.get("iteration", 1) + 1
        print(f"Resuming from iteration {start_iter}")
    else:
        print(f"\nInitializing {cfg.num_gaussians:,} Gaussians...")
        gaussians.initialize_from_volume(volume, num_gaussians=cfg.num_gaussians)

    # Move to device AFTER initialization so all parameters land on the right device
    gaussians.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(gaussians.get_optimizer_params(cfg), lr=0.0, eps=1e-15)
    xyz_scheduler = get_expon_lr(cfg.position_lr, cfg.position_lr * 0.01, cfg.iterations)

    # Renderer (volume_shape enables dynamic axis-based sizing)
    renderer = VolumeRenderer(volume_shape=(H, W, D))

    # Precompute all slices for fast GT lookup
    all_axial = extract_slices(volume, axis=2)    # (D, H, W)
    all_coronal = extract_slices(volume, axis=1)   # (W, H, D)  — used in dense mode
    all_sagittal = extract_slices(volume, axis=0)  # (H, W, D)  — used in dense mode
    gt_min, gt_max = volume.data.min(), volume.data.max()

    def get_gt_tensor(axis: int, idx: int) -> torch.Tensor:
        sl = {0: all_sagittal, 1: all_coronal, 2: all_axial}[axis][idx]
        sl_norm = (sl - gt_min) / (gt_max - gt_min + 1e-8)
        return torch.from_numpy(sl_norm[None, None].astype(np.float32)).to(gaussians.get_xyz.device)

    print(f"\nStarting training from iter {start_iter}...\n")

    for iteration in range(start_iter, cfg.iterations + 1):
        update_lr(optimizer, "xyz", xyz_scheduler(iteration))

        # ── Sample a training slice ──────────────────────────────────────
        if cfg.sparse_factor > 0:
            # Sparse mode: only axial, only from training set
            axis = 2
            slice_idx = random.choice(training_axial)
        else:
            # Dense mode: any axis, any slice
            axis = random.choice([0, 1, 2])
            slice_idx = random.randint(0, volume.data.shape[axis] - 1)

        # ── Render & loss ────────────────────────────────────────────────
        gt_t = get_gt_tensor(axis, slice_idx)
        render_pkg = renderer(gaussians, axis=axis, slice_idx=slice_idx)
        rendered = render_pkg["render"].unsqueeze(0)  # (1, 1, H, W)

        l1    = F.l1_loss(rendered, gt_t)
        dssim = ssim_loss(rendered.squeeze(0), gt_t.squeeze(0))
        loss  = (1 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * dssim

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gaussians.update_xyz_grad_accum()
        optimizer.step()

        # ── Densification ────────────────────────────────────────────────
        if cfg.densify_from_iter <= iteration <= cfg.densify_until_iter:
            if iteration % cfg.densify_every == 0:
                stats = gaussians.densify_and_prune(cfg.densify_grad_threshold, optimizer)
                if writer:
                    writer.add_scalar("Stats/pruned", stats["pruned"], iteration)
                    writer.add_scalar("Stats/cloned", stats["cloned"], iteration)
                    writer.add_scalar("Stats/split",  stats["split"],  iteration)

        if iteration % cfg.opacity_reset_every == 0 and iteration <= cfg.densify_until_iter:
            gaussians.reset_opacity(optimizer)

        # ── Logging ──────────────────────────────────────────────────────
        if iteration % 100 == 0:
            n_gauss = gaussians.num_gaussians
            print(f"Iter {iteration:6d} | loss={loss.item():.4f} | "
                  f"L1={l1.item():.4f} | SSIM={1-dssim.item():.4f} | "
                  f"N={n_gauss:,} | axis={axis} slice={slice_idx}")
            if writer:
                writer.add_scalar("Loss/total",          loss.item(),  iteration)
                writer.add_scalar("Loss/L1",             l1.item(),    iteration)
                writer.add_scalar("Loss/DSSIM",          dssim.item(), iteration)
                writer.add_scalar("Stats/num_gaussians", n_gauss,      iteration)

        # ── Evaluation ───────────────────────────────────────────────────
        if iteration in cfg.test_iterations:
            evaluate(
                gaussians, renderer, volume, iteration,
                training_axial, holdout_axial,
                writer=writer, save_dir=eval_dir,
            )

        # ── Checkpoint ───────────────────────────────────────────────────
        if iteration in cfg.save_iterations:
            gaussians.save_ply(ckpt_dir / f"iteration_{iteration:06d}.ply", iteration=iteration)

    if writer:
        writer.close()
    print("\nTraining complete.")
    print(f"Checkpoints saved to: {ckpt_dir}/")


if __name__ == "__main__":
    main()
