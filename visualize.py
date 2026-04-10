"""
Visualize comparison outputs.

Modes
-----
ply    — render Gaussian centres as a point cloud
nii    — slice montage of a NIfTI volume
compare— side-by-side GT vs reconstruction(s) at held-out test positions

Usage
-----
  # Point cloud (interactive or saved)
  python visualize.py ply outputs/comparisons/BraTS-GLI-00005-100_gs.ply
  python visualize.py ply outputs/comparisons/BraTS-GLI-00005-100_gs.ply --save

  # NIfTI montage
  python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz
  python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz --axis 2 --slice 80
  python visualize.py nii outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz --save

  # Compare GT vs GS only
  python visualize.py compare outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz \
      --gt /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/Data/BraTS-GLI/train/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t1c.nii.gz \
      --save

  # Compare GT vs GS + Baseline + DVR + MIP
  python visualize.py compare outputs/comparisons/BraTS-GLI-00005-100_gs_volume.nii.gz \
      --gt /scratch/eecs504s001w26_class_root/eecs504s001w26_class/fsyang/Data/BraTS-GLI/train/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t1c.nii.gz \
      --also outputs/comparisons/BraTS-GLI-00005-100_baseline_volume.nii.gz \
      --also outputs/comparisons/BraTS-GLI-00005-100_dvr_volume.nii.gz \
      --also outputs/comparisons/BraTS-GLI-00005-100_mip_volume.nii.gz \
      --save

Options
-------
  --save [file.png]   Save to file instead of opening a window.
                      If no path given, auto-saves to outputs/images/.
  --axis  INT         Slice axis: 0=sagittal, 1=coronal, 2=axial (default: 2)
  --slice INT         Single slice index for nii mode (default: montage)
  --gt    PATH        Ground-truth NIfTI for compare mode (required)
  --also  PATH        Additional reconstruction to compare; repeatable
  --test-slices INT   Number of held-out test positions to show (default: 10)

Output layout (compare mode)
-----------------------------
  Each row = one test slice position
  Columns  = GT | Method1 | |GT-Method1| | Method2 | |GT-Method2| | ...
  Difference maps use a hot colormap clipped at 0.3 for visibility.
"""

import sys
from pathlib import Path

import numpy as np


def view_ply(path: str, save: str = None) -> None:
    """Render Gaussian centres as a coloured point cloud."""
    from plyfile import PlyData

    print(f"Loading {path} ...")
    ply = PlyData.read(path)
    v = ply["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    intensity = np.clip(v["intensity"].astype(np.float64), 0, 1)
    opacity   = np.clip(v["opacity"].astype(np.float64),   0, 1)
    brightness = intensity * opacity

    print(f"  {len(xyz):,} Gaussians loaded")

    if save is None:
        # Interactive Open3D window (requires display)
        import open3d as o3d
        colors = np.stack([brightness, brightness, brightness], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("  Controls: left-drag=rotate, scroll=zoom, shift+drag=pan, Q=quit")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=Path(path).stem,
            width=900,
            height=700,
        )
    else:
        # Headless: 3-view scatter plot saved to PNG
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Subsample for speed (max 20k points)
        n = len(xyz)
        idx = np.random.default_rng(0).choice(n, min(n, 20_000), replace=False)
        xyz_s = xyz[idx]
        c_s   = brightness[idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        views = [
            ("X-Y (axial)",    0, 1),
            ("X-Z (coronal)",  0, 2),
            ("Y-Z (sagittal)", 1, 2),
        ]
        for ax, (title, xi, yi) in zip(axes, views):
            ax.scatter(xyz_s[:, xi], xyz_s[:, yi], c=c_s, cmap="gray",
                       s=0.5, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(Path(path).stem, fontsize=10)
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save}")


def view_nii(path: str, axis: int = 2, slice_idx: int = None, save: str = None) -> None:
    """Show a NIfTI volume as a slice montage."""
    import nibabel as nib

    if save is not None:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Loading {path} ...")
    vol = nib.load(path).get_fdata(dtype=np.float32)
    vol = vol.squeeze()  # remove any size-1 dimensions
    print(f"  Volume shape: {vol.shape}")

    # Move the chosen axis to the front so vol[i] is the i-th slice
    vol = np.moveaxis(vol, axis, 0)  # (D, *, *)
    n_slices = vol.shape[0]

    if slice_idx is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(vol[slice_idx], cmap="gray", origin="lower")
        ax.set_title(f"{Path(path).stem}  |  axis={axis}  slice={slice_idx}/{n_slices-1}")
        ax.axis("off")
    else:
        # Montage of up to 25 evenly-spaced slices
        picks = np.linspace(0, n_slices - 1, min(25, n_slices), dtype=int)
        cols = 5
        rows = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        for i, s in enumerate(picks):
            axes[i].imshow(vol[s], cmap="gray", origin="lower")
            axes[i].set_title(f"s={s}", fontsize=8)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(Path(path).stem, fontsize=10)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save}")
    else:
        plt.show()


def view_compare(
    recon_path: str,
    gt_path: str,
    axis: int = 2,
    test_slices: int = 10,
    also: list[str] = None,
    save: str = None,
) -> None:
    """Show GT vs reconstruction(s) at held-out test slice positions.

    Uses the same interleaved test split as compare.py:
      step = n_total // test_slices
      test_indices = [step//2 + i*step for i in range(test_slices)]

    Columns: GT | Recon | Diff  (×3 if --also is given)
    """
    import nibabel as nib

    if save is not None:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def load_norm(path: str, axis: int) -> np.ndarray:
        """Load NIfTI, move slice axis to front, normalise to [0,1]."""
        vol = nib.load(path).get_fdata(dtype=np.float32).squeeze()
        vol = np.moveaxis(vol, axis, 0)   # (D, H, W)
        vmin, vmax = vol.min(), vol.max()
        return (vol - vmin) / (vmax - vmin + 1e-8)

    print(f"Loading GT:    {gt_path}")
    gt_vol   = load_norm(gt_path, axis)
    print(f"Loading recon: {recon_path}")
    rec_vol  = load_norm(recon_path, axis)

    extra_vols, extra_labels = [], []
    for p in (also or []):
        print(f"Loading also:  {p}")
        extra_vols.append(load_norm(p, axis))
        extra_labels.append(Path(p).stem)

    n_total = gt_vol.shape[0]
    step = n_total // test_slices
    test_indices = [step // 2 + i * step for i in range(test_slices)
                    if step // 2 + i * step < n_total]
    print(f"Test indices ({len(test_indices)}): {test_indices}")

    recon_label = Path(recon_path).stem
    all_recons  = [(rec_vol, recon_label)] + list(zip(extra_vols, extra_labels))
    n_methods   = len(all_recons)          # each method → 3 columns (GT | Rec | Diff)
    n_rows      = len(test_indices)
    n_cols      = 1 + n_methods * 2        # GT once, then (Rec + Diff) × n_methods

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.5),
        squeeze=False,
    )

    for row, idx in enumerate(test_indices):
        gt_slice = gt_vol[idx]

        # GT column (first column only)
        ax = axes[row][0]
        ax.imshow(gt_slice, cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.axis("off")
        if row == 0:
            ax.set_title("GT", fontsize=9, fontweight="bold")
        axes[row][0].set_ylabel(f"s={idx}", fontsize=8, rotation=0, labelpad=28, va="center")

        for m, (vol, label) in enumerate(all_recons):
            rec_slice  = vol[idx] if idx < vol.shape[0] else np.zeros_like(gt_slice)
            diff_slice = np.abs(gt_slice - rec_slice)

            col_rec  = 1 + m * 2
            col_diff = 2 + m * 2

            axes[row][col_rec].imshow(rec_slice, cmap="gray", vmin=0, vmax=1, origin="lower")
            axes[row][col_rec].axis("off")

            axes[row][col_diff].imshow(diff_slice, cmap="hot", vmin=0, vmax=0.3, origin="lower")
            axes[row][col_diff].axis("off")

            if row == 0:
                axes[row][col_rec].set_title(label, fontsize=8)
                axes[row][col_diff].set_title(f"|GT−{label}|", fontsize=8)

    plt.suptitle(
        f"GT vs Reconstruction — axis={axis}, test positions={test_indices}",
        fontsize=10,
    )
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save}")
    else:
        plt.show()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()
    path = sys.argv[2]
    save = None
    axis = 2
    slice_idx = None
    gt_path = None
    also = []
    test_slices = 10

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--save":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                save = args[i + 1]; i += 2
            else:
                save = ""; i += 1
        elif args[i] == "--axis"        and i + 1 < len(args): axis        = int(args[i + 1]); i += 2
        elif args[i] == "--slice"       and i + 1 < len(args): slice_idx   = int(args[i + 1]); i += 2
        elif args[i] == "--gt"          and i + 1 < len(args): gt_path     = args[i + 1];      i += 2
        elif args[i] == "--also"        and i + 1 < len(args): also.append(args[i + 1]);       i += 2
        elif args[i] == "--test-slices" and i + 1 < len(args): test_slices = int(args[i + 1]); i += 2
        else: i += 1

    # Resolve save path: no dir → outputs/images/, empty → auto filename
    out_dir = Path("outputs/images")
    if save is not None:
        p = Path(save)
        if save == "" or p.parent == Path("."):
            out_dir.mkdir(parents=True, exist_ok=True)
            if save == "":
                stem = Path(path).stem.split(".")[0]
                if mode == "nii" and slice_idx is not None:
                    fname = f"{stem}_axis{axis}_slice{slice_idx}.png"
                elif mode == "compare":
                    fname = f"{stem}_compare.png"
                else:
                    fname = f"{stem}.png"
            else:
                fname = p.name
            save = str(out_dir / fname)

    if mode == "ply":
        view_ply(path, save=save)
    elif mode == "nii":
        view_nii(path, axis=axis, slice_idx=slice_idx, save=save)
    elif mode == "compare":
        if gt_path is None:
            print("Error: --gt <gt_nii> is required for compare mode.")
            sys.exit(1)
        view_compare(path, gt_path, axis=axis, test_slices=test_slices, also=also, save=save)
    else:
        print(f"Unknown mode '{mode}'. Use 'ply', 'nii', or 'compare'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
