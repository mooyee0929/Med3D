"""
Visualization utilities for MRI slices and Gaussian statistics.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SEG_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),   # background: transparent
    1: (1.0, 0.0, 0.0, 0.6),   # NCR: red
    2: (0.0, 1.0, 0.0, 0.6),   # SNFH/edema: green
    3: (0.0, 0.0, 1.0, 0.6),   # ET: blue
    4: (1.0, 1.0, 0.0, 0.6),   # tumor core: yellow
}


def plot_slice_grid(
    slices: list[np.ndarray],
    titles: list[str],
    output_path: str | Path,
    cols: int = 5,
    cmap: str = "gray",
) -> None:
    """
    Save a grid of 2D slices as a PNG.

    Args:
        slices: List of (H, W) numpy arrays, values in [0, 1].
        titles: Title for each slice.
        output_path: Where to save the PNG.
        cols: Number of columns in the grid.
    """
    rows = (len(slices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, (sl, title) in enumerate(zip(slices, titles)):
        axes[i].imshow(sl, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    for j in range(len(slices), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def overlay_seg_on_slice(
    image_slice: np.ndarray,
    seg_slice: np.ndarray,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Overlay segmentation labels on a grayscale MRI slice.

    Returns an (H, W, 4) RGBA array. Optionally saves to output_path.
    """
    h, w = image_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = np.stack([image_slice] * 3, axis=-1)
    rgba[:, :, 3] = 1.0

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        overlay[mask] = color

    # Alpha blend
    alpha = overlay[:, :, 3:4]
    blended = rgba.copy()
    blended[:, :, :3] = rgba[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha

    if output_path is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(blended)
        patches = [
            mpatches.Patch(color=c[:3], label=name)
            for (label, c), name in zip(
                SEG_COLORS.items(),
                ["Background", "NCR", "SNFH/Edema", "ET", "Tumor Core"],
            )
            if label > 0
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7)
        ax.axis("off")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    return blended


def plot_gaussian_stats(gaussian_model, output_path: str | Path) -> None:
    """
    Save histogram plots of Gaussian parameter distributions.
    gaussian_model must have: get_scaling, get_opacity, get_intensity properties.
    """
    scaling = gaussian_model.get_scaling.detach().cpu().numpy()
    opacity = gaussian_model.get_opacity.detach().cpu().numpy().flatten()
    intensity = gaussian_model.get_intensity.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(scaling.max(axis=1), bins=50, color="steelblue")
    axes[0].set_title(f"Max Scale (N={len(opacity):,})")
    axes[0].set_xlabel("voxels")

    axes[1].hist(opacity, bins=50, color="coral")
    axes[1].set_title("Opacity")
    axes[1].set_xlabel("[0, 1]")

    axes[2].hist(intensity, bins=50, color="seagreen")
    axes[2].set_title("Intensity")
    axes[2].set_xlabel("[0, 1]")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
