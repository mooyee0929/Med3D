"""
MRI NIfTI loader and preprocessor.
Converts 3D MRI volumes to 2D slice stacks with camera poses for 3DGS.
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MRIVolume:
    data: np.ndarray        # (H, W, D) float32, normalized [0, 1]
    affine: np.ndarray      # (4, 4) voxel-to-world transform
    voxel_size: np.ndarray  # (3,) mm per voxel


def load_nifti(path: str | Path) -> MRIVolume:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    voxel_size = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    return MRIVolume(data=data, affine=img.affine, voxel_size=voxel_size)


def extract_slices(volume: MRIVolume, axis: int = 2,
                   slice_range: Optional[tuple] = None) -> np.ndarray:
    """Extract 2D slices along given axis. Returns (N, H, W) array."""
    data = np.moveaxis(volume.data, axis, 0)  # (D, H, W)
    if slice_range:
        data = data[slice_range[0]:slice_range[1]]
    return data


def get_camera_poses(volume: MRIVolume, axis: int = 2) -> np.ndarray:
    """
    Generate camera-to-world (c2w) matrices for each slice.
    Treats each MRI slice as a fronto-parallel camera.
    Returns (N, 4, 4) array.
    """
    D = volume.data.shape[axis]
    poses = []
    for i in range(D):
        # Translate camera to slice position along axis
        t = np.zeros(3)
        t[axis] = i * volume.voxel_size[axis]
        c2w = np.eye(4)
        c2w[:3, 3] = t
        poses.append(c2w)
    return np.stack(poses)
