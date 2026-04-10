"""
Traditional volume rendering for MRI reconstruction.

Implements classical volume rendering techniques (ray casting, max intensity projection,
average intensity projection) for comparison with Gaussian Splatting rendering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalRenderer(nn.Module):
    """
    Classical volume renderer using ray casting and compositing.
    
    This renderer accumulates voxel intensities along viewing rays to reconstruct
    2D slices from 3D volumes. Provides faster baseline than learning-based methods
    for slice extraction from pre-computed volumes.
    """

    def __init__(self, image_size: tuple[int, int]):
        """
        Initialize the traditional renderer.
        
        Args:
            image_size: (H, W) of output rendered slices.
        """
        super().__init__()
        self.H, self.W = image_size

    def render_mip(
        self,
        volume: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Maximum Intensity Projection (MIP) along specified axis.
        
        Args:
            volume: 3D tensor of shape (D, H, W).
            axis: Projection axis (0=sagittal, 1=coronal, 2=axial).
            
        Returns:
            2D projected image of shape (H, W).
        """
        # Take max intensity along the specified axis
        if axis == 0:
            return torch.max(volume, dim=0)[0]
        elif axis == 1:
            return torch.max(volume, dim=1)[0]
        elif axis == 2:
            return torch.max(volume, dim=2)[0]
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def render_aip(
        self,
        volume: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Average Intensity Projection (AIP) along specified axis.
        
        Args:
            volume: 3D tensor of shape (D, H, W).
            axis: Projection axis (0=sagittal, 1=coronal, 2=axial).
            
        Returns:
            2D projected image of shape (H, W).
        """
        # Take mean intensity along the specified axis
        if axis == 0:
            return torch.mean(volume, dim=0)
        elif axis == 1:
            return torch.mean(volume, dim=1)
        elif axis == 2:
            return torch.mean(volume, dim=2)
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def render_slice_direct(
        self,
        volume: torch.Tensor,
        axis: int,
        slice_idx: int,
    ) -> torch.Tensor:
        """
        Direct slice extraction (no interpolation).
        
        Args:
            volume: 3D tensor of shape (D, H, W).
            axis: Slice axis (0=sagittal, 1=coronal, 2=axial).
            slice_idx: Index along the slicing axis.
            
        Returns:
            2D slice of shape (H, W).
        """
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        elif axis == 2:
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def render_slice_interpolated(
        self,
        volume: torch.Tensor,
        axis: int,
        slice_pos: float,
    ) -> torch.Tensor:
        """
        Slice rendering with trilinear interpolation for sub-voxel positions.
        
        Args:
            volume: 3D tensor of shape (D, H, W).
            axis: Slice axis (0=sagittal, 1=coronal, 2=axial).
            slice_pos: Position along slicing axis (supports fractional indices).
            
        Returns:
            2D interpolated slice of shape (H, W).
        """
        D, H, W = volume.shape
        device = volume.device
        
        # Create coordinate grids
        h_coords = torch.linspace(-1, 1, H, device=device)
        w_coords = torch.linspace(-1, 1, W, device=device)
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        
        # Create depth coordinate based on axis
        d_normalized = (slice_pos / (D - 1) - 0.5) * 2  # Normalize to [-1, 1]
        d_grid = torch.full_like(h_grid, d_normalized)
        
        # Stack into grid for grid_sample
        if axis == 0:
            # Sagittal: slice along x-axis
            grid = torch.stack([d_grid, w_grid, h_grid], dim=-1)  # (H, W, 3)
        elif axis == 1:
            # Coronal: slice along y-axis
            grid = torch.stack([h_grid, d_grid, w_grid], dim=-1)  # (H, W, 3)
        elif axis == 2:
            # Axial: slice along z-axis
            grid = torch.stack([h_grid, w_grid, d_grid], dim=-1)  # (H, W, 3)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        # Prepare volume for grid_sample: (1, 1, D, H, W)
        volume_batch = volume.unsqueeze(0).unsqueeze(0)
        grid_batch = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3) — must be 5D to match 5D volume
        
        # Apply trilinear interpolation
        interpolated = F.grid_sample(
            volume_batch,
            grid_batch,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        
        return interpolated.squeeze()

    def render_dvr(
        self,
        volume: torch.Tensor,
        axis: int,
        n_steps: int = 64,
    ) -> torch.Tensor:
        """
        Direct Volume Rendering (DVR) with alpha compositing along viewing rays.
        
        Args:
            volume: 3D tensor of shape (D, H, W).
            axis: Which axis the rays are perpendicular to (viewing direction).
            n_steps: Number of integration steps along ray.
            
        Returns:
            2D rendered image of shape (H, W).
        """
        D, H, W = volume.shape
        device = volume.device
        
        # Create ray sample positions
        depth_samples = torch.linspace(0, D - 1, n_steps, device=device)
        
        # Initialize composited image
        image = torch.zeros(H, W, device=device, dtype=torch.float32)
        accumulator = torch.ones(H, W, device=device, dtype=torch.float32)
        
        for d_idx in depth_samples:
            # Extract slice at this depth with interpolation
            slice_2d = self.render_slice_interpolated(volume, axis, d_idx.item())
            
            # Convert intensity to pseudo-opacity
            opacity = slice_2d.clamp(0, 1)
            
            # Alpha composite: accumulator = accumulator * (1 - opacity)
            image = image + accumulator * opacity
            accumulator = accumulator * (1 - opacity)
        
        return image
