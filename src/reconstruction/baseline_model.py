"""
Baseline voxel-based 3D reconstruction for MRI using traditional methods.

Reconstructs a 3D volume from 2D slices using learned voxel intensities.
Unlike Gaussian Splatting which uses sparse Gaussian primitives, this approach
maintains dense voxel grids optimized via direct intensity learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineVoxelModel(nn.Module):
    """
    Traditional voxel-based MRI reconstruction model.
    
    Maintains a 3D dense voxel grid and optimizes voxel intensities directly
    to reconstruct slices. Provides baseline comparison for memory usage, inference
    speed, and quality metrics against Gaussian Splatting.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        learning_rate: float = 0.01,
    ):
        """
        Initialize the voxel-based model.
        
        Args:
            volume_shape: (D, H, W) shape of 3D volume to reconstruct.
            learning_rate: Learning rate for voxel intensity optimization.
        """
        super().__init__()
        self.D, self.H, self.W = volume_shape
        self.lr = learning_rate

        # Dense 3D voxel grid: learnable intensity values
        self.voxels = nn.Parameter(
            torch.randn(self.D, self.H, self.W) * 0.1,  # Initialize near 0
            requires_grad=True,
        )

        self.optimizer = torch.optim.Adam([self.voxels], lr=learning_rate)

    def get_slice(self, axis: int, slice_idx: int) -> torch.Tensor:
        """
        Extract a 2D slice from the 3D volume.
        
        Args:
            axis: Slicing axis (0=sagittal, 1=coronal, 2=axial).
            slice_idx: Index along the slicing axis.
            
        Returns:
            2D slice tensor of shape (H, W).
        """
        if axis == 0:
            return self.voxels[slice_idx, :, :]
        elif axis == 1:
            return self.voxels[:, slice_idx, :]
        elif axis == 2:
            return self.voxels[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def forward(self, axis: int, slice_idx: int) -> torch.Tensor:
        """
        Render a slice by extracting from the voxel grid.
        
        Args:
            axis: Slicing axis.
            slice_idx: Index along the slicing axis.
            
        Returns:
            Rendered 2D slice, normalized to [0, 1].
        """
        slice_2d = self.get_slice(axis, slice_idx)
        # Normalize to [0, 1]
        return torch.sigmoid(slice_2d)

    def parameters_per_sample(self) -> int:
        """Return total number of learnable parameters."""
        return self.voxels.numel()

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        return self.voxels.numel() * 4 / (1024 ** 2)  # float32 = 4 bytes


class TrilinearVoxelModel(BaselineVoxelModel):
    """
    Voxel-based model with trilinear interpolation for smooth rendering.
    
    Extends baseline by using trilinear interpolation when rendering at
    arbitrary positions (not just aligned grid indices).
    """

    def render_at_position(
        self,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render voxel intensity at arbitrary 3D position using trilinear interpolation.
        
        Args:
            position: (N, 3) tensor of 3D coordinates in normalized [-1, 1] space.
            
        Returns:
            (N,) tensor of interpolated intensities.
        """
        # Normalize coordinates to grid space
        batch_size = position.shape[0]
        
        # Use grid_sample for trilinear interpolation
        grid = position.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 3)
        voxels_batch = self.voxels.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        # grid_sample expects (N, C, D, H, W) volume and (N, D_out, H_out, W_out, 3) grid
        interpolated = F.grid_sample(
            voxels_batch,
            grid,
            mode="bilinear",  # 3D uses trilinear
            padding_mode="border",
            align_corners=False,
        )
        
        return torch.sigmoid(interpolated.squeeze())
