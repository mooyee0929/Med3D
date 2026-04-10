"""
Differentiable slice renderer for 3D Gaussian models on MRI volumes.

Renders a 2D slice from a 3D Gaussian scene by:
  1. Culling Gaussians outside 3σ of the slice plane
  2. Projecting surviving Gaussians onto the slice plane
  3. Alpha-compositing front-to-back (sorted by depth along slice axis)

Uses diff-gaussian-rasterization CUDA kernels when available,
falls back to pure PyTorch otherwise.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


class VolumeRenderer(nn.Module):
    def __init__(self, volume_shape: tuple[int, int, int]):
        """
        Args:
            volume_shape: (H, W, D) shape of the 3D volume.
                         Allows dynamic image_size calculation based on axis.
        """
        super().__init__()
        self.volume_shape = volume_shape  # (H, W, D)

    def forward(
        self,
        gaussian_model,
        axis: int,
        slice_idx: int,
        max_active: int = 50_000,
    ) -> dict[str, torch.Tensor]:
        """
        Render one MRI slice from the Gaussian model.

        Args:
            gaussian_model: GaussianModel instance.
            axis:           Slice axis (0=sagittal, 1=coronal, 2=axial).
            slice_idx:      Index along the slice axis.

        Returns:
            dict with keys:
              "render": (1, H, W) float32 — rendered intensity
              "alpha":  (1, H, W) float32 — accumulated alpha
              "depth":  (1, H, W) float32 — weighted average depth
        """
        if _CUDA_AVAILABLE and gaussian_model.get_xyz.is_cuda:
            return self._render_cuda(gaussian_model, axis, slice_idx)
        return self._render_pytorch(gaussian_model, axis, slice_idx, max_active=max_active)

    # ------------------------------------------------------------------
    # Pure-PyTorch renderer
    # ------------------------------------------------------------------

    def _render_pytorch(
        self,
        gaussian_model,
        axis: int,
        slice_idx: int,
        max_active: int = 50_000,
    ) -> dict[str, torch.Tensor]:
        """
        Fully vectorised, in-place-free renderer.
        Uses weighted-average compositing (differentiable, no transmittance state).
        Memory: O(max_active × H × W) — capped to avoid OOM.
        """
        # Calculate H, W dynamically based on axis
        volume_h, volume_w, volume_d = self.volume_shape
        axis_to_shape = {0: (volume_w, volume_d), 1: (volume_h, volume_d), 2: (volume_h, volume_w)}
        H, W = axis_to_shape[axis]
        
        device = gaussian_model.get_xyz.device

        xyz      = gaussian_model.get_xyz        # (N, 3)
        scaling  = gaussian_model.get_scaling    # (N, 3)
        opacity  = gaussian_model.get_opacity    # (N, 1)
        intensity = gaussian_model.get_intensity # (N, 1)

        # 1. Cull: keep Gaussians within 3σ of the slice plane
        depth_vals  = xyz[:, axis]
        sigma_axis  = scaling[:, axis]
        depth_diff  = (depth_vals - float(slice_idx)).abs()
        keep        = depth_diff < 3.0 * sigma_axis

        if keep.sum() == 0:
            # No active Gaussians — return near-zero render still connected to parameters
            # so backward can reach _intensity and _opacity even for empty slices.
            dummy = (intensity.sum() + opacity.sum()) * 0.0
            empty = torch.zeros(1, H, W, device=device) + dummy
            return {"render": empty, "alpha": torch.zeros(1, H, W, device=device), "depth": torch.zeros(1, H, W, device=device)}

        # 2. Sort by distance to slice (nearest first) and cap count
        dist_k  = depth_diff[keep]
        order   = dist_k.argsort()[:max_active]

        xyz_k       = xyz[keep][order]        # (M, 3)
        scaling_k   = scaling[keep][order]    # (M, 3)
        opacity_k   = opacity[keep][order]    # (M, 1)
        intensity_k = intensity[keep][order]  # (M, 1)
        depth_diff_k = dist_k[order]          # (M,)
        sigma_axis_k = sigma_axis[keep][order]  # (M,)
        depth_vals_k = depth_vals[keep][order]  # (M,)

        M = xyz_k.shape[0]

        # 3. 2D projected positions
        axes_2d = [i for i in range(3) if i != axis]
        u  = xyz_k[:, axes_2d[0]]     # (M,)
        v  = xyz_k[:, axes_2d[1]]     # (M,)
        su = scaling_k[:, axes_2d[0]] # (M,)
        sv = scaling_k[:, axes_2d[1]] # (M,)

        # 4. Pixel grid — shape (H, W)
        ug = torch.arange(W, dtype=torch.float32, device=device)
        vg = torch.arange(H, dtype=torch.float32, device=device)
        ug, vg = torch.meshgrid(ug, vg, indexing="xy")  # (H, W)

        # 5. Vectorised 2D Gaussian weights: (M, H, W)
        #    Reshape u/v/su/sv to (M, 1, 1) for broadcasting
        du  = vg.unsqueeze(0) - u.view(M, 1, 1)   # (M, H, W)  H-axis (vertical) distance
        dv  = ug.unsqueeze(0) - v.view(M, 1, 1)   # (M, H, W)  W-axis (horizontal) distance
        w2d = torch.exp(
            -0.5 * (du ** 2 / (su.view(M, 1, 1) ** 2 + 1e-8)
                  + dv ** 2 / (sv.view(M, 1, 1) ** 2 + 1e-8))
        )  # (M, H, W)

        # 6. Depth weight and combined alpha: (M, 1, 1) for broadcasting
        depth_w = torch.exp(
            -0.5 * (depth_diff_k / sigma_axis_k.clamp(min=1e-4)) ** 2
        ).view(M, 1, 1)
        alpha   = (opacity_k.view(M, 1, 1) * depth_w).clamp(0, 1)

        # 7. Front-to-back alpha compositing (standard 3DGS formula)
        #    Gaussians are already sorted nearest-first (step 2).
        #    a[i]  = per-pixel effective opacity of Gaussian i
        #    T[i]  = transmittance arriving at Gaussian i
        #          = prod_{j<i}(1 - a[j])   (exclusive cumprod)
        #    C     = sum_i  T[i] * a[i] * intensity[i]
        a = (alpha * w2d).clamp(0, 1)                       # (M, H, W)
        one_minus_a = 1.0 - a                               # (M, H, W)
        cumprod = torch.cumprod(one_minus_a, dim=0)         # inclusive: prod_{j<=i}
        # Exclusive cumprod: T[0]=1, T[i]=prod_{j<i}(1-a[j])
        T = torch.cat([torch.ones(1, H, W, device=device), cumprod[:-1]], dim=0)  # (M, H, W)

        contrib   = T * a                                   # (M, H, W)
        render    = (contrib * intensity_k.view(M, 1, 1)).sum(0)           # (H, W)
        alpha_acc = (1.0 - cumprod[-1]).clamp(0, 1)         # (H, W)
        w_sum     = contrib.sum(0) + 1e-8
        depth_out = (contrib * depth_vals_k.view(M, 1, 1)).sum(0) / w_sum  # (H, W)

        return {
            "render": render.unsqueeze(0),    # (1, H, W)
            "alpha": alpha_acc.unsqueeze(0),  # (1, H, W)
            "depth": depth_out.unsqueeze(0),  # (1, H, W)
        }

    # ------------------------------------------------------------------
    # CUDA renderer (via diff-gaussian-rasterization)
    # ------------------------------------------------------------------

    def _render_cuda(
        self,
        gaussian_model,
        axis: int,
        slice_idx: int,
    ) -> dict[str, torch.Tensor]:
        """
        CUDA-accelerated rendering using diff-gaussian-rasterization.
        Sets up an orthographic camera aligned with the slice plane.
        """
        # Calculate H, W dynamically based on axis
        volume_h, volume_w, volume_d = self.volume_shape
        axis_to_shape = {0: (volume_w, volume_d), 1: (volume_h, volume_d), 2: (volume_h, volume_w)}
        H, W = axis_to_shape[axis]
        
        device = gaussian_model.get_xyz.device

        # Build an orthographic view matrix for the given axis/slice
        # Camera looks along the slice axis; up = next axis
        axis_dirs = torch.eye(3, device=device)
        look = axis_dirs[axis]           # e.g. [0, 0, 1] for axial
        up = axis_dirs[(axis + 1) % 3]

        # Camera position: at the slice plane
        cam_pos = torch.zeros(3, device=device)
        cam_pos[axis] = float(slice_idx)

        # Build view matrix (world-to-camera)
        z_ax = look
        x_ax = torch.cross(up, z_ax).normalize()
        y_ax = torch.cross(z_ax, x_ax)
        R = torch.stack([x_ax, y_ax, z_ax], dim=0)  # (3, 3)
        t = -R @ cam_pos

        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # Orthographic intrinsics: map voxel coords to [-1, 1]
        # Focal length approximation for ortho projection
        tanfovx = 0.5 * W / max(H, W)
        tanfovy = 0.5 * H / max(H, W)
        fovx = 2 * math.atan(tanfovx)
        fovy = 2 * math.atan(tanfovy)

        proj = _get_projection_matrix(fovx, fovy, znear=0.01, zfar=1000.0, device=device)

        bg = torch.zeros(1, device=device)  # black background (intensity 0)

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=proj @ w2c,
            sh_degree=0,
            campos=cam_pos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        xyz = gaussian_model.get_xyz
        # SH features: (N, 1, 1) for sh_degree=0, single intensity channel
        intensity = gaussian_model.get_intensity  # (N, 1)
        shs = intensity.unsqueeze(1)              # (N, 1, 1) — treat as DC SH coeff

        rendered, radii = rasterizer(
            means3D=xyz,
            means2D=torch.zeros_like(xyz[:, :2], requires_grad=True),
            shs=shs,
            colors_precomp=None,
            opacities=gaussian_model.get_opacity,
            scales=gaussian_model.get_scaling,
            rotations=gaussian_model.get_rotation,
            cov3D_precomp=None,
        )
        # rendered: (1, H, W) — single intensity channel
        alpha = (rendered > 0).float()

        return {
            "render": rendered,
            "alpha": alpha,
            "depth": torch.zeros_like(rendered),
        }


def _get_projection_matrix(
    fovx: float, fovy: float, znear: float, zfar: float, device: torch.device
) -> torch.Tensor:
    """OpenGL-style perspective projection matrix."""
    tanHalfFovY = math.tan(fovy / 2)
    tanHalfFovX = math.tan(fovx / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2 * znear / (right - left)
    P[1, 1] = 2 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = -1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * zfar * znear / (zfar - znear)
    return P
