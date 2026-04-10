"""
3D Gaussian model for MRI volume representation.

Replaces RGB color with scalar intensity. Each Gaussian has:
  _xyz        (N, 3)  — position in voxel coordinates
  _scaling    (N, 3)  — log-scale of ellipsoid axes
  _rotation   (N, 4)  — unit quaternion (w, x, y, z)
  _intensity  (N, 1)  — scalar MRI intensity (raw logit → sigmoid)
  _opacity    (N, 1)  — opacity (raw logit → sigmoid)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.mri_loader import MRIVolume


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (N, 4) [w,x,y,z] to rotation matrices (N, 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).view(-1, 3, 3)
    return R


class GaussianModel(nn.Module):
    def __init__(self, max_scale: float = 32.0):
        super().__init__()
        self.max_scale = max_scale

        # Learnable parameters (unactivated)
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._intensity = nn.Parameter(torch.empty(0, 1))
        self._opacity = nn.Parameter(torch.empty(0, 1))

        # Gradient accumulation buffers for densification
        self.register_buffer("xyz_grad_accum", torch.empty(0), persistent=False)
        self.register_buffer("xyz_grad_denom", torch.empty(0), persistent=False)

    # ------------------------------------------------------------------
    # Property accessors (activated)
    # ------------------------------------------------------------------

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling).clamp(max=self.max_scale)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation / self._rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @property
    def get_intensity(self) -> torch.Tensor:
        return torch.sigmoid(self._intensity)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Return (N, 3, 3) covariance matrices: Sigma = R @ S^2 @ R^T."""
        S = torch.diag_embed(self.get_scaling * scaling_modifier)  # (N, 3, 3)
        R = _quat_to_rotmat(self.get_rotation)                      # (N, 3, 3)
        return R @ S @ S @ R.transpose(-1, -2)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_volume(
        self,
        volume: MRIVolume,
        num_gaussians: int = 100_000,
        seed: int = 42,
    ) -> None:
        """Seed Gaussians uniformly across all brain tissue voxels."""
        rng = np.random.default_rng(seed)
        data = volume.data  # (H, W, D), float32 [0, 1]

        # Mask all brain tissue voxels (not just bright ones)
        mask = data > 0.01
        if mask.sum() < num_gaussians:
            num_gaussians = int(mask.sum())

        coords = np.argwhere(mask).astype(np.float32)  # (M, 3)
        intensities = data[mask]                         # (M,)

        # Sample weighted by intensity
        probs = intensities / intensities.sum()
        indices = rng.choice(len(coords), size=num_gaussians, replace=True, p=probs)
        sampled_coords = coords[indices]        # (N, 3)
        sampled_intensities = intensities[indices]  # (N,)

        # Add small jitter to avoid degenerate initialization
        sampled_coords += rng.uniform(-0.5, 0.5, sampled_coords.shape).astype(np.float32)

        # Initialize parameters
        xyz = torch.from_numpy(sampled_coords)
        scaling = torch.full((num_gaussians, 3), np.log(0.5))  # half-voxel radius
        rotation = torch.zeros(num_gaussians, 4)
        rotation[:, 0] = 1.0  # identity quaternion (w=1)
        intensity = torch.logit(
            torch.from_numpy(sampled_intensities).float().clamp(0.01, 0.99).unsqueeze(1)
        )
        opacity = torch.full((num_gaussians, 1), torch.logit(torch.tensor(0.1)).item())

        self._xyz = nn.Parameter(xyz)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rotation)
        self._intensity = nn.Parameter(intensity)
        self._opacity = nn.Parameter(opacity)

        # Reset grad accum buffers
        n = num_gaussians
        self.xyz_grad_accum = torch.zeros(n, device=xyz.device)
        self.xyz_grad_denom = torch.zeros(n, device=xyz.device)

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def get_optimizer_params(self, cfg) -> list[dict]:
        # Support both flat Config (cfg.position_lr) and nested (cfg.gaussian.position_lr)
        g = getattr(cfg, "gaussian", cfg)
        return [
            {"params": [self._xyz],       "lr": g.position_lr, "name": "xyz"},
            {"params": [self._intensity], "lr": g.feature_lr,  "name": "intensity"},
            {"params": [self._opacity],   "lr": g.opacity_lr,  "name": "opacity"},
            {"params": [self._scaling],   "lr": g.scaling_lr,  "name": "scaling"},
            {"params": [self._rotation],  "lr": g.rotation_lr, "name": "rotation"},
        ]

    # ------------------------------------------------------------------
    # Densification helpers
    # ------------------------------------------------------------------

    def update_xyz_grad_accum(self) -> None:
        """Accumulate L2 norm of xyz gradients for densification."""
        if self._xyz.grad is not None:
            grad_norm = self._xyz.grad.norm(dim=-1)
            self.xyz_grad_accum += grad_norm
            self.xyz_grad_denom += 1

    def _prune_points(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """Remove Gaussians where mask is True. Updates optimizer state in-place."""
        keep = ~mask
        for group in optimizer.param_groups:
            name = group["name"]
            param = group["params"][0]
            new_param = nn.Parameter(param.data[keep].clone())
            # Surgically update Adam state
            state = optimizer.state.pop(param, {})
            new_state = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == param.shape[0]:
                    new_state[k] = v[keep].clone()
                else:
                    new_state[k] = v
            group["params"][0] = new_param
            optimizer.state[new_param] = new_state
            setattr(self, f"_{name}", new_param)

        self.xyz_grad_accum = self.xyz_grad_accum[keep]
        self.xyz_grad_denom = self.xyz_grad_denom[keep]

    def _add_points(
        self,
        new_xyz: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_intensity: torch.Tensor,
        new_opacity: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Append new Gaussians. Extends optimizer state with zeros."""
        new_params = {
            "xyz": new_xyz, "scaling": new_scaling, "rotation": new_rotation,
            "intensity": new_intensity, "opacity": new_opacity,
        }
        for group in optimizer.param_groups:
            name = group["name"]
            param = group["params"][0]
            ext = new_params[name]
            new_param = nn.Parameter(torch.cat([param.data, ext], dim=0))
            state = optimizer.state.pop(param, {})
            new_state = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == param.shape[0]:
                    zeros = torch.zeros(ext.shape[0], *v.shape[1:], dtype=v.dtype, device=v.device)
                    new_state[k] = torch.cat([v, zeros], dim=0)
                else:
                    new_state[k] = v
            group["params"][0] = new_param
            optimizer.state[new_param] = new_state
            setattr(self, f"_{name}", new_param)

        n_new = new_xyz.shape[0]
        device = self._xyz.device
        self.xyz_grad_accum = torch.cat([self.xyz_grad_accum, torch.zeros(n_new, device=device)])
        self.xyz_grad_denom = torch.cat([self.xyz_grad_denom, torch.zeros(n_new, device=device)])

    def densify_and_prune(
        self,
        grad_threshold: float,
        optimizer: torch.optim.Optimizer,
        min_opacity: float = 0.005,
    ) -> dict:
        """Split high-gradient large Gaussians, clone high-gradient small ones, prune weak ones."""
        stats = {"pruned": 0, "cloned": 0, "split": 0}

        avg_grad = torch.zeros(self.num_gaussians, device=self._xyz.device)
        valid = self.xyz_grad_denom > 0
        avg_grad[valid] = self.xyz_grad_accum[valid] / self.xyz_grad_denom[valid]

        # Reset accumulators
        self.xyz_grad_accum.zero_()
        self.xyz_grad_denom.zero_()

        scaling = self.get_scaling        # (N, 3)
        max_scale = scaling.max(dim=1).values  # (N,)
        scale_threshold = self.max_scale * 0.1  # 10% of max allowed scale

        high_grad = avg_grad > grad_threshold
        large = max_scale > scale_threshold
        small = ~large

        # Clone: high gradient + small scale
        clone_mask = high_grad & small
        if clone_mask.any():
            new_xyz = self._xyz.data[clone_mask] + torch.randn_like(self._xyz.data[clone_mask]) * 0.1
            self._add_points(
                new_xyz,
                self._scaling.data[clone_mask],
                self._rotation.data[clone_mask],
                self._intensity.data[clone_mask],
                self._opacity.data[clone_mask],
                optimizer,
            )
            stats["cloned"] = int(clone_mask.sum())

        # Split: high gradient + large scale
        split_mask = high_grad & large
        if split_mask.any():
            n_split = int(split_mask.sum())
            R = _quat_to_rotmat(self.get_rotation[split_mask])  # (n, 3, 3)
            principal = R[:, :, 0]  # first axis of ellipsoid
            offset = principal * scaling[split_mask, 0:1] * 0.5
            xyz_a = self._xyz.data[split_mask] + offset
            xyz_b = self._xyz.data[split_mask] - offset
            new_scaling = self._scaling.data[split_mask] - np.log(2)  # halve scale
            for new_xyz in [xyz_a, xyz_b]:
                self._add_points(
                    new_xyz, new_scaling,
                    self._rotation.data[split_mask],
                    self._intensity.data[split_mask],
                    self._opacity.data[split_mask],
                    optimizer,
                )
            stats["split"] = n_split

        # Prune: low opacity or oversized
        # Recompute scaling for the (possibly grown) set of Gaussians
        new_scaling = self.get_scaling
        new_max_scale = new_scaling.max(dim=1).values
        prune_mask = (self.get_opacity.squeeze(-1) < min_opacity) | (new_max_scale > self.max_scale)
        # Also prune the original split Gaussians (split_mask was computed on original N)
        if split_mask.any():
            original_n = split_mask.shape[0]
            full_split = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self._xyz.device)
            full_split[:original_n] = split_mask
            prune_mask = prune_mask | full_split

        if prune_mask.any():
            self._prune_points(prune_mask, optimizer)
            stats["pruned"] = int(prune_mask.sum())

        return stats

    def reset_opacity(self, optimizer: torch.optim.Optimizer) -> None:
        """Reset all opacities to logit(0.01), forcing the model to re-earn opacity."""
        new_val = torch.full_like(self._opacity.data, torch.logit(torch.tensor(0.01)).item())
        for group in optimizer.param_groups:
            if group["name"] == "opacity":
                param = group["params"][0]
                new_param = nn.Parameter(new_val.clone())
                state = optimizer.state.pop(param, {})
                # Reset Adam moments for opacity
                optimizer.state[new_param] = {
                    k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else v
                    for k, v in state.items()
                }
                group["params"][0] = new_param
                self._opacity = new_param
                break

    # ------------------------------------------------------------------
    # PLY save / load
    # ------------------------------------------------------------------

    def save_ply(self, path: str | Path, iteration: int = 0) -> None:
        from plyfile import PlyData, PlyElement
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        intensities = self.get_intensity.detach().cpu().numpy().flatten()
        opacities = self.get_opacity.detach().cpu().numpy().flatten()
        scalings = self._scaling.detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("intensity", "f4"), ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        elements["nx"] = elements["ny"] = elements["nz"] = 0.0
        elements["intensity"] = intensities
        elements["opacity"] = opacities
        elements["scale_0"], elements["scale_1"], elements["scale_2"] = (
            scalings[:, 0], scalings[:, 1], scalings[:, 2]
        )
        elements["rot_0"], elements["rot_1"], elements["rot_2"], elements["rot_3"] = (
            rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        )

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(path))

        # Save companion metadata
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({"iteration": iteration, "num_gaussians": self.num_gaussians}, f)

        print(f"Saved {self.num_gaussians:,} Gaussians → {path}")

    def load_ply(self, path: str | Path) -> None:
        from plyfile import PlyData
        plydata = PlyData.read(str(path))
        v = plydata["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        intensity = v["intensity"][:, None]
        opacity = v["opacity"][:, None]
        scaling = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        rotation = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)

        self._xyz = nn.Parameter(torch.from_numpy(xyz.astype(np.float32)))
        self._scaling = nn.Parameter(torch.from_numpy(scaling.astype(np.float32)))
        self._rotation = nn.Parameter(torch.from_numpy(rotation.astype(np.float32)))
        self._intensity = nn.Parameter(
            torch.logit(torch.from_numpy(intensity.astype(np.float32)).clamp(0.01, 0.99))
        )
        self._opacity = nn.Parameter(
            torch.logit(torch.from_numpy(opacity.astype(np.float32)).clamp(0.01, 0.99))
        )

        n = self._xyz.shape[0]
        self.xyz_grad_accum = torch.zeros(n, device=self._xyz.device)
        self.xyz_grad_denom = torch.zeros(n, device=self._xyz.device)
        print(f"Loaded {n:,} Gaussians from {path}")
