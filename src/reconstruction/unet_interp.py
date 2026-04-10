"""3D U-Net for slice interpolation on MRI volumes.

Used as a learned 3D baseline alongside Gaussian Splatting in compare.py
and trainable as a standalone model from train.py (model_type=unet).

Strategy: masked autoencoding.
  - Input  : (B, 1, D, H, W) — sparse volume where unknown axial slices are
             filled by per-pixel linear interpolation from training slices.
  - Output : (B, 1, D, H, W) — refined volume.
  - Loss   : L1 on a held-out subset of the training slices (so the network
             never sees its supervision targets in the input).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
        nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class TinyUNet3D(nn.Module):
    """Compact 2-level 3D U-Net with residual shortcut from input.

    Output = clamp(input + delta, 0, 1) where delta is learned.
    The residual design lets the model start from the linear-interp input
    and only learn small corrections, avoiding the need to reconstruct
    the entire volume from scratch with limited parameters.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.up2  = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1  = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.out  = nn.Conv3d(base, out_ch, kernel_size=1)
        # Initialize output conv near zero so initial delta ≈ 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        delta = self.out(d1)          # learned correction
        return torch.clamp(x + delta, 0.0, 1.0)  # residual: input + correction


def pad_to_multiple_3d(tensor: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Pad D, H, W of (..., D, H, W) tensor to next multiple. Returns padded tensor and (pd, ph, pw)."""
    d, h, w = tensor.shape[-3:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return tensor, (0, 0, 0)
    return F.pad(tensor, (0, pw, 0, ph, 0, pd)), (pd, ph, pw)


def build_input_volume(all_slices, train_indices: list[int]) -> torch.Tensor:
    """Build a (1, 1, D, H, W) input volume by linear interpolation along z
    from the given training slice positions. Used at both train and test time.

    Vectorized with np.searchsorted — O(D log M) instead of O(D × M).
    Boundary positions outside [train_arr[0], train_arr[-1]] are clamped.
    """
    import numpy as np

    if isinstance(all_slices, torch.Tensor):
        np_slices = all_slices.detach().cpu().numpy()
    else:
        np_slices = all_slices

    D = np_slices.shape[0]
    train_arr = np.asarray(sorted(train_indices), dtype=np.int64)  # (M,)
    M = len(train_arr)

    z = np.arange(D, dtype=np.int64)
    # right neighbour index in train_arr for each z (in [1, M-1] after clamp)
    right = np.clip(np.searchsorted(train_arr, z, side="right"), 1, M - 1)
    left = right - 1

    z_l = train_arr[left]
    z_r = train_arr[right]
    denom = np.maximum(z_r - z_l, 1).astype(np.float32)
    t = ((z - z_l) / denom).astype(np.float32)            # (D,)
    t_clamped = np.clip(t, 0.0, 1.0)                      # boundary clamp

    sl_left  = np_slices[z_l]                             # (D, H, W)
    sl_right = np_slices[z_r]                             # (D, H, W)
    out = (1.0 - t_clamped[:, None, None]) * sl_left + t_clamped[:, None, None] * sl_right
    out = out.astype(np.float32)

    # Force exact GT at training positions (avoids tiny rounding error)
    out[train_arr] = np_slices[train_arr]

    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)