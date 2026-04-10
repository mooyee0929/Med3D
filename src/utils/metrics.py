"""
Image quality metrics: PSNR, SSIM (pure PyTorch, differentiable).
"""
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio. Both tensors in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Differentiable SSIM loss = 1 - SSIM.
    pred, target: (B, C, H, W) or (C, H, W), values in [0, 1].
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1.0 - ssim_map.mean()
