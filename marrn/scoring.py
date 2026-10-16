"""Observed-only top1 reconstruction score; no alarm calibration/postprocessing."""

import torch


def observed_residual_score(truth, recon, observed):
    """Return window score, per-sensor MSE and observed counts.

    Unavailable sensors contribute zero. A fully unavailable window receives
    zero by the original convention, indicating no evidence, not normality.
    """
    if not (truth.shape == recon.shape == observed.shape):
        raise ValueError("Window, reconstruction and mask shapes must match.")
    if truth.ndim == 4:
        truth, recon, observed = truth.squeeze(-1), recon.squeeze(-1), observed.squeeze(-1)
    if truth.ndim != 3:
        raise ValueError("Expected [B, W, N] or [B, W, N, 1].")
    mask = observed.float()
    delta = torch.where(mask.bool(), truth.float() - recon.float(), 0.0)
    count = mask.sum(1)
    sensor_mse = (delta.square() * mask).sum(1) / count.clamp_min(1)
    sensor_mse = torch.where(count > 0, sensor_mse, 0.0)
    return sensor_mse.max(1).values, sensor_mse, count
