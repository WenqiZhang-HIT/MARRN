"""Complementary masking, V13 objective, and default optimizer settings."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 96
    max_epochs: int = 40
    gradient_clip_norm: float = 2.0
    masking_rate: float = 0.2
    point_mask_probability: float = 0.5
    hidden_weight: float = 1.0
    beta: float = 0.1
    free_bits: float = 0.05
    warmup_epochs: int = 10


def complementary_masks(observed, rate=0.2, point_probability=0.5, generator=None):
    """Choose point-wise OR sensor-wise corruption independently for each window.

    This is training-time augmentation, not a dataset missingness generator.
    Randomness uses PyTorch rather than the original NumPy dataset sampler;
    the masking distribution is preserved, not the seed-by-seed draw sequence.
    """
    if not 0 <= rate <= 1 or not 0 <= point_probability <= 1:
        raise ValueError("Mask probabilities must lie in [0, 1].")
    legacy = observed.ndim == 4
    mask = observed.squeeze(-1) if legacy else observed
    if mask.ndim != 3:
        raise ValueError("Expected [B, W, N] or [B, W, N, 1].")
    keep = torch.empty_like(mask)
    for b in range(len(mask)):
        point = torch.rand((), device=mask.device, generator=generator) < point_probability
        shape = mask[b].shape if point else (1, mask.shape[2])
        keep[b] = (torch.rand(shape, device=mask.device, generator=generator) >= rate).to(mask.dtype)
    visible = mask * keep
    hidden = mask - visible
    return (visible.unsqueeze(-1), hidden.unsqueeze(-1)) if legacy else (visible, hidden)


def kl_warmup(epoch_index, warmup_epochs=10):
    """Zero-based epoch: epoch 0 has zero KL weight; epoch 10 has full weight."""
    if epoch_index < 0 or warmup_epochs < 0:
        raise ValueError("Epoch values must be nonnegative.")
    return 1.0 if warmup_epochs == 0 else min(1.0, epoch_index / warmup_epochs)


def complementary_loss(truth, recon, visible, hidden, mu, logvar,
                       beta=0.1, kl_weight=1.0, free_bits=0.05, hidden_weight=1.0):
    """Targets are originally observed values, including artificially hidden ones.

    No genuinely missing ground truth is accessed. KL free bits are applied
    AFTER averaging each latent dimension over the batch, exactly as in V13.
    """
    if not (truth.shape == recon.shape == visible.shape == hidden.shape):
        raise ValueError("Targets, reconstruction and target masks must match.")
    axes = tuple(range(1, truth.ndim))
    v, u = visible.float(), hidden.float()
    delta = torch.where((v + u) > 0, truth.float() - recon.float(), 0.0)
    squared = delta.square()
    visible_count = v.sum(axes).clamp_min(1)
    rec_visible = ((squared * v).sum(axes) / visible_count).mean()
    hidden_count = u.sum(axes)
    valid = hidden_count > 0
    hidden_per = (squared * u).sum(axes) / hidden_count.clamp_min(1)
    rec_hidden = (hidden_per * valid).sum() / valid.sum().clamp_min(1)
    dimensions = 0.5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float())
    kl_natural = dimensions.sum(1).mean()
    kl_regularized = dimensions.mean(0).clamp_min(free_bits).sum() / visible_count.mean()
    total = rec_visible + hidden_weight * rec_hidden + beta * kl_weight * kl_regularized
    return {"loss": total, "visible_mse": rec_visible, "hidden_mse": rec_hidden,
            "kl_natural": kl_natural, "kl_regularized": kl_regularized}


def make_optimizer(model, config=TrainingConfig()):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)
