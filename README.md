# MARRN — core model implementation

**Missing-Aware Relational Reconstruction Network**

Core implementation accompanying *Beyond Missingness-Induced Deviations:
Anomaly Detection via Missing-Aware Relational Reconstruction in Cyber-Physical Systems*.

## Included

- Availability-conditioned, fully connected directed graph attention with self-loops.
- Fixed-order flattening, a global Gaussian latent bottleneck, and an MLP decoder.
- Point-wise OR sensor-wise artificial masking, selected per training window.
- Separately normalized visible/hidden MSE, batch-wise free bits, normalized KL and warm-up.
- Maximum observed sensor-level MSE as a continuous anomaly score.

## Default value example: SWaT dataset configuration

| Setting | Default |
|---|---|
| Sensors / window length | 51 / 30 |
| GAT layers / heads / head aggregation | 2 / 4 / mean |
| Hidden / latent dimensions | 64 / 32 |
| Attention and output dropout | 0.1 |
| Decoder widths | 32 → 64 → 64 → (N × W) |
| Decoder activations | ReLU; dropout after the first activation |
| Cross-sensor edge feature | product of window availability fractions |
| Self-loop edge feature | 1 |
| Edge attention initialization | zero att_edge; random lin_edge |
| Artificial masking rate / point-wise selection probability | 0.2 / 0.5 |
| Visible / hidden reconstruction weights | 1 / 1 |
| KL coefficient / free bits | 0.1 / 0.05 nats per dimension |
| KL warm-up | min(1, zero-based epoch / 10) |
| AdamW learning rate / weight decay | 8e-4 / 1e-4 |
| Batch size / maximum epochs / gradient clipping norm | 96 / 40 / 2 |
| Original training seed | 42 |

These are the selected configuration's defaults, not a guarantee of optimality
on other datasets. Sensor count may be changed; the flatten and output dimensions
are computed automatically. Keep the same sensor order during training and inference.

## Interface

Inputs are already standardized float tensors `[batch, window, sensor]`. Masks are binary
with **1 = observed**. A real measurement of zero is valid. The model applies the
mask internally, using zero placeholders rather than estimating missing inputs.
Observed entries must be finite. Masks, values and model must share a device.

```python
import torch
from marrn import (MARRN, TrainingConfig, complementary_masks,
                   complementary_loss, kl_warmup, make_optimizer,
                   observed_residual_score)

model = MARRN(n_sensors=51)
cfg = TrainingConfig()
optimizer = make_optimizer(model, cfg)

# x and observed are user-supplied, already windowed tensors.
# x contains the ORIGINAL observed targets, not only the additionally masked input.
model.train()
visible, hidden = complementary_masks(observed, cfg.masking_rate,
                                       cfg.point_mask_probability)
recon, mu, logvar = model(x, visible)
losses = complementary_loss(x, recon, visible, hidden, mu, logvar,
                            beta=cfg.beta, free_bits=cfg.free_bits,
                            hidden_weight=cfg.hidden_weight,
                            kl_weight=kl_warmup(epoch_index, cfg.warmup_epochs))
optimizer.zero_grad(set_to_none=True)
losses['loss'].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
optimizer.step()

model.eval()
with torch.no_grad():
    recon, mu, logvar = model(x, observed)
    scores, sensor_errors, counts = observed_residual_score(x, recon, observed)
```

The example illustrates the core operations; it is not a training command.
Original runs used automatic mixed precision. If adding AMP, unscale gradients
before clipping. Checkpoint selection and validation scheduling are intentionally
outside this model-only API.

## Dependencies and release status

Install PyTorch for your compute platform, then PyTorch Geometric. This release
does not require CUDA for use. Exact GPU determinism across versions is not promised.

A license is deliberately not chosen on the authors' behalf.
