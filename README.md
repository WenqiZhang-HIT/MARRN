# MARRN — core model implementation

**Missing-Aware Relational Reconstruction Network**

Core implementation accompanying *Beyond Missingness-Induced Deviations:
Anomaly Detection via Missing-Aware Relational Reconstruction in Cyber-Physical Systems*.

This repository extracts the architecture, complementary learning objective,
and observed-residual scoring of the selected V13 model. It is a model-only
release, **not a complete reproduction pipeline for the reported metrics**.

## Included

- Availability-conditioned, fully connected directed graph attention with self-loops.
- Fixed-order flattening, a global Gaussian latent bottleneck, and an MLP decoder.
- Point-wise OR sensor-wise artificial masking, selected per training window.
- Separately normalized visible/hidden MSE, batch-wise free bits, normalized KL and warm-up.
- Maximum observed sensor-level MSE as a continuous anomaly score.

Dataset loading, standardization, window generation, evaluation splits, model
selection, thresholds, postprocessing, metric computation and checkpoints are
intentionally excluded. No dataset-specific threshold is embedded in the model.

## Defaults from the selected V13 configuration

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

Inputs are already standardized float tensors `[batch, window, sensor]`; the
legacy `[batch, window, sensor, 1]` interface is also accepted. Masks are binary
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

## Implementation fidelity and scope

- Parameter names match the selected legacy backbone (`gat_layers`, `fc_mu`,
  `fc_logvar`, `decoder`). Matching-shape legacy model state dictionaries can be
  loaded directly. State tensors, not random initialization sequences, define
  checkpoint equivalence.
- Decoder coordinates preserve the legacy **sensor-major output then transpose**.
- Training samples latent variables; evaluation decodes the posterior mean.
- Edge reliability is a learned attention feature, not a hard gate or a measured
  probability of physical dependency. Zero reliability does not delete an edge.
- The artificial masking distribution matches the original sampler; this release
  uses PyTorch RNG instead of NumPy and does not reproduce identical mask draws
  from the same numeric seed.
- Sensor MSE uses observed entries only. Entirely unobserved sensors do not supply
  residual evidence. An all-missing window has a zero-score convention, not a
  certified normal decision.
- No posterior KL score or two-dimensional alarm rule is part of the V13 scorer.
- The original selected model used normal-data fitting and labeled validation
  selection. Removing that pipeline does not make the original experiment fully
  label-free or reproduce its final F1.

## Dependencies and release status

Install PyTorch for your compute platform, then PyTorch Geometric. The original
training environment recorded PyTorch 2.8.0+cu128 on an RTX 5080. This release
does not require CUDA for use. Exact GPU determinism across versions is not promised.

A license is deliberately not chosen on the authors' behalf. Add an appropriate
LICENSE before describing the public repository as licensed open-source software.
