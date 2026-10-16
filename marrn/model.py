"""V13's availability-conditioned GAT and global variational reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class MARRN(nn.Module):
    """Inputs are standardized windows [B, W, N] or legacy [B, W, N, 1].

    Masks are binary, with one indicating an observed measurement. Outputs
    follow the input window shape. Sensor ordering must remain fixed.
    Parameter names and decoder ordering match the selected V13 backbone.
    """

    def __init__(self, n_sensors=51, window_size=30, hidden_dim=64,
                 latent_dim=32, num_layers=2, heads=4, dropout=0.1):
        super().__init__()
        if min(n_sensors, window_size, hidden_dim, latent_dim, num_layers, heads) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.gat_layers = nn.ModuleList([
            GATConv(window_size if i == 0 else hidden_dim, hidden_dim,
                    heads=heads, concat=False, dropout=dropout, edge_dim=1,
                    add_self_loops=True, fill_value=1.0)
            for i in range(num_layers)
        ])
        for layer in self.gat_layers:
            # Only att_edge is zeroed; lin_edge retains its random initialization.
            nn.init.zeros_(layer.att_edge)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(n_sensors * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(n_sensors * hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_sensors * window_size),
        )
        edges = torch.tensor([(j, i) for j in range(n_sensors)
                              for i in range(n_sensors) if j != i], dtype=torch.long)
        self.register_buffer("base_edges", edges.reshape(-1, 2).t().contiguous(),
                             persistent=False)

    def _prepare(self, x, mask):
        if x.shape != mask.shape:
            raise ValueError("Window and mask shapes must match.")
        legacy = x.ndim == 4
        if legacy:
            if x.shape[-1] != 1:
                raise ValueError("The legacy trailing channel dimension must be 1.")
            x, mask = x.squeeze(-1), mask.squeeze(-1)
        if x.ndim != 3 or x.shape[1:] != (self.window_size, self.n_sensors):
            raise ValueError("Expected [B, window_size, n_sensors] windows.")
        # Placeholder handling, not statistical imputation. Observed zeros stay valid.
        return torch.where(mask.bool(), x, 0.0), mask.to(x.dtype), legacy

    def encode(self, x, mask):
        x, mask, _ = self._prepare(x, mask)
        batch = x.shape[0]
        h = x.transpose(1, 2).contiguous().reshape(batch * self.n_sensors,
                                                self.window_size)
        availability = mask.mean(1).reshape(-1)
        offsets = torch.arange(batch, device=x.device).view(-1, 1, 1) * self.n_sensors
        edges = (self.base_edges.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        reliability = (availability[edges[0]] * availability[edges[1]]).unsqueeze(-1)
        for layer in self.gat_layers:
            h = self.dropout(F.elu(layer(h, edges, edge_attr=reliability)))
        h = h.reshape(batch, -1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)

    def decode(self, z):
        # Legacy decoder emits sensor-major coordinates, then transposes to time-major.
        return self.decoder(z).reshape(-1, self.n_sensors, self.window_size).transpose(1, 2).contiguous()

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        recon = self.decode(z)
        return (recon.unsqueeze(-1) if x.ndim == 4 else recon), mu, logvar
