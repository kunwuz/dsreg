# Adapted from klindtlab/lejepa-identifiability (MIT License, Copyright (c) 2026 David Klindt).
"""The world of the synthetic benchmark with learned encoders.

Latent states are standard Gaussian in R^N, consecutive states follow one
stationary Ornstein-Uhlenbeck step, and observations come from a random
triangular residual MLP x_j = z_j + f_j(z_0, ..., z_{j-1}).
"""

import torch


def sample_latents(num, N, device):
    """Draw num independent standard Gaussian latent states in R^N."""
    return torch.randn(num, N, device=device)


def ou_step(z, rho, n_views=2):
    """Successor states z' = rho * z + sqrt(1 - rho^2) * eta with Gaussian eta.

    Draws n_views independent successors of every row of z and returns a
    tensor of shape (n_views, len(z), N).
    """
    fac = (1 - rho ** 2) ** 0.5
    D, N = z.shape
    eta = sample_latents(n_views * D, N, z.device)
    eta = eta.reshape(n_views, D, N)
    return rho * z.unsqueeze(0) + fac * eta


def make_observation_map(N, seed, device, hidden=16, strength=0.08):
    """Random observation map x = g(z) with nested latent footprints.

    Coordinate j is x_j = z_j + strength * w_j^T tanh(W_j^T z_{<j} + b_j), so
    x_j depends on z_0, ..., z_j and latent k has footprint {k, ..., N-1}.
    The footprints are pairwise distinct and nested for every N.
    """
    torch.manual_seed(seed)
    params = []
    for j in range(N):
        # x_0 = z_0 has no parents. Its parameters are still drawn, so that the
        # random stream, and hence the map of every seed, is the one used in the paper.
        width = max(j, 1)
        W1 = torch.randn(width, hidden, device=device) / (width ** 0.5)
        b1 = 0.25 * torch.randn(hidden, device=device)
        w2 = torch.randn(hidden, device=device) / (hidden ** 0.5)
        params.append((torch.arange(j, device=device), W1, b1, w2))

    def observe(z):
        x = z.clone()
        for j in range(1, N):
            parents, W1, b1, w2 = params[j]
            x[..., j] = z[..., j] + strength * (torch.tanh(z[..., parents] @ W1 + b1) @ w2)
        return x

    return observe
