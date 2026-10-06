# Adapted from klindtlab/lejepa-identifiability (MIT License, Copyright (c) 2026 David Klindt).
"""MLP encoder h = f(x), trained from scratch with the LeJEPA objective."""

import torch
import torch.nn as nn


def make_encoder(N, seed, device, hidden=768, n_layers=5):
    """GELU MLP from R^N to R^N with n_layers hidden layers of width hidden.

    Weights use the PyTorch default initialization, drawn on the CPU after
    torch.manual_seed(seed), and the module is then moved to device.
    """
    torch.manual_seed(seed)
    layers = [nn.Linear(N, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, N))
    return nn.Sequential(*layers).to(device)
