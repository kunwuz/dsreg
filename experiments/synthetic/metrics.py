# Adapted from klindtlab/lejepa-identifiability (MIT License, Copyright (c) 2026 David Klindt).
"""Evaluation metrics: latent MCC and dense R^2."""

import torch
from scipy.optimize import linear_sum_assignment


def latent_mcc(z, h):
    """Mean absolute Pearson correlation after one-to-one matching.

    Columns of h are matched to columns of z by the Hungarian algorithm on the
    absolute correlation matrix, and the matched absolute correlations are
    averaged. The score is 1 exactly when every matched coordinate of h is an
    affine function of its latent.
    """
    eps = 1e-8
    z_std = (z - z.mean(dim=0)) / z.std(dim=0, unbiased=True).clamp_min(eps)
    h_std = (h - h.mean(dim=0)) / h.std(dim=0, unbiased=True).clamp_min(eps)
    corr = (z_std.T @ h_std) / (len(z_std) - 1)
    abs_corr = corr.abs().clamp(max=1.0)
    rows, cols = linear_sum_assignment(-abs_corr.detach().cpu().numpy())
    row_t = torch.tensor(rows.tolist(), device=z.device)
    col_t = torch.tensor(cols.tolist(), device=h.device)
    return corr[row_t, col_t].abs().mean().item()


def dense_r2(z, h):
    """R^2 of the best affine prediction of z from h.

    It measures whether h retains the latent state as a span. In exact
    arithmetic it is invariant to any invertible linear map of h, so the DSReg
    rotation leaves it unchanged.
    """
    h1 = torch.cat([h, torch.ones(len(h), 1, device=h.device)], dim=1)
    W = torch.linalg.lstsq(h1, z).solution
    ss_res = ((z - h1 @ W) ** 2).sum()
    ss_tot = ((z - z.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()
