"""Whitening and the DSReg rotation fit for a frozen representation."""

import torch

from .jacobian import local_decoder_jacobians
from .rotation import l1_criterion, optimize_rotation


def whiten(h, eps=1e-6):
    """Whiten a representation with the symmetric inverse square root of its covariance.

    h: (n, d) samples of the representation, a tensor or an array. Returns
    (h_white, mean, W) with h_white = (h - mean) @ W and W = Cov(h)^{-1/2},
    where Cov(h) is the sample covariance and its eigenvalues below eps are
    raised to eps. W is symmetric, and the sample covariance of h_white is the
    identity. New samples are mapped the same way, (h_new - mean) @ W. The
    computation runs in float64 and the outputs are returned in torch's
    default dtype on the device of h.
    """
    h = torch.as_tensor(h)
    dtype = torch.get_default_dtype()
    h64 = h.to(torch.float64)
    mean = h64.mean(dim=0)
    centered = h64 - mean
    cov = centered.T @ centered / (len(h64) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    W = evecs @ torch.diag(evals.clamp_min(eps).rsqrt()) @ evecs.T
    return (centered @ W).to(dtype), mean.to(dtype), W.to(dtype)


def fit_dsreg(h, x, *, k=32, ridge=1e-3, steps=3000, lr=1e-2, restarts=12):
    """Fit the DSReg rotation R of a frozen representation h.

    h: (n, d) representation, whitened (see whiten), and x: (n, p) the
    observations it was computed from. Every row is an anchor. The local
    Jacobian dx/dh at h_i is the ridge regression (penalty ridge) of x on h over
    the k nearest neighbors of h_i in representation space, and R minimizes the
    mean absolute entry of the rotated Jacobians over the orthogonal group,
    with restarts initializations (the identity first) and steps Adam steps of
    learning rate lr each. Run time grows with n, and the synthetic benchmark
    fits on 512 rows.

    The DSReg representation is z_tilde = R h, which is h @ R.T when samples
    are rows, and its coordinates estimate the individual latents up to sign
    and permutation. Returns (R, info), where info["criterion_before"] and
    info["criterion_after"] are the criterion at the identity and at R. Inputs
    are converted to torch's default dtype, and x is moved to the device of h.
    """
    dtype = torch.get_default_dtype()
    h = torch.as_tensor(h, dtype=dtype)
    x = torch.as_tensor(x, dtype=dtype, device=h.device)
    if h.dim() != 2 or x.dim() != 2 or len(h) != len(x):
        raise ValueError(f"expected h of shape (n, d) and x of shape (n, p), got {tuple(h.shape)} and {tuple(x.shape)}")
    b_xh = local_decoder_jacobians(h, x, k=k, ridge=ridge)
    criterion_before = l1_criterion(b_xh, torch.eye(h.shape[-1], device=h.device)).item()
    rotation, criterion_after = optimize_rotation(b_xh, steps=steps, lr=lr, restarts=restarts)
    return rotation, {"criterion_before": criterion_before, "criterion_after": criterion_after}
