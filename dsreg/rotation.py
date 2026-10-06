"""The DSReg l1 criterion and its optimizer over the orthogonal group O(d)."""

import math

import torch


def l1_criterion(b_xh, rotation):
    """Mean absolute entry of the rotated Jacobians dx/d(Rh) = B R^T.

    b_xh: (num_anchors, x_dim, d) local Jacobians dx/dh; rotation: (d, d),
    applied to the representation as y = R h.
    """
    return torch.einsum("bij,kj->bik", b_xh, rotation).abs().mean()


def random_orthogonal(n, device):
    """Haar-distributed random orthogonal matrix (QR of a Gaussian matrix, sign-corrected)."""
    q, r = torch.linalg.qr(torch.randn(n, n, device=device))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def optimize_rotation(b_xh, *, steps=3000, lr=1e-2, restarts=12):
    """Minimize the l1 criterion over orthogonal R.

    Each restart writes R = exp(A - A^T) R_0 with A initialized at zero, starts
    from R_0 = I for the first restart and from a random orthogonal R_0 for the
    others, and runs Adam on A with gradient-norm clipping at 10. Returns the
    rotation with the lowest criterion and that criterion value. The identity
    is kept unless some restart improves on it.
    """
    device = b_xh.device
    n = b_xh.shape[-1]
    best_rotation = torch.eye(n, device=device)
    best_score = l1_criterion(b_xh, best_rotation).item()

    starts = [torch.eye(n, device=device)] + [
        random_orthogonal(n, device=device) for _ in range(max(restarts - 1, 0))
    ]

    for start in starts:
        raw = torch.zeros(n, n, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)
        for _ in range(steps):
            skew = raw - raw.T
            rotation = torch.linalg.matrix_exp(skew) @ start
            loss = l1_criterion(b_xh, rotation)
            if not torch.isfinite(loss):
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], max_norm=10.0)
            opt.step()

        with torch.no_grad():
            skew = raw - raw.T
            rotation = torch.linalg.matrix_exp(skew) @ start
            score = l1_criterion(b_xh, rotation).item()
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_rotation = rotation.detach().clone()

    return best_rotation, best_score
