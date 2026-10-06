# Adapted from klindtlab/lejepa-identifiability (MIT License, Copyright (c) 2026 David Klindt).
"""LeJEPA objective (alignment plus SIGReg) and the encoder training loop."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import dense_r2, latent_mcc
from world import ou_step, sample_latents


class SIGReg(nn.Module):
    """Sliced characteristic-function Gaussianity regularizer (Balestriero and LeCun, 2025)."""

    def __init__(self, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-t**2 / 2))
        self.register_buffer("weights", w * torch.exp(-t**2 / 2))

    def forward(self, h):
        """h: (V, B, N) -> scalar. Draws fresh random slice directions on every call."""
        flat = h.flatten(0, 1)
        A = F.normalize(torch.randn(flat.size(-1), self.n_slices, device=flat.device), dim=0)
        xt = (flat @ A).unsqueeze(-1) * self.t
        err = (xt.cos().mean(0) - self.phi) ** 2 + xt.sin().mean(0) ** 2
        return (err @ self.weights).mean() * flat.size(0)


def alignment_loss(h):
    """Pull the views of each state together. h: (V, B, N) -> scalar."""
    return (h.mean(0) - h).square().mean()


def lr_at(step, total_steps, base_lr):
    """Constant for the first half of training, cosine decay to zero over the second half."""
    warmup = total_steps // 2
    if step < warmup:
        return base_lr
    t = (step - warmup) / (total_steps - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * t))


def encode_heldout(encoder, observe, z_eval, x_eval, rho):
    """Encode the held-out states and one fresh Ornstein-Uhlenbeck successor of each."""
    encoder.eval()
    with torch.no_grad():
        h = encoder(x_eval)
        z_next = ou_step(z_eval, rho, n_views=1).squeeze(0)
        h_next = encoder(observe(z_next))
    return h, h_next


def train_encoder(encoder, observe, z_eval, x_eval, *, N, rho, lamb, steps, batch_size, lr, log_every):
    """Train encoder in place with the LeJEPA loss on online pairs of views.

    Every step samples a fresh batch of latent states z and draws two
    independent Ornstein-Uhlenbeck successors of each. The two observed
    successors are the two views of that state, and z itself is not observed.
    The loss is lamb * SIGReg + (1 - lamb) * alignment. Every log_every steps,
    and every 100 steps during the first 1000, the training loss is logged
    together with held-out diagnostics. The diagnostics never enter the loss,
    but they draw one Ornstein-Uhlenbeck successor per held-out state from the
    global generator, so removing them would change every later training batch.
    Returns the list of log entries. Encoder restarts are compared by the loss
    of the last entry, which is the loss of the final step whenever steps is a
    multiple of log_every.
    """
    device = next(encoder.parameters()).device
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    log = []

    for step in range(steps + 1):
        current_lr = lr_at(step, steps, lr)
        for group in opt.param_groups:
            group["lr"] = current_lr

        z = sample_latents(batch_size, N, device)
        x = observe(ou_step(z, rho)).flatten(0, 1)
        h = encoder(x).reshape(2, batch_size, N)

        align = alignment_loss(h)
        sig = sigreg(h)
        loss = lamb * sig + (1 - lamb) * align

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or (step < 1000 and step % 100 == 0):
            h_eval, h_next = encode_heldout(encoder, observe, z_eval, x_eval, rho)
            encoder.train()
            entry = {
                "step": step,
                "loss": loss.item(),
                "heldout_alignment": ((h_next - h_eval) ** 2).sum(dim=1).mean().item(),
                "r2": dense_r2(z_eval, h_eval),
                "mcc": latent_mcc(z_eval, h_eval),
            }
            log.append(entry)
            if step % (log_every * 10) == 0:
                print(
                    f"  step {step:5d} | lr={current_lr:.1e} loss={entry['loss']:.4e} "
                    f"heldout_alignment={entry['heldout_alignment']:.4f} "
                    f"R2(h->z)={entry['r2']:.4f} MCC={entry['mcc']:.4f}",
                    flush=True,
                )

    return log
