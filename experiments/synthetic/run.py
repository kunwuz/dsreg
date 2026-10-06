# Adapted from klindtlab/lejepa-identifiability (MIT License, Copyright (c) 2026 David Klindt).
"""One (N, seed) run of the synthetic benchmark with learned encoders, LeJEPA followed by DSReg.

From the repository root,

    python experiments/synthetic/run.py --N 8 --seed 3

trains the encoder, fits the DSReg rotation, evaluates both representations on
held-out states and writes results/synthetic/N8_seed3.json. The defaults are
the protocol used for the paper.
"""

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

if importlib.util.find_spec("dsreg") is None:  # not installed: use the copy in this repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dsreg import fit_dsreg
from encoder import make_encoder
from lejepa import encode_heldout, train_encoder
from metrics import dense_r2, latent_mcc
from world import make_observation_map, sample_latents

PROTOCOL = {
    # world
    "obs_hidden": 16,
    "obs_strength": 0.08,
    "rho": 0.95,
    # LeJEPA encoder and its training
    "hidden": 768,
    "n_layers": 5,
    "steps": 20000,
    "batch_size": 256,
    "lr": 3e-3,
    "lamb": 0.01,
    "log_every": 500,
    "encoder_restarts": 6,
    # held-out evaluation
    "num_eval": 5000,
    # DSReg
    "jac_samples": 512,
    "knn": 32,
    "local_ridge": 1e-3,
    "rot_steps": 3000,
    "rot_lr": 1e-2,
    "rot_restarts": 12,
}

HELP = {
    "obs_hidden": "hidden width of each f_j in the observation map x_j = z_j + f_j(z_<j)",
    "obs_strength": "scale of each f_j in the observation map",
    "rho": "Ornstein-Uhlenbeck correlation between a latent state and its successor",
    "hidden": "encoder hidden width",
    "n_layers": "number of encoder hidden layers",
    "steps": "encoder training steps",
    "batch_size": "latent states per training batch, each giving two views",
    "lr": "AdamW learning rate, constant for the first half of training, then cosine decay to zero",
    "lamb": "SIGReg weight lamb in the loss lamb * SIGReg + (1 - lamb) * alignment",
    "log_every": "log the training loss every this many steps (and every 100 steps during the first 1000)",
    "encoder_restarts": "encoder initializations; the one with the lowest training loss at the last logged step is kept",
    "num_eval": "held-out states used for evaluation",
    "jac_samples": "anchor states at which the local Jacobians dx/dh are estimated",
    "knn": "nearest neighbors in representation space per local Jacobian fit",
    "local_ridge": "ridge penalty of each local Jacobian fit",
    "rot_steps": "Adam steps per rotation restart",
    "rot_lr": "Adam learning rate of the rotation",
    "rot_restarts": "rotation restarts; the first starts at the identity, the others at random rotations",
}


def train_lejepa(N, seed, cfg, device):
    """Train the encoder from several initializations and keep the best one.

    The observation map and the held-out set are fixed by the seed. Restart r
    initializes the encoder from seed + 77777 + 100003 * r, and the restart with
    the lowest training loss at the last logged step is kept (with the default
    settings, the last logged step is the final step). Ground-truth latents
    never enter this selection.
    """
    observe = make_observation_map(N, seed, device, hidden=cfg["obs_hidden"], strength=cfg["obs_strength"])
    base_seed = seed + 77777
    encoder = make_encoder(N, base_seed, device, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    z_eval = sample_latents(cfg["num_eval"], N, device)
    x_eval = observe(z_eval)

    best, best_loss, selected, final_losses = None, float("inf"), 0, []
    for r in range(cfg["encoder_restarts"]):
        if r > 0:
            encoder = make_encoder(N, base_seed + 100003 * r, device, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
        print(f"encoder restart {r}", flush=True)
        log = train_encoder(
            encoder,
            observe,
            z_eval,
            x_eval,
            N=N,
            rho=cfg["rho"],
            lamb=cfg["lamb"],
            steps=cfg["steps"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            log_every=cfg["log_every"],
        )
        final = log[-1]["loss"]
        final_losses.append(final)
        if final < best_loss:
            best, best_loss, selected = encoder, final, r
    return best, observe, z_eval, x_eval, final_losses, selected


def fit_rotation(encoder, observe, N, cfg, device):
    """Fit DSReg on the representation of fresh anchor states.

    Returns the rotation and the DSReg criterion at the identity (LeJEPA) and
    at the rotation (DSReg).
    """
    z_jac = sample_latents(cfg["jac_samples"], N, device)
    x_jac = observe(z_jac)
    encoder.eval()
    with torch.no_grad():
        h_jac = encoder(x_jac)
    rotation, info = fit_dsreg(
        h_jac,
        x_jac,
        k=cfg["knn"],
        ridge=cfg["local_ridge"],
        steps=cfg["rot_steps"],
        lr=cfg["rot_lr"],
        restarts=cfg["rot_restarts"],
    )
    return rotation, info["criterion_before"], info["criterion_after"]


def run(N, seed, cfg, device):
    """Train LeJEPA, fit DSReg and evaluate both representations on the held-out states."""
    start_time = time.time()
    encoder, observe, z_eval, x_eval, final_losses, selected = train_lejepa(N, seed, cfg, device)
    rotation, lejepa_dep, dsreg_dep = fit_rotation(encoder, observe, N, cfg, device)

    h, h_next = encode_heldout(encoder, observe, z_eval, x_eval, cfg["rho"])
    h_rot = h @ rotation.T
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()

    return {
        "N": N,
        "seed": seed,
        **cfg,
        "lejepa_mcc": latent_mcc(z_eval, h),
        "dsreg_mcc": latent_mcc(z_eval, h_rot),
        "lejepa_r2": dense_r2(z_eval, h),
        "dsreg_r2": dense_r2(z_eval, h_rot),
        "lejepa_dep_sparsity": lejepa_dep,
        "dsreg_dep_sparsity": dsreg_dep,
        "heldout_alignment": ((h_next - h) ** 2).sum(dim=1).mean().item(),
        "restart_final_losses": final_losses,
        "selected_restart": selected,
        "torch_version": torch.__version__,
        "device": device.type,
        "device_name": device_name,
        "wall_time_s": time.time() - start_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="One run of the synthetic benchmark with learned encoders: LeJEPA, then DSReg."
    )
    parser.add_argument("--N", type=int, required=True, help="latent dimension")
    parser.add_argument(
        "--seed", type=int, required=True, help="fixes the observation map, all sampled data and all initializations"
    )
    for name, default in PROTOCOL.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(default), default=default, help=HELP[name] + " (default: %(default)s)"
        )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="the paper's runs used cuda; cpu runs are slower and do not reproduce the GPU numbers (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("results", "synthetic"),
        help="directory for the JSON record N<N>_seed<seed>.json (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.encoder_restarts < 1:
        parser.error("--encoder-restarts must be at least 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Pass --device cpu to run on the CPU.")
    cfg = {name: getattr(args, name) for name in PROTOCOL}
    record = run(args.N, args.seed, cfg, torch.device(args.device))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"N{args.N}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {path}")
    print(
        f"MCC LeJEPA={record['lejepa_mcc']:.4f} DSReg={record['dsreg_mcc']:.4f} | "
        f"dense R2 LeJEPA={record['lejepa_r2']:.4f} DSReg={record['dsreg_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
