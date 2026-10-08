# DSReg

Code for the paper [*DSReg: Provably Recovering Individual World Latents without Reconstruction*](https://arxiv.org/abs/2610.09457). Project page: https://dsreg.github.io/

DSReg (Dependency-Sparsity Regularization) starts from a representation that is identified up to an orthogonal transformation, which LeJEPA provides for Gaussian latent worlds, and selects the rotation under which the observations depend on the fewest latents. When different latents leave distinct dependency footprints on the observations (Structural Diversity), the selected representation recovers each individual world latent up to sign and permutation, with no decoder, reconstruction, or labels.

The repository contains an implementation of DSReg, a Lean 4 formalization of the paper's identifiability theorem, and the paper's synthetic benchmark with learned encoders.

## Using DSReg

```bash
pip install -e .
```

DSReg takes a frozen representation `h` of shape `(n, d)` and the observations `x` of shape `(n, p)` it was computed from. It estimates the Jacobian of the observations with respect to the representation by local ridge regressions over nearest neighbors, and then searches the orthogonal group for the rotation that makes these Jacobians sparsest in the l1 sense.

```python
from dsreg import fit_dsreg, whiten

h, mean, W = whiten(h)       # frozen representation, shape (n, d)
R, info = fit_dsreg(h, x)    # observations x, shape (n, p)
z_tilde = h @ R.T            # individual latents, up to sign and permutation
```

`fit_dsreg` also accepts the neighborhood size `k`, the ridge penalty, and the number of optimization steps and restarts. The neighborhood must contain more than `d` points.

## The identifiability theorem in Lean

`lean/` proves the paper's identifiability theorem in Lean 4 with mathlib. The statement is built from the paper's objects: the observation map `g` and its Jacobian, the representation `h` with the LeJEPA premise `h(z) = Qz`, the candidate latents `R h(z)`, and the support count of the Jacobian of the observations with respect to the candidate latents. `DSReg.identifiability` shows that every orthogonal minimizer `R` of this count makes `RQ` a signed permutation under Structural Diversity and Functional no-cancellation, and `DSReg.exists_minimizer` shows that a minimizer exists. Two certificates check the theorem on concrete instances under the standard Gaussian law, one with nested footprints and a nonlinear `g` and one showing that the conclusion fails when two footprints coincide.

```bash
cd lean
lake exe cache get
lake build
```

The proofs use no `sorry` and no axioms beyond the three standard ones. `lean/README.md` maps each object in the paper to its Lean name and states the scope of the formalization.

## Synthetic benchmark

`experiments/synthetic/` trains an encoder from scratch with the LeJEPA objective on consecutive states of a synthetic Gaussian world, fits DSReg from the observations and the frozen representation alone, and reports latent MCC before and after the rotation on held-out samples. The observation map is `x_j = z_j + f_j(z_<j)` with small random MLPs `f_j`, so latent `z_k` affects `x_k, ..., x_N` and the footprints are nested, a regime that earlier structural conditions exclude. The benchmark sweeps the latent dimension `N` over 4, 6, 8, 10, 12 and 14, with seeds 0 to 19 for each.

```bash
pip install -r requirements.txt
python experiments/synthetic/run.py --N 8 --seed 3
```

The defaults are the settings used in the paper. A run takes about 8 to 11 minutes on one NVIDIA L40 and writes `results/synthetic/N8_seed3.json`. On a SLURM cluster, the full sweep of 120 runs and its summary are

```bash
sbatch --partition=<gpu-partition> experiments/synthetic/sweep.sbatch
python experiments/synthetic/aggregate.py    # results/synthetic.csv
python experiments/synthetic/plot.py         # results/synthetic.pdf
```

`experiments/synthetic/paper_results.csv` holds the numbers reported in the paper, and every column that `aggregate.py` writes appears there under the same name. The table below gives the mean and standard deviation over the twenty seeds.

| N | LeJEPA MCC | DSReg MCC |
|---|---|---|
| 4 | 0.769 ± 0.074 | 0.997 ± 0.000 |
| 6 | 0.684 ± 0.051 | 0.995 ± 0.001 |
| 8 | 0.640 ± 0.031 | 0.994 ± 0.001 |
| 10 | 0.596 ± 0.031 | 0.993 ± 0.000 |
| 12 | 0.557 ± 0.024 | 0.993 ± 0.002 |
| 14 | 0.524 ± 0.021 | 0.973 ± 0.046 |

We reran all 120 runs with this code and torch 2.5.1 on NVIDIA L40 and L40S GPUs, on two separate clusters. Every run reproduced its record behind the paper bit for bit, and `plot.py` redraws the paper's figure pixel for pixel. Individual runs can come out differently on other GPU models.

## Citation

```bibtex
@article{zheng2026dsreg,
  title   = {DSReg: Provably Recovering Individual World Latents without Reconstruction},
  author  = {Zheng, Yujia and Klindt, David and Balestriero, Randall and Sch{\"o}lkopf, Bernhard},
  journal = {arXiv preprint arXiv:2610.09457},
  year    = {2026}
}
```

## Third-party code

Parts of `experiments/synthetic/` are adapted from [klindtlab/lejepa-identifiability](https://github.com/klindtlab/lejepa-identifiability) under the MIT License. See `THIRD_PARTY_NOTICES.md`.
