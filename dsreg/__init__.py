"""DSReg: Dependency-Sparsity Regularization for recovering individual world latents."""

from .fit import fit_dsreg, whiten
from .jacobian import local_decoder_jacobians
from .rotation import l1_criterion, optimize_rotation

__version__ = "0.1.0"

__all__ = ["fit_dsreg", "whiten", "local_decoder_jacobians", "l1_criterion", "optimize_rotation"]
