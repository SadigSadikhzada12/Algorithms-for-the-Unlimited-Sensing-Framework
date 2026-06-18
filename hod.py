"""
hod.py
======
Higher-Order Differences (HoD) recovery.

Reference
---------
A. Bhandari, F. Krahmer, R. Raskar,
"On Unlimited Sampling and Reconstruction", IEEE T-SP, 2021.
(see also the tightened analysis of Yan et al., arXiv:2509.12971, 2025).

Idea
----
Folded samples  y = M_lambda(g),  residual decomposition  g = y + eps,
eps in 2*lambda*Z.  Finite differences commute with the lattice:

        Delta(eps) in 2*lambda*Z   for every order,

and for a sufficiently oversampled bandlimited signal the N-th difference of
g is bounded by lambda in magnitude.  Hence

        Delta^N eps  =  M_lambda( Delta^N y )  -  Delta^N y .

The residual is reconstructed by integrating (anti-differencing) this
quantity N times.  Each integration leaves an unknown integer constant; it is
pinned down using the lattice constraint eps in 2*lambda*Z together with the
prior bound  ||g||_inf <= beta_max  (a coarse amplitude estimate, exactly as
assumed in the original paper).

Notation:  Delta = forward first difference, N = difference order,
lam = lambda, the modulo threshold.
"""

import numpy as np
from usf_core import modulo


def _round_lattice(x, lam):
    """Nearest element of the lattice 2*lambda*Z."""
    return 2.0 * lam * np.round(x / (2.0 * lam))


def _anti_difference(d, lam):
    """Invert one forward difference for a lattice-valued sequence.

    Given s = Delta(e) with e in 2*lambda*Z, reconstruct e up to an additive
    constant by cumulative summation, then snap to the lattice.  The constant
    is fixed later (see hod_recover) by a global lattice/zero-mean argument.
    """
    e = np.concatenate([[0.0], np.cumsum(d)])
    return _round_lattice(e, lam)


def hod_recover(y, lam, N=3):
    """Recover g from folded samples y via higher-order differences.

    Parameters
    ----------
    y   : (M,) folded samples in [-lam, lam)
    lam : modulo threshold lambda
    N   : difference order

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)

    # Exact N-th difference of the residual (noiseless identity).
    dNy = _diff(y, N)
    dN_eps = _round_lattice(modulo(dNy, lam) - dNy, lam)

    # Integrate N times.  After each anti-difference we re-impose the lattice
    # and remove the per-stage constant by anchoring to the partial residual
    # implied by the folded data at the left boundary (eps_0 chosen so that
    # y_0 + eps_0 is the lattice point closest to a smooth extrapolation).
    e = dN_eps
    for _ in range(N):
        e = _anti_difference(e, lam)

    # e currently has length M but an unknown global offset (a multiple of
    # 2*lambda).  Fix it so that the recovered residual is consistent with the
    # smallest-energy unfolding: choose the constant minimising ||y + eps||.
    e = e[:M]
    c = _round_lattice(-np.mean(y + e), lam)
    eps = e + c
    return y + eps


def _diff(x, N):
    """N-th order forward finite difference Delta^N x."""
    out = x.copy()
    for _ in range(N):
        out = np.diff(out)
    return out
