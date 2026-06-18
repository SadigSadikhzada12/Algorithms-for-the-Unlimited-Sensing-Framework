"""
adaptive_local.py
=================
Noisy recovery via Adaptive Modulo Representations / Adaptive Local
Representations.

Reference
---------
F. Pagginelli Patricio, P. Catala, F. Krahmer,
"Noisy Recovery in Unlimited Sampling via Adaptive Modulo Representations",
CoSeRa 2024;  and
"Stable Retrieval for Unlimited Sampling via Adaptive Local Representations",
IEEE SSP 2025.

Idea
----
This is the only line of work that explicitly targets the PRE-modulo noise
model y = M_lambda(g + eta).  The domain is partitioned into many short
sub-intervals.  On each sub-interval the algorithm

  1. applies a local re-centring (an adaptive shift of the modulo range): the
     window is unfolded so that it continues smoothly from the running
     estimate, removing the strong artefacts noise creates near fold
     boundaries; and
  2. fits the de-folded data by least squares against a low-degree polynomial
     basis (a local low-order model for the bandlimited signal), which denoises
     the window.

These two steps are alternated.  Per the SSP 2025 analysis there is a
trade-off: more (shorter) sub-intervals improve the local-model fit but worsen
the conditioning (noise amplification) of each least-squares solve.

Implementation
--------------
Windows are swept left to right.  Within a window the folded data is first
locally unwrapped (np.unwrap-style, but on the 2*lambda lattice) to remove
intra-window folds, then re-centred onto the previous window's last value, then
least-squares de-noised with a low-degree polynomial.  A few sweeps refine it.
"""

import numpy as np


def _lattice_unwrap(seg, lam):
    """Remove 2*lambda jumps within a window by unwrapping consecutive diffs."""
    d = np.diff(seg)
    d_unwrapped = d - 2.0 * lam * np.round(d / (2.0 * lam))
    return seg[0] + np.concatenate([[0.0], np.cumsum(d_unwrapped)])


def _poly_denoise(t, y, degree):
    V = np.vander(t, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, y, rcond=None)
    return V @ coef


def adaptive_local_recover(y, lam, n_intervals=24, degree=2, n_iter=3):
    """Recover g from (possibly pre-modulo-noisy) folded samples.

    Parameters
    ----------
    y           : (M,) folded samples
    lam         : modulo threshold lambda
    n_intervals : number of sub-intervals the domain is split into
    degree      : polynomial degree of the local model
    n_iter      : number of refinement sweeps

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)
    bounds = np.linspace(0, M, n_intervals + 1).astype(int)
    g_hat = np.array(y, dtype=float)

    for it in range(n_iter):
        prev_val = None
        for b in range(n_intervals):
            lo, hi = bounds[b], bounds[b + 1]
            w = hi - lo
            if w < degree + 2:
                continue
            t = np.linspace(0, 1, w)

            # 1. Local de-fold: unwrap intra-window jumps on the lattice.
            seg = _lattice_unwrap(y[lo:hi], lam)

            # 2. Adaptive re-centring: shift the whole window by a multiple of
            #    2*lambda so it continues from the previous window's last value.
            if prev_val is not None:
                shift = 2.0 * lam * np.round((prev_val - seg[0]) / (2.0 * lam))
                seg = seg + shift

            # 3. Least-squares de-noise against a low-degree polynomial.
            fit = _poly_denoise(t, seg, degree)
            g_hat[lo:hi] = fit
            prev_val = fit[-1]

    eps = 2.0 * lam * np.round((g_hat - y) / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(y + eps) / (2.0 * lam))
    return y + eps + c
