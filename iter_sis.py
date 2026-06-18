"""
iter_sis.py
===========
Algorithm B (proposed): Iterative Signal Sieving for the Modulo Matched Filter
(ITER-SIS).

Setting
-------
This algorithm targets a different signal model from the bandlimited recovery
methods: a *parameter estimation* (localisation) problem.  A probing pulse
phi(t) = sinc(t/T) is reflected by K point targets with delays {t_k} and
amplitudes {c_k}.  After matched filtering and sampling at spacing T, the
(unwrapped) matched-filter output is

    y_n = sum_{k=0}^{K-1} c_k * sinc(n - t_k / T),   n = 0, ..., N-1.

A high-dynamic-range receiver folds this with a modulo ADC, so the actual
measurements are z_n = mod_lambda(y_n) = y_n - lambda * e_n, with e_n the
unknown integer fold sequence.

Annihilation mechanism (ideal, unfolded case)
---------------------------------------------
Define the alternating polynomial P(u) = prod_k (u - u_k) with u_k = t_k / T,
P(u) = sum_{m=0}^{K} p_m u^m, p_K = 1.  By the interpolation properties of the
sinc kernel, weighting the sequence by (-1)^n P(n) maps it into the null space
of the K-th order forward difference Delta^K:

    Delta^K( (-1)^n P(n) y_n ) = 0.

Expanding P gives a linear system V(y) p = 0, where

    [V(y)]_{n,m} = Delta^K( (-1)^n n^m y_n ).

The polynomial coefficients p are the right singular vector of V(y) of minimum
singular value (a total-least-squares annihilation); the target locations are
the roots of P.

The modulo bottleneck and iterative sieving
-------------------------------------------
With folded data, V(z) p != 0 because the abrupt 2*lambda jumps break the
annihilation.  ITER-SIS alternates two "sieves":
  * Sieve 1 (algebraic): given the current unwrapped estimate y_hat, re-solve
    the TLS annihilation for p.
  * Sieve 2 (structural): given p, any residual leakage of the annihilation
    relation must map onto the integer * lambda lattice; use it to update the
    piecewise-constant fold sequence e_hat.
The iteration is seeded by estimating the sparse fold jumps from the first
differences of z.

This file implements the pipeline of the self-contained specification.
"""

import numpy as np
from numpy.polynomial import polynomial as P


def _alternating_diff_matrix(sig, K):
    """Build V where [V]_{n,m} = Delta^K( (-1)^n n^m sig_n ), m=0..K."""
    N = len(sig)
    n = np.arange(N)
    sign = (-1.0) ** n
    cols = []
    for m in range(K + 1):
        seq = sign * (n ** m) * sig
        d = seq.copy()
        for _ in range(K):                  # K-th forward difference
            d = np.diff(d)
        cols.append(d)
    return np.column_stack(cols)            # (N-K) x (K+1)


def _tls_annihilate(y_hat, K):
    """Total-least-squares: p = right singular vector of V(y_hat) of min sigma."""
    V = _alternating_diff_matrix(y_hat, K)
    _, _, Wt = np.linalg.svd(V)
    p = Wt[-1]                              # last row of W^T (min singular vec)
    return p, V


def iter_sis_recover_signal(z, lam, K, eps_tol=1e-6, max_iter=30):
    """Run ITER-SIS and return the recovered UNWRAPPED signal y_hat.

    Parameters
    ----------
    z        : (N,) folded matched-filter samples
    lam      : modulo threshold lambda
    K        : number of targets
    eps_tol  : convergence tolerance on the annihilation residual
    max_iter : iteration cap

    Returns
    -------
    y_hat : (N,) recovered unwrapped signal
    info  : dict with the final polynomial coefficients and residual
    """
def _DK_matrix(N, K):
    """Matrix form of the K-th forward difference operator: (N-K) x N."""
    M = np.eye(N)
    for _ in range(K):
        M = np.diff(M, axis=0)
    return M


def _omp_sparse(A, b, k, tol=1e-9):
    """Orthogonal matching pursuit for a k-sparse solution of A x = b."""
    res = b.copy()
    idx = []
    x = np.zeros(A.shape[1])
    for _ in range(k):
        j = int(np.argmax(np.abs(A.T @ res)))
        if j not in idx:
            idx.append(j)
        xs, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
        res = b - A[:, idx] @ xs
        x[:] = 0
        x[idx] = xs
        if np.linalg.norm(res) < tol:
            break
    return x


def iter_sis_recover_signal(z, lam, K, eps_tol=1e-6, max_iter=30, max_folds=None):
    """Run ITER-SIS and return the recovered UNWRAPPED signal y_hat.

    Sieve 1 re-solves the TLS annihilation for the polynomial p; Sieve 2, given
    p, recovers a SPARSE fold-jump sequence Delta(e) by OMP so that the
    annihilation leakage V(z + 2*lam*e) p is minimised.  Each Sieve-2 update is
    accepted only if it reduces the leakage (a monotonic safeguard), which keeps
    the iteration stable.
    """
    N = len(z)
    step = 2.0 * lam
    if max_folds is None:
        max_folds = max(4, N // 4)

    DK = _DK_matrix(N, K)
    Cumsum = np.tril(np.ones((N, N)))      # e = Cumsum @ de

    # --- Initialise the fold sequence from sparse first-difference jumps ---
    de0 = -np.round(np.diff(z, prepend=z[0]) / step)
    e = np.cumsum(de0)
    e = e - e[0]
    y_hat = z + step * e

    def leakage(sig, p):
        return np.linalg.norm(DK @ (((-1.0) ** np.arange(N)) *
                              np.polynomial.polynomial.polyval(np.arange(N), p) * sig))

    p, _ = _tls_annihilate(y_hat, K)
    best_res = leakage(y_hat, p) ** 2

    for i in range(max_iter):
        # --- Sieve 1: TLS annihilation for polynomial coefficients ---
        p, _ = _tls_annihilate(y_hat, K)

        # --- Sieve 2: sparse fold-jump recovery given p ---
        n = np.arange(N)
        w = ((-1.0) ** n) * np.polynomial.polynomial.polyval(n, p)
        B = step * (DK @ np.diag(w))       # leakage = B @ e + DK@(w*z)
        A = B @ Cumsum                     # in terms of de
        rhs = -(DK @ (w * z))
        de = _omp_sparse(A, rhs, max_folds)
        de = np.round(de)
        e_new = np.cumsum(de)
        y_new = z + step * e_new

        res_new = leakage(y_new, p) ** 2
        if res_new < best_res:             # accept only if it helps
            best_res, e, y_hat = res_new, e_new, y_new
        if best_res < eps_tol:
            break

    return y_hat, {"p": p, "residual": best_res, "iters": i + 1}


def iter_sis_estimate_targets(z, lam, K, T=1.0, **kw):
    """Full ITER-SIS: recover unwrapped signal, then extract target params.

    Returns target delays t_k and amplitudes c_k.
    """
    N = len(z)
    y_hat, info = iter_sis_recover_signal(z, lam, K, **kw)
    p = info["p"]

    # --- Step C: parameter extraction ---
    # roots of P(u) = sum_m p_m u^m  (coeffs given low-order-first for polyroots)
    roots = np.polynomial.polynomial.polyroots(p)
    roots = roots[np.isfinite(roots)]
    u = np.real(roots)
    # keep the K roots lying within the observation window
    u = u[(u > -1) & (u < N + 1)]
    if len(u) > K:
        u = u[:K]
    t = T * u

    # amplitudes via least squares on the routing matrix A_{n,k} = sinc(n - u_k)
    n = np.arange(N)
    A = np.sinc(n[:, None] - u[None, :])
    c, *_ = np.linalg.lstsq(A, y_hat, rcond=None)
    return t, c, y_hat, info


# ---------------------------------------------------------------------------
# Adapter so ITER-SIS can be scored in the common bandlimited-recovery harness.
# It simply returns the recovered unwrapped signal y_hat (signal-level output).
# ---------------------------------------------------------------------------

def iter_sis_recover(z, lam, OF, K):
    """Common-interface wrapper: return the unwrapped signal estimate.

    Here K is the number of sinc components / targets used as the model order.
    """
    y_hat, _ = iter_sis_recover_signal(z, lam, max(1, K))
    # anchor global lattice constant for fair scoring
    e = 2.0 * lam * np.round((y_hat - z) / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(z + e) / (2.0 * lam))
    return z + e + c
