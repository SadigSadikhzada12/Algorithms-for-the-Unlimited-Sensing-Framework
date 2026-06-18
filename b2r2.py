"""
b2r2.py
=======
Beyond-Bandwidth Residual Recovery (B2R2).

Reference
---------
E. Azar, S. Mulleti, Y. C. Eldar,
"Unlimited sampling beyond modulo",
Applied and Computational Harmonic Analysis, 2025.
(see also Azar et al., "Residual Recovery Algorithm for Modulo Sampling",
ICASSP 2022, and the LASSO variant of Shah et al., ICASSP 2023.)

Idea
----
Write g = y + eps with eps in 2*lambda*Z.  The signal g is bandlimited, so its
DFT is (essentially) supported on the low-frequency band B.  The residual eps,
being piecewise constant on the lattice, has energy spread across ALL
frequencies, including the high-frequency band beyond the signal bandwidth.

B2R2 exploits this:  on the OUT-OF-BAND frequencies, the folded samples'
spectrum equals minus the residual spectrum (since g contributes nothing
there).  The first-order difference  r = Delta(eps)  is SPARSE -- it is
non-zero only at the (few) fold instants.  We therefore solve a sparse
recovery problem

        find sparse r  s.t.  F_H r  =  - F_H (Delta y)

where F_H selects the high-frequency (beyond-bandwidth) DFT rows.  Once r is
found, eps = cumulative-sum(r) (snapped to the lattice) and g = y + eps.

The sparse system is solved by Orthogonal Matching Pursuit (OMP), which keeps
the implementation dependency-light and matches the spirit of the residual
recovery papers.  A LASSO solver gives the closely related LASSO-B2R2 variant.
"""

import numpy as np


def _dft_matrix(M):
    n = np.arange(M)
    F = np.exp(-2j * np.pi * np.outer(n, n) / M) / np.sqrt(M)
    return F


def _omp_fft(out_band, b, M, n_nonzero, tol=1e-8):
    """OMP where the dictionary is the IDFT restricted to out-of-band rows.

    The measurement operator is A = F[out_band, :] (selected DFT rows).  Its
    adjoint applied to a residual is an inverse-DFT followed by zeroing of the
    in-band frequencies -- both computed with the FFT, so no dense M x M matrix
    is ever formed.  This is the standard fast B2R2 implementation.
    """
    idx = []
    x = np.zeros(M, dtype=complex)
    residual = b.copy()

    # Precompute the (small) selected-row sub-DFT lazily per chosen atom.
    chosen_cols = []

    def adjoint(r):
        # embed r back onto the out-of-band frequencies, then inverse DFT
        full = np.zeros(M, dtype=complex)
        full[out_band] = r
        return np.fft.ifft(full) * M / np.sqrt(M)

    # Build measurement columns on demand: column j of A is F[out_band, j].
    n = np.arange(M)
    ob = np.where(out_band)[0]

    def column(j):
        return np.exp(-2j * np.pi * ob * j / M) / np.sqrt(M)

    for _ in range(n_nonzero):
        corr = np.abs(adjoint(residual))
        j = int(np.argmax(corr))
        if j not in idx:
            idx.append(j)
            chosen_cols.append(column(j))
        As = np.column_stack(chosen_cols)
        xs, *_ = np.linalg.lstsq(As, b, rcond=None)
        residual = b - As @ xs
        x[:] = 0
        x[idx] = xs
        if np.linalg.norm(residual) < tol:
            break
    return x


def b2r2_recover(y, lam, OF, K_sparse=None, bandwidth_frac=None):
    """Recover g from folded samples y via beyond-bandwidth residual recovery.

    Parameters
    ----------
    y              : (M,) folded samples
    lam            : modulo threshold lambda
    OF             : oversampling factor (sets the signal bandwidth fraction)
    K_sparse       : expected number of fold transitions (sparsity of Delta eps).
                     If None, set to a generous fraction of M.
    bandwidth_frac : fraction of DFT bins occupied by the signal.  If None,
                     derived from OF as 1/OF.

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)
    if bandwidth_frac is None:
        bandwidth_frac = 1.0 / OF
    if K_sparse is None:
        K_sparse = max(1, int(0.5 * M))

    # First-order difference of the folded samples.
    dy = np.diff(y, prepend=y[0])

    # Out-of-band (high-frequency) DFT rows: there the signal vanishes, so the
    # measurements equal the spectrum of Delta(eps) = r (the sparse spike train).
    half_band = int(np.ceil(bandwidth_frac * M / 2))
    freqs = np.arange(M)
    in_band = (freqs <= half_band) | (freqs >= M - half_band)
    out_band = ~in_band
    if not np.any(out_band):
        out_band = freqs != 0

    Dy = np.fft.fft(dy) / np.sqrt(M)
    meas = -Dy[out_band]                       # = F_H @ r  on out-of-band

    # Sparse recovery of r = Delta(eps) via FFT-based OMP.  The number of
    # measurements (out-of-band bins) caps how many spikes are identifiable;
    # beyond that the problem is ill-posed.  We also impose an absolute cap on
    # the iteration count: if more folds than this are present the sparse-folds
    # premise has already failed, so there is no value in running OMP further.
    max_k = max(1, len(meas) // 2)
    K_sparse = min(K_sparse, max_k, 150)
    r = _omp_fft(out_band, meas, M, n_nonzero=K_sparse)
    r = np.real(r)
    r = 2.0 * lam * np.round(r / (2.0 * lam))      # snap to lattice steps

    # Integrate to obtain the residual, then unfold.
    eps = np.cumsum(r)
    eps = 2.0 * lam * np.round(eps / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(y + eps) / (2.0 * lam))
    return y + eps + c
