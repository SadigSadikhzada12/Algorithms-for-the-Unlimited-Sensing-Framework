"""
fourier_prony.py
================
Fourier-Prony recovery via matrix pencil.

References
----------
A. Bhandari, F. Krahmer, T. Poskitt,
"Unlimited Sampling From Theory to Practice: Fourier-Prony Recovery and
Prototype ADC", IEEE T-SP, 2022.

R. Guo, A. Bhandari,
"HoD-FP Algorithm for Unlimited Sensing: Where Time Meets Frequency."

Idea
----
The residual eps is piecewise constant, so its first difference
r = Delta(eps) is a stream of spikes located at the fold instants:

        r[n] = sum_j c_j * delta[n - t_j],   c_j in 2*lambda*Z.

Since g is bandlimited it contributes essentially nothing on the out-of-band
(high) DFT bins, so there

        DFT(r)[k]  =  - DFT(Delta y)[k]        (k out of band).

A spike train is a finite-rate-of-innovation signal: a contiguous block of its
DFT obeys a Prony structure.  We estimate the fold locations t_j and amplitudes
c_j from those out-of-band coefficients using the MATRIX PENCIL method (an
ESPRIT-style estimator that is markedly more robust than the classical
annihilating-filter approach, especially when there are many folds).  We then
integrate r to obtain eps and unfold g = y + eps.
"""

import numpy as np


def _matrix_pencil(s, hi, M, K):
    """Recover up to K spikes from a contiguous DFT block s = DFT(r)[hi].

    s[m] = sum_j c_j * z_j^{hi[m]},  z_j = exp(-2*pi*i*t_j / M).
    Uses the matrix-pencil generalised-eigenvalue formulation.

    Returns (integer locations, real amplitudes).
    """
    L = len(s)
    K = min(K, L // 2)
    if K < 1:
        return np.array([], dtype=int), np.array([])

    # Hankel data matrices forming the pencil (Y1, Y0).
    Y0 = np.array([[s[i + j]     for j in range(K)] for i in range(L - K)])
    Y1 = np.array([[s[i + j + 1] for j in range(K)] for i in range(L - K)])

    # Generalised eigenvalues z_j = exp(-2*pi*i*t_j/M).
    z = np.linalg.eigvals(np.linalg.pinv(Y0) @ Y1)
    locs = np.mod(np.round(-np.angle(z) / (2 * np.pi) * M), M).astype(int)

    # Amplitudes by least squares on the Vandermonde system.
    Phi = np.exp(-2j * np.pi * np.outer(hi, locs) / M)
    c, *_ = np.linalg.lstsq(Phi, s, rcond=None)
    return locs, np.real(c)


def _out_of_band(M, OF):
    half_band = int(np.ceil(M / (2 * OF)))
    return np.arange(half_band + 1, M // 2)


def fourier_prony_recover(y, lam, OF, K=None):
    """Fourier-Prony recovery via matrix pencil.

    Parameters
    ----------
    y   : (M,) folded samples
    lam : modulo threshold lambda
    OF  : oversampling factor (sets the out-of-band region)
    K   : expected number of fold instants (sparsity of Delta eps)

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)
    if K is None:
        K = max(1, int(0.2 * M))
    hi = _out_of_band(M, OF)

    dy = np.diff(y, prepend=y[0])
    s = -np.fft.fft(dy)[hi]                      # = DFT(r)[hi]
    locs, amps = _matrix_pencil(s, hi, M, K)

    spikes = np.zeros(M)
    for loc, a in zip(locs, amps):
        spikes[loc] += a
    spikes = 2.0 * lam * np.round(spikes / (2.0 * lam))

    eps = 2.0 * lam * np.round(np.cumsum(spikes) / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(y + eps) / (2.0 * lam))
    return y + eps + c


# ---------------------------------------------------------------------------
# HoD-FP (higher-order-difference Fourier-Prony).
#
# Implementation status: the higher-order-difference hybrid of Guo & Bhandari
# repeatedly differences y to sharpen out-of-band content before the Prony
# step, then recovers the residual through the recursive kappa_l integration
# of their eq. (17) and a final bandlimited projection.  A faithful, exactly
# recovering implementation requires the difference-order rule (their eq. 11)
# and the kappa recursion (eq. 17), which are not reproduced in full in the
# sources available here.  Our attempts to reconstruct it from the published
# pseudocode were numerically unstable (the (h+1)-fold integration amplifies
# spike-localisation error), so HoD-FP is left as future work; plain
# matrix-pencil Fourier-Prony above is the frequency-domain representative in
# the experiments.  See report Section (Algorithms) for discussion.
# ---------------------------------------------------------------------------

def hodfp_recover(y, lam, OF, N=2, K=None):
    """Placeholder: see module note. Falls back to plain Fourier-Prony."""
    raise NotImplementedError(
        "HoD-FP requires the kappa recursion (eq. 17) and difference-order "
        "rule (eq. 11) from Guo & Bhandari; left as future work."
    )
