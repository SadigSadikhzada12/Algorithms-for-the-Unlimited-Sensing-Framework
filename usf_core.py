"""
usf_core.py
===========
Core building blocks for the Unlimited Sensing Framework (USF).

Everything here follows the notation of Bhandari, Krahmer & Raskar,
"On Unlimited Sampling and Reconstruction", IEEE T-SP 2021, and the
related literature.

A bandlimited signal g(t) is sampled at rate 1/T_s.  Before sampling, a
modulo non-linearity M_lambda folds the signal into [-lambda, lambda).
The forward model is

        y_n = M_lambda( g(n T_s) )                      (noiseless)

The residual decomposition that underlies every recovery algorithm is

        g_n = y_n + eps_n,   eps_n in 2*lambda*Z

so recovering g reduces to recovering the integer-valued residual eps_n.

This file is deliberately small and dependency-light (numpy only); the
recovery algorithms live in separate modules.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Modulo non-linearity
# ---------------------------------------------------------------------------

def modulo(x, lam):
    """Centred modulo M_lambda(x), mapping x into [-lam, lam).

    M_lambda(x) = x - 2*lam*floor((x + lam) / (2*lam))
    """
    return x - 2.0 * lam * np.floor((x + lam) / (2.0 * lam))


# ---------------------------------------------------------------------------
# Bandlimited test signals
# ---------------------------------------------------------------------------

def bandlimited_signal(N, K, beta, OF, seed=0, taper=True):
    """Generate a real bandlimited signal of length N samples.

    The signal is a sum of K random complex sinusoids whose frequencies lie
    inside the band determined by the oversampling factor OF.  With OF = 1 we
    sample exactly at Nyquist; OF > 1 means we oversample.

    A smooth boundary taper (raised-cosine ramp on each end) is applied by
    default.  This brings the signal into the ADC range at the boundaries --
    matching the finite-observation-window setting used throughout the USF
    literature -- which (i) gives time-domain recursions a valid in-range seed
    and (ii) reduces Fourier-domain spectral leakage.

    Parameters
    ----------
    N     : number of samples
    K     : number of sinusoids (controls effective bandwidth)
    beta  : peak amplitude -> dynamic range rho = beta / lambda
    OF    : oversampling factor relative to Nyquist
    seed  : RNG seed
    taper : apply the boundary taper (default True)

    Returns
    -------
    g : (N,) real signal with max|g| = beta
    """
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    f_max = np.pi / OF
    g = np.zeros(N)
    for _ in range(K):
        omega = rng.uniform(0.05 * f_max, f_max)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        g += amp * np.cos(omega * n + phase)
    if taper:
        r = max(1, N // 10)
        win = np.ones(N)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))   # raised cosine
        win[:r] = ramp
        win[-r:] = ramp[::-1]
        g = g * win
    g = g / np.max(np.abs(g)) * beta
    return g


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------
#
# Two distinct noise models appear in the literature:
#
#   (a) POST-modulo noise:  y_n = M_lambda(g_n) + eta_n
#       This is by far the most common assumption (quantisation / thermal
#       noise after the folding ADC).  e.g. Bhandari 2021, Azar 2025.
#
#   (b) PRE-modulo noise:   y_n = M_lambda(g_n + eta_n)
#       Noise enters *before* the non-linearity.  Much harder, because the
#       fold pattern itself is corrupted.  Considered explicitly only by
#       Pagginelli Patricio et al. (CoSeRa 2024 / SSP 2025).
# ---------------------------------------------------------------------------

def add_noise_post_modulo(y, sigma, seed=0):
    """y_n = (clean folded samples) + Gaussian(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    return y + sigma * rng.standard_normal(y.shape)


def fold_with_pre_modulo_noise(g, lam, sigma, seed=0):
    """y_n = M_lambda(g_n + Gaussian(0, sigma^2))."""
    rng = np.random.default_rng(seed)
    return modulo(g + sigma * rng.standard_normal(g.shape), lam)


# ---------------------------------------------------------------------------
# Sampling jitter
# ---------------------------------------------------------------------------
#
# In practice samples are taken at n*T_s + delta_n rather than n*T_s.
# We model bounded jitter delta_n ~ Uniform(-tau, tau) * T_s and realise it
# by resampling the underlying continuous signal.  This is the only
# non-ideality analysed by Yan et al. (arXiv 2025).
# ---------------------------------------------------------------------------

def sample_with_jitter(signal_fn, N, Ts, tau, seed=0):
    """Sample a continuous-time callable signal_fn(t) with bounded jitter.

    delta_n ~ Uniform(-tau, tau) (in units of T_s).  Returns the jittered
    samples and the perturbation vector (in samples).
    """
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    delta = rng.uniform(-tau, tau, size=N)          # in units of samples
    t = (n + delta) * Ts
    return signal_fn(t), delta


# ---------------------------------------------------------------------------
# Reconstruction error metrics
# ---------------------------------------------------------------------------

def nrmse(g_hat, g):
    """Normalised root-mean-square error (dimensionless)."""
    return np.linalg.norm(g_hat - g) / np.linalg.norm(g)


def nrmse_anchored(g_hat, g, lam):
    """NRMSE after removing the unrecoverable global 2*lambda DC ambiguity.

    Residual-recovery algorithms determine g only up to a global additive
    constant in 2*lambda*Z (the modulo operation discards the mean fold level).
    For a fair comparison we remove the best such constant before scoring.
    """
    c = 2.0 * lam * np.round(np.mean(g - g_hat) / (2.0 * lam))
    return np.linalg.norm(g_hat + c - g) / np.linalg.norm(g)


def mse_db(g_hat, g):
    """Mean-square error in dB:  10 log10( ||g_hat-g||^2 / ||g||^2 )."""
    num = np.linalg.norm(g_hat - g) ** 2
    den = np.linalg.norm(g) ** 2
    return 10.0 * np.log10(num / den + 1e-300)


def residual_jumps(g, y, lam):
    """Number of modulo folds (non-zero first differences of the residual).

    This quantifies the 'sparsity of modulo jumps' assumption that most
    algorithms rely on.
    """
    eps = np.round((g - y) / (2.0 * lam)).astype(int)
    return int(np.count_nonzero(np.diff(eps)))
