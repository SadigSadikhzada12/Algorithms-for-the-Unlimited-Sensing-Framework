"""
prediction.py
=============
Prediction-based recovery ("Above the Nyquist Rate, Modulo Folding Does Not
Hurt").

Reference
---------
E. Romanov, O. Ordentlich,
"Above the Nyquist Rate, Modulo Folding Does Not Hurt",
IEEE Signal Processing Letters, 2019.

Idea
----
For a bandlimited signal sampled above Nyquist, the next sample g_n can be
predicted from its past with arbitrarily small error using a finite linear
predictor h = (h_1, ..., h_p):

        g_hat_n = sum_{k=1}^{p} h_k g_{n-k},   |g_n - g_hat_n| < lambda.

Romanov & Ordentlich use predictors derived from Chebyshev polynomials so the
prediction error stays below lambda for any sampling rate above Nyquist.
Given the recovered past samples g_{n-1}, ..., g_{n-p}, the current unfolded
sample is the unique lattice representative of y_n closest to the prediction:

        g_n = y_n + 2*lambda * round( (g_hat_n - y_n) / (2*lambda) ).

The recursion is seeded with the first p samples (assumed un-folded, i.e.
the signal starts inside the dynamic range, as in the paper).
"""

import numpy as np
from scipy.linalg import solve_toeplitz


def bandlimited_predictor(p, OF):
    """Order-p optimal linear predictor for a signal bandlimited to pi/OF.

    Romanov & Ordentlich show that above Nyquist the one-step prediction error
    of a bandlimited signal can be made smaller than lambda by increasing the
    predictor order.  The minimum-mean-square predictor for a process with an
    ideal low-pass power spectrum (cutoff omega_c = pi/OF) has autocorrelation

            R[k] = sinc(k / OF)

    and the predictor coefficients h solve the Yule-Walker / normal equations
    R[1:p+1] = Toeplitz(R[0:p]) h.  This gives a prediction error that shrinks
    rapidly with p, exactly the mechanism behind the recovery guarantee.
    """
    k = np.arange(p + 1)
    R = np.sinc(k / OF)                      # numpy sinc = sin(pi x)/(pi x)
    h = solve_toeplitz((R[:p], R[:p]), R[1:p + 1])
    return h


def prediction_recover(y, lam, p=10, OF=8):
    """Recover g from folded samples y by sequential linear prediction.

    Parameters
    ----------
    y   : (M,) folded samples
    lam : modulo threshold lambda
    p   : predictor order
    OF  : oversampling factor (sets the predictor's cutoff)

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)
    h = bandlimited_predictor(p, OF)
    g = np.array(y, dtype=float)

    # Seed: the first p samples may themselves be folded.  Bootstrap them with
    # a short higher-order-difference unfold (exact in the noiseless case), so
    # the predictor has a correct history to start from.
    from hod import hod_recover
    n_seed = min(M, max(2 * p, 16))
    g[:n_seed] = hod_recover(y[:n_seed], lam, N=3)

    for n in range(n_seed, M):
        past = g[n - p:n][::-1]                         # g_{n-1}, ..., g_{n-p}
        g_pred = np.dot(h, past)
        g[n] = y[n] + 2.0 * lam * np.round((g_pred - y[n]) / (2.0 * lam))

    # The recursion recovers g up to a global multiple of 2*lambda; anchor it.
    eps = 2.0 * lam * np.round((g - y) / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(y + eps) / (2.0 * lam))
    return y + eps + c
