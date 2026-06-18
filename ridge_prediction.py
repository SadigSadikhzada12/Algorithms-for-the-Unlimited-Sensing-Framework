"""
ridge_prediction.py
===================
Algorithm A (proposed): Ridge-Regression Prediction.

Motivation
----------
The prediction-based recovery of Romanov & Ordentlich (see prediction.py) is
exact in the noiseless case but, as the authors themselves note, is highly
sensitive to noise: the analytic (Chebyshev) predictor is designed to suppress
in-band content at the cost of *amplifying* out-of-band content, so additive
noise is amplified and the sequential recursion then propagates the error.

Idea
----
Instead of fixing the predictor coefficients h = (h_1, ..., h_p) analytically,
we *learn* them from data by (regularised) least squares.  Two observations
make this possible directly from the folded measurements:

  1. Before the first fold, the samples are unfolded: y_n = g_n.  These clean
     samples provide supervised (input, target) pairs for fitting a predictor.
  2. The modulo nesting property M_{2lam}( M_lam(x) ) = M_lam(x) (more generally
     M_{lam2} o M_{lam1} = M_{lam2} when lam2 in lam1 Z) means earlier samples
     can be folded *further* to synthesise additional training pairs at no cost.

The predictor is then fit by RIDGE regression,

    min_h  sum_n ( g_n - sum_k h_k g_{n-k} )^2  +  mu * ||h||_2^2 ,

whose closed-form solution is  h = (X^T X + mu I)^{-1} X^T t .  The ridge
penalty mu > 0 explicitly discourages the large coefficients that amplify
noise, trading the exact recovery of the analytic predictor for robustness.
Once h is learned, the sequential unwrapping is identical to the analytic
prediction method:

    g_n = y_n + 2 lam * round( (h . past - y_n) / (2 lam) ).

This is the regression-based prediction proposed in the interim report.
"""

import numpy as np


def _build_training(seed_samples, p):
    """Construct (X, t) regression pairs from a 1-D clean (unfolded) segment.

    Row i:  target t_i = seed[i],  features X_i = [seed[i-1], ..., seed[i-p]].
    """
    s = np.asarray(seed_samples, dtype=float)
    rows_X, rows_t = [], []
    for i in range(p, len(s)):
        rows_X.append(s[i - p:i][::-1])     # g_{i-1}, ..., g_{i-p}
        rows_t.append(s[i])
    if not rows_X:
        return np.zeros((0, p)), np.zeros(0)
    return np.array(rows_X), np.array(rows_t)


def ridge_fit(X, t, mu):
    """Closed-form ridge regression:  h = (X^T X + mu I)^{-1} X^T t."""
    p = X.shape[1]
    A = X.T @ X + mu * np.eye(p)
    return np.linalg.solve(A, X.T @ t)


def ridge_prediction_recover(y, lam, OF, p=12, mu=1e-2, n_seed=None,
                             augment=True):
    """Recover g from folded samples y via a ridge-learned linear predictor.

    Parameters
    ----------
    y       : (M,) folded samples
    lam     : modulo threshold lambda
    OF      : oversampling factor (used only to size the seed window)
    p       : predictor order
    mu      : ridge regularisation strength
    n_seed  : number of leading samples assumed unfolded (training data).
              If None, a window proportional to the predictor order is used.
    augment : if True, synthesise extra training pairs by folding the seed
              further (the modulo nesting property), improving conditioning.

    Returns
    -------
    g_hat : (M,) estimated unfolded samples
    """
    M = len(y)
    if n_seed is None:
        n_seed = min(M, max(4 * p, 32))

    # The leading n_seed samples are assumed in-range (g = y there); to be
    # robust when that prefix already contains a fold, unwrap it first with a
    # short higher-order-difference pass (exact for a clean prefix).
    from hod import hod_recover
    seed = hod_recover(y[:n_seed], lam, N=3)

    # Training pairs from the (unfolded) seed, optionally augmented by folding.
    X, t = _build_training(seed, p)
    if augment:
        for extra_lam in (lam, 2 * lam):
            from usf_core import modulo
            Xa, ta = _build_training(modulo(seed, extra_lam), p)
            if len(Xa):
                X = np.vstack([X, Xa])
                t = np.concatenate([t, ta])

    if len(X) < p:                       # not enough data: fall back to flat predictor
        h = np.zeros(p)
        h[0] = 1.0
    else:
        h = ridge_fit(X, t, mu)

    # Sequential unwrapping with the learned predictor.
    g = np.array(y, dtype=float)
    g[:n_seed] = seed
    for n in range(n_seed, M):
        g_pred = np.dot(h, g[n - p:n][::-1])
        g[n] = y[n] + 2.0 * lam * np.round((g_pred - y[n]) / (2.0 * lam))

    # Anchor the global 2*lambda DC ambiguity.
    eps = 2.0 * lam * np.round((g - y) / (2.0 * lam))
    c = 2.0 * lam * np.round(-np.mean(y + eps) / (2.0 * lam))
    return y + eps + c
