"""
algorithms.py
=============
A registry that exposes every recovery algorithm through one common interface:

    recover(y, lam, OF, oracle_jumps) -> g_hat

This lets the experiment driver loop over algorithms uniformly.  `oracle_jumps`
is the true number of modulo folds (used by the sparse/Prony methods to set
their model order); in a fully blind setting it would be replaced by an
estimate, but for a controlled comparison we give every algorithm the same
information.

The two student-proposed algorithms are registered as TODO stubs.
"""

from hod import hod_recover
from prediction import prediction_recover
from b2r2 import b2r2_recover
from adaptive_local import adaptive_local_recover
from fourier_prony import fourier_prony_recover
from ridge_prediction import ridge_prediction_recover


def _hod(y, lam, OF, J):
    return hod_recover(y, lam, N=3)


def _prediction(y, lam, OF, J):
    return prediction_recover(y, lam, p=12, OF=OF)


def _b2r2(y, lam, OF, J):
    return b2r2_recover(y, lam, OF, K_sparse=J + 5)


def _adaptive(y, lam, OF, J):
    return adaptive_local_recover(y, lam)


def _fp(y, lam, OF, J):
    return fourier_prony_recover(y, lam, OF, K=max(J, 1))


def _ridge(y, lam, OF, J):
    return ridge_prediction_recover(y, lam, OF, p=12, mu=1e-2)


# Ordered registry: name -> (function, time/frequency domain, noise-robust?)
ALGORITHMS = {
    "HoD":           (_hod,        "time",      False),
    "Prediction":    (_prediction, "time",      False),
    "B2R2":          (_b2r2,       "frequency", True),
    "Adaptive-Local": (_adaptive,  "time",      True),
    "Fourier-Prony": (_fp,         "frequency", True),
    "Ridge-Pred (A)": (_ridge,     "time",      True),
    # Algorithm B (ITER-SIS) targets an FRI parameter-estimation model, not the
    # bandlimited recovery model used in this harness, so it is evaluated
    # separately (see iter_sis.py and the dedicated FRI experiment).
}

