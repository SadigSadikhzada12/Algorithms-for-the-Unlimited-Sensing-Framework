"""
experiment_itersis.py
=====================
Dedicated evaluation of Algorithm B (ITER-SIS) on its native FRI / matched-filter
parameter-estimation model.

ITER-SIS does not fit the bandlimited-recovery harness used for the other
algorithms (it estimates target delays/amplitudes from modulo matched-filter
samples, not a bandlimited waveform).  We therefore evaluate it on its own
terms: localisation error of K point targets as the dynamic range grows, and we
compare against the natural baseline of running the same annihilation directly
on the folded samples without any unwrapping.

The figure produced is figures/exp5_itersis.pdf .
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usf_core import modulo
from iter_sis import iter_sis_estimate_targets, _tls_annihilate

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"figure.figsize": (7.0, 4.3), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11, "lines.linewidth": 1.8})
LAM = 1.0


def _localize_naive(sig, K, N):
    """Annihilation localisation run directly on a sequence (no unwrapping)."""
    p, _ = _tls_annihilate(sig, K)
    r = np.polynomial.polynomial.polyroots(p)
    r = np.real(r[np.isfinite(r)])
    return np.sort(r[(r > 0) & (r < N)])


def _match_error(t_est, t_true):
    t_est, t_true = np.sort(t_est), np.sort(t_true)
    if len(t_est) < len(t_true):
        t_est = np.pad(t_est, (0, len(t_true) - len(t_est)), constant_values=1e3)
    return np.mean(np.abs(t_est[:len(t_true)] - t_true))


def run(N=120, K=3, T=1.0, amps=(1.5, 2, 3, 4, 6, 8, 10), trials=20):
    print("ITER-SIS FRI localisation experiment ...")
    its_err, naive_err, oracle_err = [], [], []
    for amp in amps:
        e_its, e_naive, e_oracle = [], [], []
        for tr in range(trials):
            rng = np.random.default_rng(tr)
            t_true = np.sort(rng.uniform(15, N - 15, K))
            c_true = amp * rng.uniform(0.7, 1.0, K) * rng.choice([-1, 1], K)
            n = np.arange(N)
            y = sum(ck * np.sinc(n - tk / T) for tk, ck in zip(t_true, c_true))
            z = modulo(y, LAM)
            t_its, _, _, _ = iter_sis_estimate_targets(z, LAM, K, T=T, max_iter=30)
            e_its.append(_match_error(t_its, t_true))
            e_naive.append(_match_error(_localize_naive(z, K, N), t_true))
            e_oracle.append(_match_error(_localize_naive(y, K, N), t_true))
        its_err.append(np.median(e_its))
        naive_err.append(np.median(e_naive))
        oracle_err.append(np.median(e_oracle))

    plt.figure()
    plt.semilogy(amps, oracle_err, "o-", label="Oracle (unfolded)", color="tab:green")
    plt.semilogy(amps, naive_err, "s--", label="Naive annihilation (folded)", color="tab:gray")
    plt.semilogy(amps, its_err, "^-", label="ITER-SIS (Algorithm B)", color="tab:red")
    plt.xlabel(r"Pulse amplitude (dynamic range $\rho$)")
    plt.ylabel("Median localisation error (samples)")
    plt.title(r"FRI target localisation from modulo matched-filter samples (K=%d)" % K)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp5_itersis.pdf"))
    plt.close()
    print("saved exp5_itersis.pdf")
    return amps, its_err, naive_err, oracle_err


if __name__ == "__main__":
    run()
