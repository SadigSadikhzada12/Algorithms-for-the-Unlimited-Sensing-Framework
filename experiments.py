"""
experiments.py
==============
Reproduces every figure in the Results chapter.

The study compares all reconstruction algorithms on EQUAL FOOTING.  We probe
four axes, each isolating one stress factor:

  EXP 1  Oversampling factor sweep         (how much oversampling each method
                                            needs for stable recovery)
  EXP 2  Modulo-jump sparsity stress       (push the 'sparse folds' assumption
                                            by increasing the dynamic range rho)
  EXP 3  Noise robustness, two models      (post-modulo and pre-modulo Gaussian)
  EXP 4  Sampling jitter                    (bounded timing perturbations)

Each data point is averaged over several random signal realisations (trials).
Run:  python experiments.py
Figures are written to ../figures/ .
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usf_core import (modulo, bandlimited_signal, nrmse_anchored,
                      residual_jumps, add_noise_post_modulo,
                      fold_with_pre_modulo_noise, sample_with_jitter)
from algorithms import ALGORITHMS

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Consistent styling across figures.
plt.rcParams.update({
    "figure.figsize": (7.0, 4.3),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LAM = 1.0


def _style(i):
    return dict(marker=MARKERS[i % len(MARKERS)])


# Floor/ceiling for log-scale plotting: exact-zero errors are shown at the
# floor, divergent (inf/nan) errors at the ceiling, so breakdowns are visible.
ERR_FLOOR = 1e-16
ERR_CEIL = 5e1


def _clamp(vals):
    out = []
    for v in vals:
        if v is None or not np.isfinite(v):
            out.append(ERR_CEIL)
        elif v < ERR_FLOOR:
            out.append(ERR_FLOOR)
        else:
            out.append(min(v, ERR_CEIL))
    return out


def _safe(fn, *args):
    """Run an algorithm, returning np.nan-filled signal on failure."""
    try:
        return fn(*args)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# EXPERIMENT 1 : Oversampling factor sweep
# ---------------------------------------------------------------------------

def exp1_oversampling(N=1024, beta=4.0, OFs=(2, 3, 4, 6, 8, 12, 16, 24, 32),
                      trials=8):
    print("EXP 1: oversampling factor sweep ...")
    results = {name: [] for name in ALGORITHMS}
    for OF in OFs:
        acc = {name: [] for name in ALGORITHMS}
        for tr in range(trials):
            g = bandlimited_signal(N, K=3, beta=beta, OF=OF, seed=100 + tr)
            y = modulo(g, LAM)
            J = residual_jumps(g, y, LAM)
            for name, (fn, _, _) in ALGORITHMS.items():
                gh = _safe(fn, y, LAM, OF, J)
                acc[name].append(np.inf if gh is None
                                 else nrmse_anchored(gh, g, LAM))
        for name in ALGORITHMS:
            results[name].append(np.median(acc[name]))

    plt.figure()
    styles = ["-", "--", "-.", ":", "-"]
    for i, name in enumerate(ALGORITHMS):
        plt.semilogy(OFs, _clamp(results[name]), label=name,
                     marker=MARKERS[i % len(MARKERS)],
                     linestyle=styles[i % len(styles)],
                     markersize=7, alpha=0.85, zorder=5 - i)
    plt.xlabel("Oversampling factor (OF)")
    plt.ylabel("Median NRMSE")
    plt.title(r"Reconstruction error vs oversampling factor ($\rho=%d$)" % beta)
    plt.ylim(1e-17, 1e2)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp1_oversampling.pdf"))
    plt.close()
    return OFs, results


# ---------------------------------------------------------------------------
# EXPERIMENT 2 : Modulo-jump sparsity stress (dynamic range sweep)
# ---------------------------------------------------------------------------

def exp2_sparsity(N=1024, OF=16, betas=(2, 4, 8, 16, 32, 64, 128, 256),
                  trials=8):
    print("EXP 2: modulo-jump sparsity stress ...")
    results = {name: [] for name in ALGORITHMS}
    jump_frac = []
    for beta in betas:
        acc = {name: [] for name in ALGORITHMS}
        jf = []
        for tr in range(trials):
            g = bandlimited_signal(N, K=3, beta=float(beta), OF=OF, seed=200 + tr)
            y = modulo(g, LAM)
            J = residual_jumps(g, y, LAM)
            jf.append(J / N)
            for name, (fn, _, _) in ALGORITHMS.items():
                gh = _safe(fn, y, LAM, OF, J)
                acc[name].append(np.inf if gh is None
                                 else nrmse_anchored(gh, g, LAM))
        jump_frac.append(np.mean(jf))
        for name in ALGORITHMS:
            results[name].append(np.median(acc[name]))

    plt.figure()
    styles = ["-", "--", "-.", ":", "-"]
    for i, name in enumerate(ALGORITHMS):
        plt.semilogy(betas, _clamp(results[name]), label=name,
                     marker=MARKERS[i % len(MARKERS)],
                     linestyle=styles[i % len(styles)],
                     markersize=7, alpha=0.85, zorder=5 - i)
    plt.xscale("log", base=2)
    plt.xlabel(r"Dynamic range  $\rho = \beta / \lambda$")
    plt.ylabel("Median NRMSE")
    plt.title(r"Pushing the sparse-folds assumption (OF=%d)" % OF)
    plt.ylim(1e-17, 1e2)
    plt.legend(fontsize=9, loc="center left")
    # annotate fold density on a secondary axis description
    ax2 = plt.gca().twiny()
    ax2.set_xscale("log", base=2)
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(list(betas))
    ax2.set_xticklabels(["%.2f" % jf for jf in jump_frac], fontsize=7)
    ax2.set_xlabel("fold density (# folds / # samples)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp2_sparsity.pdf"))
    plt.close()
    return betas, jump_frac, results


# ---------------------------------------------------------------------------
# EXPERIMENT 3 : Noise robustness (two models)
# ---------------------------------------------------------------------------

def _snr_to_sigma(g, snr_db):
    p = np.mean(g ** 2)
    return np.sqrt(p / (10 ** (snr_db / 10.0)))


def exp3_noise(N=1024, OF=8, beta=4.0, snrs=(40, 30, 20, 15, 10, 5),
               trials=10):
    print("EXP 3: noise robustness (post- and pre-modulo) ...")
    post = {name: [] for name in ALGORITHMS}
    pre = {name: [] for name in ALGORITHMS}
    for snr in snrs:
        acc_post = {name: [] for name in ALGORITHMS}
        acc_pre = {name: [] for name in ALGORITHMS}
        for tr in range(trials):
            g = bandlimited_signal(N, K=3, beta=beta, OF=OF, seed=300 + tr)
            y_clean = modulo(g, LAM)
            J = residual_jumps(g, y_clean, LAM)
            sigma = _snr_to_sigma(g, snr)

            # (a) post-modulo additive Gaussian noise
            y_post = add_noise_post_modulo(y_clean, sigma, seed=tr)
            # (b) pre-modulo additive Gaussian noise
            y_pre = fold_with_pre_modulo_noise(g, LAM, sigma, seed=tr)

            for name, (fn, _, _) in ALGORITHMS.items():
                gp = _safe(fn, y_post, LAM, OF, J)
                gq = _safe(fn, y_pre, LAM, OF, J)
                acc_post[name].append(np.inf if gp is None
                                      else nrmse_anchored(gp, g, LAM))
                acc_pre[name].append(np.inf if gq is None
                                     else nrmse_anchored(gq, g, LAM))
        for name in ALGORITHMS:
            post[name].append(np.median(acc_post[name]))
            pre[name].append(np.median(acc_pre[name]))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for i, name in enumerate(ALGORITHMS):
        ax[0].semilogy(snrs, _clamp(post[name]), label=name, **_style(i))
        ax[1].semilogy(snrs, _clamp(pre[name]), label=name, **_style(i))
    for a, ttl in zip(ax, ["Post-modulo noise", "Pre-modulo noise"]):
        a.set_xlabel("Input SNR (dB)")
        a.set_title(ttl)
        a.invert_xaxis()
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Median NRMSE")
    ax[1].legend(fontsize=9)
    fig.suptitle("Noise robustness under two noise models (OF=%d, $\\rho$=%d)"
                 % (OF, beta))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "exp3_noise.pdf"))
    plt.close(fig)
    return snrs, post, pre


# ---------------------------------------------------------------------------
# EXPERIMENT 4 : Sampling jitter
# ---------------------------------------------------------------------------

def exp4_jitter(N=1024, OF=8, beta=4.0,
                taus=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), trials=10):
    print("EXP 4: sampling jitter ...")
    results = {name: [] for name in ALGORITHMS}
    for tau in taus:
        acc = {name: [] for name in ALGORITHMS}
        for tr in range(trials):
            # Build a continuous-time band-limited signal as a callable.
            rng = np.random.default_rng(400 + tr)
            f_max = np.pi / OF
            comps = [(rng.uniform(0.05 * f_max, f_max),
                      rng.uniform(0, 2 * np.pi),
                      rng.uniform(0.5, 1.0)) for _ in range(3)]

            def g_cont(t, comps=comps):
                return sum(a * np.cos(w * t + p) for (w, p, a) in comps)

            # normalise amplitude on the uniform grid
            n = np.arange(N)
            g_uniform = g_cont(n)
            scale = beta / np.max(np.abs(g_uniform))

            def g_scaled(t, scale=scale):
                return scale * g_cont(t)

            # jittered samples of the *clean* signal, then fold
            g_jit, delta = sample_with_jitter(g_scaled, N, Ts=1.0, tau=tau,
                                              seed=tr)
            g_ref = g_scaled(n)            # ideal uniform samples (target)
            y = modulo(g_jit, LAM)
            J = residual_jumps(g_jit, y, LAM)

            for name, (fn, _, _) in ALGORITHMS.items():
                gh = _safe(fn, y, LAM, OF, J)
                acc[name].append(np.inf if gh is None
                                 else nrmse_anchored(gh, g_ref, LAM))
        for name in ALGORITHMS:
            results[name].append(np.median(acc[name]))

    plt.figure()
    styles = ["-", "--", "-.", ":", "-"]
    for i, name in enumerate(ALGORITHMS):
        plt.semilogy(taus, _clamp(results[name]), label=name,
                     marker=MARKERS[i % len(MARKERS)],
                     linestyle=styles[i % len(styles)],
                     markersize=7, alpha=0.85, zorder=5 - i)
    plt.xlabel(r"Jitter bound $\tau$ (fraction of $T_s$)")
    plt.ylabel("Median NRMSE")
    plt.title(r"Robustness to sampling jitter (OF=%d, $\rho$=%d)" % (OF, beta))
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp4_jitter.pdf"))
    plt.close()
    return taus, results


# ---------------------------------------------------------------------------
# Illustrative figure: a folded signal and its recovery
# ---------------------------------------------------------------------------

def fig_teaser(N=400, OF=8, beta=6.0):
    print("Teaser figure ...")
    g = bandlimited_signal(N, K=3, beta=beta, OF=OF, seed=1)
    y = modulo(g, LAM)
    from hod import hod_recover
    gh = hod_recover(y, LAM, N=3)
    plt.figure(figsize=(7.5, 3.6))
    plt.plot(g, label="original $g$", color="tab:blue")
    plt.plot(y, label="folded $y=\\mathcal{M}_\\lambda(g)$",
             color="tab:orange", alpha=0.8)
    plt.plot(gh, "--", label="recovered $\\hat g$", color="tab:green")
    plt.axhline(LAM, color="gray", lw=0.8, ls=":")
    plt.axhline(-LAM, color="gray", lw=0.8, ls=":")
    plt.xlabel("sample index $n$")
    plt.ylabel("amplitude")
    plt.title(r"Modulo folding and recovery ($\rho=%d$)" % beta)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "teaser.pdf"))
    plt.close()


if __name__ == "__main__":
    fig_teaser()
    exp1_oversampling()
    exp2_sparsity()
    exp3_noise()
    exp4_jitter()
    print("All figures written to", os.path.abspath(FIGDIR))
