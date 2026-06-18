# USF Reconstruction Algorithms — Simulation Code

Reference implementations of the Unlimited Sensing Framework (USF)
reconstruction algorithms compared in the report, plus the experiment drivers
that produce every figure.

## Files
| File                  | Contents                                                       |
|-----------------------|----------------------------------------------------------------|
| `usf_core.py`         | Modulo operator, signal generation, noise models, jitter, metrics |
| `hod.py`              | Higher-Order Differences (Bhandari et al. 2021)                |
| `prediction.py`       | Prediction-based recovery (Romanov & Ordentlich 2019)         |
| `b2r2.py`             | Beyond-Bandwidth Residual Recovery (Azar et al. 2025)         |
| `adaptive_local.py`   | Adaptive local representations (Pagginelli Patricio et al.)    |
| `fourier_prony.py`    | Fourier–Prony via matrix pencil (Bhandari et al. 2022)        |
| `ridge_prediction.py` | **Algorithm A** — ridge-regression prediction (this work)     |
| `iter_sis.py`         | **Algorithm B** — annihilation modulo matched filter (this work) |
| `algorithms.py`       | Registry exposing the bandlimited-recovery algorithms uniformly |
| `experiments.py`      | Reproduces the teaser and Experiments 1–4                     |
| `experiment_itersis.py` | Reproduces Experiment 5 (Algorithm B FRI localisation)      |

## Reproducing the figures
```bash
pip install numpy scipy matplotlib
python experiments.py          # teaser + exp1..exp4  -> ../figures/
python experiment_itersis.py   # exp5                  -> ../figures/
```

## Notes on the two proposed algorithms
- **Algorithm A (ridge prediction)** works: exact in the noiseless case for
  OF >= ~4-6, and at high SNR it is the most accurate method of all, rescuing
  prediction-based recovery from its noise fragility. `mu=1e-2` is the default
  (keeps low-OF exactness while giving the noise benefit). It is registered in
  `algorithms.py` and appears in Experiments 1–4 automatically.
- **Algorithm B (annihilation matched filter)** is an honest negative result.
  The annihilation identity is exact on clean data (verified), but recovering
  the fold sequence from folded data is ill-conditioned because the operator is
  weighted by (-1)^n P(n) ~ n^K. It does not beat naive annihilation on the
  folded samples; Experiment 5 quantifies this and the report diagnoses why.

## Conventions
- Signals use a raised-cosine boundary taper (finite-window USF setting) which
  gives the sequential methods a valid in-range seed.
- `nrmse_anchored` removes the unrecoverable global 2*lambda DC ambiguity.
- Spectral/sparse methods are given an oracle fold count for fair comparison.
