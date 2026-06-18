"""E6 (best-effort competitor): AdaSSP vs BinAgg vs OLS on D7 (Air Quality).

The paper compares BinAgg against AdaSSP (Wang, 2018) and DP-GD. Neither is in the
BinAgg package, so we add a best-effort AdaSSP implementation (Adaptive Sufficient
Statistics Perturbation) under mu-GDP and compare prediction RelMSE on D7 at mu=1.

CAVEAT (best-effort, not a guaranteed reproduction): exact constants / privacy
calibration / preprocessing in the paper's Appendix are not fully specified, so the
absolute value may differ from the paper's reported AdaSSP RelMSE (0.682). DP-GD is
out of scope (the paper notes it is highly tuning-sensitive and costly).

AdaSSP (Wang 2018), mu-GDP variant:
  split mu into 3 Gaussian mechanisms (mu_i = mu/sqrt(3)) for releasing
  lambda_min(X^T X), X^T X, and X^T y. For an L2-sensitivity Delta, the Gaussian
  mechanism achieving mu_i-GDP uses noise std sigma = Delta / mu_i. Rows are clipped
  to non-private bounds: ||x||_2 <= B (B from data), |y| <= C.

Outputs: results/e6_adassp.json/.csv, figures/e6_adassp.png
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from binagg import dp_linear_regression

from common import DATA, FIGURES, RESULTS, ols, save_json

TARGET = "CO(GT)"
SEEDS = list(range(100))
MU = 1.0
DELTA = 1e-6
PAPER_D7 = {"OLS": 0.441, "BinAgg": 0.463, "AdaSSP": 0.682, "DP-GD": 0.852}


def load_d7():
    d = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    d = d.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    d = d.replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").dropna()
    fc = [c for c in d.columns if c != TARGET]
    return d[fc].values, d[TARGET].values, fc


def relmse(X, y, beta):
    return float(np.sum((X @ beta - y) ** 2) / np.sum(y ** 2))


def adassp(X, y, mu, rng, delta=DELTA):
    """Best-effort AdaSSP (Wang 2018) under mu-GDP. Returns coefficient vector."""
    n, d = X.shape
    # Non-private bounds (paper uses non-private bounds for real-data comparison).
    B = float(np.max(np.linalg.norm(X, axis=1)))   # row L2 bound ||x||_2 <= B
    C = float(np.max(np.abs(y)))                    # |y| <= C
    XtX = X.T @ X
    Xty = X.T @ y
    mu_i = mu / np.sqrt(3.0)                         # 3-way split (composition: mu^2 = sum mu_i^2)

    s_XtX = B ** 2                                   # L2 (Frobenius) sensitivity of X^T X
    s_Xty = B * C                                    # sensitivity of X^T y
    s_lam = B ** 2                                   # sensitivity of lambda_min

    # 1) private lambda_min with a high-probability nonneg correction
    lam_min = float(np.linalg.eigvalsh(XtX)[0])
    z = rng.normal(0.0, s_lam / mu_i)
    thr = (s_lam / mu_i) * np.sqrt(2.0 * np.log(6.0 / delta))
    tilde_lam = max(0.0, lam_min + z - thr)

    # 2) adaptive ridge (scaled to the X^T X noise level)
    ridge = max(0.0, np.sqrt(d * np.log(6.0 / delta)) * (s_XtX / mu_i) - tilde_lam)

    # 3) noisy sufficient statistics
    Z = rng.normal(0.0, s_XtX / mu_i, size=(d, d))
    Z = (Z + Z.T) / np.sqrt(2.0)                     # symmetric Gaussian
    tXtX = XtX + Z
    tXty = Xty + rng.normal(0.0, s_Xty / mu_i, size=d)

    A = tXtX + ridge * np.eye(d)
    try:
        beta = np.linalg.solve(A, tXty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, tXty, rcond=None)[0]
    return beta


def main() -> None:
    X, y, fc = load_d7()
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    yb = (float(y.min()), float(y.max()))
    print(f"D7 Air Quality: n={X.shape[0]}, d={X.shape[1]}")

    ols_rel = relmse(X, y, ols(X, y))

    binagg, adas = [], []
    for s in SEEDS:
        r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=MU, random_state=s)
        if np.all(np.isfinite(r.coefficients)):
            binagg.append(relmse(X, y, r.coefficients))
        b = adassp(X, y, MU, np.random.default_rng(10_000 + s))
        if np.all(np.isfinite(b)):
            adas.append(relmse(X, y, b))
    binagg, adas = np.array(binagg), np.array(adas)

    summary = {
        "OLS": {"relmse": ols_rel, "paper": PAPER_D7["OLS"]},
        "BinAgg": {"relmse_mean": float(binagg.mean()), "relmse_std": float(binagg.std()),
                   "paper": PAPER_D7["BinAgg"]},
        "AdaSSP": {"relmse_mean": float(adas.mean()), "relmse_std": float(adas.std()),
                   "paper": PAPER_D7["AdaSSP"], "note": "best-effort; calibration may differ"},
        "DP-GD": {"relmse_mean": None, "paper": PAPER_D7["DP-GD"], "note": "out of scope"},
    }
    print(f"OLS    RelMSE = {ols_rel:.3f} (paper {PAPER_D7['OLS']})")
    print(f"BinAgg RelMSE = {binagg.mean():.3f}±{binagg.std():.3f} (paper {PAPER_D7['BinAgg']})")
    print(f"AdaSSP RelMSE = {adas.mean():.3f}±{adas.std():.3f} (paper {PAPER_D7['AdaSSP']}, best-effort)")

    save_json("e6_adassp.json", {"mu": MU, "n_reps": len(SEEDS), "n": int(X.shape[0]),
                                 "d": int(X.shape[1]), "summary": summary})
    with open(RESULTS / "e6_adassp.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "relmse_this_study", "relmse_paper_D7"])
        w.writerow(["OLS", round(ols_rel, 4), PAPER_D7["OLS"]])
        w.writerow(["BinAgg", round(float(binagg.mean()), 4), PAPER_D7["BinAgg"]])
        w.writerow(["AdaSSP", round(float(adas.mean()), 4), PAPER_D7["AdaSSP"]])
        w.writerow(["DP-GD", "", PAPER_D7["DP-GD"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["OLS", "BinAgg", "AdaSSP", "DP-GD"]
    ours = [ols_rel, float(binagg.mean()), float(adas.mean()), np.nan]
    paper = [PAPER_D7[m] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - 0.2, ours, width=0.4, color="tab:blue", label="this study (μ=1)")
    ax.bar(x + 0.2, paper, width=0.4, color="tab:red", label="paper D7 (Table 2)")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("RelMSE"); ax.set_title("E6: D7 method comparison (DP-GD: paper only)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e6_adassp.png", dpi=140)
    print("saved figures/e6_adassp.png")


if __name__ == "__main__":
    main()
