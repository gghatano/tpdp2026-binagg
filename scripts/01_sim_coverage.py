"""E1: Statistical guarantee on simulated data — empirical 95% CI coverage.

Tests the paper's headline claim that BinAgg DP linear regression yields VALID
confidence intervals (Theorem 4.2). For each privacy budget mu we draw T fresh
datasets, run dp_linear_regression, and measure how often the 95% CI covers the
true coefficient. Valid CIs => coverage ~ 0.95.

Outputs: results/e1_coverage.json, results/e1_coverage.csv, figures/e1_coverage.png
"""
from __future__ import annotations

import csv

import numpy as np
from binagg import dp_linear_regression

from common import (
    FIGURES,
    MU_GRID,
    RESULTS,
    SIM_X_BOUNDS,
    SIM_Y_BOUNDS,
    TRUE_BETA,
    make_sim_data,
    ols,
    save_json,
)

N = 2000
T = 300          # Monte-Carlo trials per mu
ALPHA = 0.05     # 95% CI
D = len(TRUE_BETA)


def covers(ci: np.ndarray, beta_true: np.ndarray) -> np.ndarray:
    return (ci[:, 0] <= beta_true) & (beta_true <= ci[:, 1])


def main() -> None:
    rows = []
    # Reference: non-private OLS coverage (uses textbook OLS CIs).
    for mu in MU_GRID:
        cover_dp = np.zeros(D)
        cover_naive = np.zeros(D)
        bias_dp = np.zeros(D)
        bias_naive = np.zeros(D)
        width_dp = np.zeros(D)
        n_valid = 0
        for t in range(T):
            X, y = make_sim_data(N, seed=1000 + t)
            res = dp_linear_regression(
                X, y, x_bounds=SIM_X_BOUNDS, y_bounds=SIM_Y_BOUNDS,
                mu=mu, alpha=ALPHA, random_state=10_000 + t,
            )
            if not np.all(np.isfinite(res.coefficients)) or not np.all(np.isfinite(res.standard_errors)):
                continue
            n_valid += 1
            cover_dp += covers(res.confidence_intervals, TRUE_BETA)
            # naive CI from naive coefficients + naive SE (z-interval)
            z = 1.959963984540054
            naive_ci = np.column_stack([
                res.naive_coefficients - z * res.naive_standard_errors,
                res.naive_coefficients + z * res.naive_standard_errors,
            ])
            cover_naive += covers(naive_ci, TRUE_BETA)
            bias_dp += res.coefficients - TRUE_BETA
            bias_naive += res.naive_coefficients - TRUE_BETA
            width_dp += res.confidence_intervals[:, 1] - res.confidence_intervals[:, 0]
        cover_dp /= n_valid
        cover_naive /= n_valid
        bias_dp /= n_valid
        bias_naive /= n_valid
        width_dp /= n_valid
        row = {
            "mu": mu,
            "n_valid_trials": n_valid,
            "coverage_dp_mean": float(cover_dp.mean()),
            "coverage_dp_per_coef": cover_dp.round(4).tolist(),
            "coverage_naive_mean": float(cover_naive.mean()),
            "abs_bias_dp_mean": float(np.abs(bias_dp).mean()),
            "abs_bias_naive_mean": float(np.abs(bias_naive).mean()),
            "ci_width_dp_mean": float(width_dp.mean()),
        }
        rows.append(row)
        print(f"mu={mu:>4}: coverage(DP)={row['coverage_dp_mean']:.3f} "
              f"coverage(naive)={row['coverage_naive_mean']:.3f} "
              f"|bias|DP={row['abs_bias_dp_mean']:.3f} "
              f"|bias|naive={row['abs_bias_naive_mean']:.3f} "
              f"CIw={row['ci_width_dp_mean']:.3f} (n={n_valid})")

    meta = {"n": N, "trials": T, "alpha": ALPHA, "true_beta": TRUE_BETA.tolist(),
            "target_coverage": 1 - ALPHA, "rows": rows}
    save_json("e1_coverage.json", meta)
    with open(RESULTS / "e1_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mu", "coverage_dp", "coverage_naive", "abs_bias_dp",
                    "abs_bias_naive", "ci_width_dp", "n_valid"])
        for r in rows:
            w.writerow([r["mu"], r["coverage_dp_mean"], r["coverage_naive_mean"],
                        r["abs_bias_dp_mean"], r["abs_bias_naive_mean"],
                        r["ci_width_dp_mean"], r["n_valid_trials"]])

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = [r["mu"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.axhline(1 - ALPHA, color="k", ls="--", lw=1, label="nominal 0.95")
    ax1.plot(mus, [r["coverage_dp_mean"] for r in rows], "o-", label="BinAgg (bias-corrected)")
    ax1.plot(mus, [r["coverage_naive_mean"] for r in rows], "s--", color="tab:red", label="naive (uncorrected)")
    ax1.set_xlabel("privacy budget μ"); ax1.set_ylabel("empirical 95% CI coverage")
    ax1.set_title("E1: CI coverage vs μ"); ax1.set_ylim(0, 1.02); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(mus, [r["abs_bias_dp_mean"] for r in rows], "o-", label="BinAgg β̃")
    ax2.plot(mus, [r["abs_bias_naive_mean"] for r in rows], "s--", color="tab:red", label="naive")
    ax2.set_xlabel("privacy budget μ"); ax2.set_ylabel("mean |bias| of coefficients")
    ax2.set_title("E1: coefficient bias vs μ"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e1_coverage.png", dpi=140)
    print("saved figures/e1_coverage.png")


if __name__ == "__main__":
    main()
