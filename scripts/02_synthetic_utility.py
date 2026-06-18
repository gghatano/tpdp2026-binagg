"""E2: Synthetic-data regression utility & the value of bias correction.

Compares, across mu, the linear-regression coefficients obtained from:
  - true beta (ground truth)
  - OLS on the original (private) data         [non-private reference]
  - naive OLS fit on BinAgg synthetic data     [shows the bias the paper warns about]
  - BinAgg DP regression (bias-corrected beta-tilde)

Utility is measured as relative L2 coefficient error (a regression guarantee),
not marginal preservation. Averaged over S seeds (mean +/- std).

Outputs: results/e2_synthetic_utility.json/.csv, figures/e2_utility.png
"""
from __future__ import annotations

import csv

import numpy as np
from binagg import dp_linear_regression, generate_synthetic_data

from common import (
    FIGURES,
    MU_GRID,
    RESULTS,
    SIM_X_BOUNDS,
    SIM_Y_BOUNDS,
    TRUE_BETA,
    make_sim_data,
    ols,
    rel_l2,
    save_json,
)

N = 2000
SEEDS = list(range(20))


def main() -> None:
    # OLS-on-original reference (does not depend on mu); average over seeds.
    ols_err = []
    for s in SEEDS:
        X, y = make_sim_data(N, seed=s)
        ols_err.append(rel_l2(ols(X, y), TRUE_BETA))
    ols_err = np.array(ols_err)

    rows = []
    for mu in MU_GRID:
        naive_syn_err, dp_err = [], []
        for s in SEEDS:
            X, y = make_sim_data(N, seed=s)
            syn = generate_synthetic_data(
                X, y, x_bounds=SIM_X_BOUNDS, y_bounds=SIM_Y_BOUNDS,
                mu=mu, random_state=2000 + s,
            )
            if syn.n_samples >= len(TRUE_BETA) + 1:
                beta_syn = ols(syn.X_synthetic, syn.y_synthetic)
                naive_syn_err.append(rel_l2(beta_syn, TRUE_BETA))
            res = dp_linear_regression(
                X, y, x_bounds=SIM_X_BOUNDS, y_bounds=SIM_Y_BOUNDS,
                mu=mu, random_state=2000 + s,
            )
            if np.all(np.isfinite(res.coefficients)):
                dp_err.append(rel_l2(res.coefficients, TRUE_BETA))
        naive_syn_err = np.array(naive_syn_err)
        dp_err = np.array(dp_err)
        row = {
            "mu": mu,
            "ols_orig_err_mean": float(ols_err.mean()),
            "naive_synthetic_err_mean": float(naive_syn_err.mean()),
            "naive_synthetic_err_std": float(naive_syn_err.std()),
            "binagg_dp_err_mean": float(dp_err.mean()),
            "binagg_dp_err_std": float(dp_err.std()),
        }
        rows.append(row)
        print(f"mu={mu:>4}: relL2  OLS(orig)={row['ols_orig_err_mean']:.3f}  "
              f"naiveOLS(syn)={row['naive_synthetic_err_mean']:.3f}  "
              f"BinAgg DP={row['binagg_dp_err_mean']:.3f}")

    save_json("e2_synthetic_utility.json",
              {"n": N, "seeds": len(SEEDS), "true_beta": TRUE_BETA.tolist(), "rows": rows})
    with open(RESULTS / "e2_synthetic_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mu", "ols_orig_err", "naive_synthetic_err", "binagg_dp_err"])
        for r in rows:
            w.writerow([r["mu"], r["ols_orig_err_mean"],
                        r["naive_synthetic_err_mean"], r["binagg_dp_err_mean"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = [r["mu"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(rows[0]["ols_orig_err_mean"], color="k", ls=":", label="OLS on original (ref)")
    ax.errorbar(mus, [r["naive_synthetic_err_mean"] for r in rows],
                yerr=[r["naive_synthetic_err_std"] for r in rows],
                fmt="s--", color="tab:red", capsize=3, label="naive OLS on synthetic")
    ax.errorbar(mus, [r["binagg_dp_err_mean"] for r in rows],
                yerr=[r["binagg_dp_err_std"] for r in rows],
                fmt="o-", color="tab:blue", capsize=3, label="BinAgg DP regression (β̃)")
    ax.set_xlabel("privacy budget μ"); ax.set_ylabel("relative L2 coefficient error")
    ax.set_title("E2: regression utility vs μ (lower is better)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e2_utility.png", dpi=140)
    print("saved figures/e2_utility.png")


if __name__ == "__main__":
    main()
