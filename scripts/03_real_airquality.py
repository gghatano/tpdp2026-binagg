"""E3: Real-data reproduction on UCI Air Quality.

Reproduces the three usage modes from BinAgg's real_data_example.py:
  Option A: DP linear regression only
  Option B: DP regression + synthetic data sharing the SAME budget
  Option C: synthetic data only
across privacy budgets mu, comparing DP coefficients / CI widths against
non-private OLS, plus synthetic-vs-original moment errors.

Bounds come from DOMAIN KNOWLEDGE (copied from the BinAgg example), never from
the data. Outputs: results/e3_airquality.json/.csv, figures/e3_airquality.png
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from binagg import dp_linear_regression, generate_synthetic_data

from common import DATA, FIGURES, MU_GRID, RESULTS, ols, save_json

# Domain-knowledge bounds (from BinAgg examples/real_data_example.py)
X_BOUNDS = [
    (0, 3000), (0, 1500), (0, 100), (0, 3000), (0, 2000), (0, 3000),
    (0, 500), (0, 3000), (0, 3000), (-20, 50), (0, 100), (0, 3),
]
Y_BOUNDS = (0, 15)
TARGET = "CO(GT)"
SEEDS = [42, 43, 44, 45, 46]


def load_airquality():
    data = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    data = data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    data = data.replace(",", ".", regex=True)
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    feature_cols = [c for c in data.columns if c != TARGET]
    X = data[feature_cols].values
    y = data[TARGET].values
    return X, y, feature_cols


def main() -> None:
    X, y, feature_cols = load_airquality()
    n, d = X.shape
    print(f"AirQuality: n={n}, d={d}, target={TARGET}")

    # REPRODUCIBILITY NOTE: this UCI dataset codes missing values as -200, and the
    # BinAgg example pipeline does NOT remove them. The DP mechanism clips to the
    # domain bounds (effectively mapping the -200 sentinels into range), so synthetic
    # moments must be compared against the CLIPPED original, not the raw column, to be
    # meaningful. We record the contamination and compare against clipped moments.
    n_sentinel_y = int((y == -200).sum())
    y_clip = np.clip(y, Y_BOUNDS[0], Y_BOUNDS[1])
    clip_mean_y, clip_std_y = float(y_clip.mean()), float(y_clip.std())

    beta_ols = ols(X, y)

    rows = []
    coef_table = {"feature": feature_cols, "ols": beta_ols.round(5).tolist()}
    for mu in MU_GRID:
        ci_widths, dp_coef_seeds = [], []
        meany_err, stdy_err = [], []
        for s in SEEDS:
            # Option B: regression + synthetic, shared budget
            reg, syn = dp_linear_regression(
                X, y, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, mu=mu,
                return_synthetic=True, clip_synthetic_output=True, random_state=s,
            )
            if np.all(np.isfinite(reg.coefficients)):
                dp_coef_seeds.append(reg.coefficients)
                w = reg.confidence_intervals[:, 1] - reg.confidence_intervals[:, 0]
                ci_widths.append(np.nanmean(w))
            if syn.n_samples > 0:
                meany_err.append(abs(syn.y_synthetic.mean() - clip_mean_y))
                stdy_err.append(abs(syn.y_synthetic.std() - clip_std_y))
        dp_coef_seeds = np.array(dp_coef_seeds)
        row = {
            "mu": mu,
            "n_seeds_ok": int(len(dp_coef_seeds)),
            "ci_width_mean": float(np.mean(ci_widths)) if ci_widths else float("nan"),
            "synthetic_mean_y_err": float(np.mean(meany_err)) if meany_err else float("nan"),
            "synthetic_std_y_err": float(np.mean(stdy_err)) if stdy_err else float("nan"),
            "dp_coef_mean": dp_coef_seeds.mean(axis=0).round(5).tolist() if len(dp_coef_seeds) else None,
        }
        rows.append(row)
        coef_table[f"dp_mu{mu}"] = row["dp_coef_mean"]
        print(f"mu={mu:>4}: CIwidth={row['ci_width_mean']:.4f}  "
              f"mean(y)err={row['synthetic_mean_y_err']:.4f}  "
              f"std(y)err={row['synthetic_std_y_err']:.4f}  (seeds ok={row['n_seeds_ok']})")

    # Option A vs C equivalence-in-distribution sanity (single seed, mu=1)
    res_a = dp_linear_regression(X, y, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, mu=1.0, random_state=42)
    syn_c = generate_synthetic_data(X, y, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, mu=1.0,
                                    clip_output=True, random_state=42)
    options_note = {
        "optionA_coef_mu1_seed42": res_a.coefficients.round(5).tolist(),
        "optionC_n_synthetic_mu1_seed42": int(syn_c.n_samples),
        "optionC_n_bins_used": int(syn_c.n_bins_used),
    }

    save_json("e3_airquality.json", {
        "n": int(n), "d": int(d), "target": TARGET, "features": feature_cols,
        "ols_coef": beta_ols.round(6).tolist(), "rows": rows, "options_note": options_note,
        "raw_mean_y": float(y.mean()), "raw_std_y": float(y.std()),
        "clipped_mean_y": clip_mean_y, "clipped_std_y": clip_std_y,
        "n_missing_sentinel_y": n_sentinel_y,
        "moment_reference": "clipped original (domain bounds); see reproducibility note",
    })
    with open(RESULTS / "e3_airquality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mu", "ci_width_mean", "synthetic_mean_y_err", "synthetic_std_y_err", "n_seeds_ok"])
        for r in rows:
            w.writerow([r["mu"], r["ci_width_mean"], r["synthetic_mean_y_err"],
                        r["synthetic_std_y_err"], r["n_seeds_ok"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = [r["mu"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(mus, [r["ci_width_mean"] for r in rows], "o-")
    ax1.set_xlabel("privacy budget μ"); ax1.set_ylabel("mean 95% CI width")
    ax1.set_title("E3: DP regression CI width vs μ"); ax1.grid(alpha=0.3)
    ax2.plot(mus, [r["synthetic_mean_y_err"] for r in rows], "o-", label="|mean(y) err|")
    ax2.plot(mus, [r["synthetic_std_y_err"] for r in rows], "s--", label="|std(y) err|")
    ax2.set_xlabel("privacy budget μ"); ax2.set_ylabel("synthetic vs original moment error")
    ax2.set_title("E3: synthetic moment error vs μ"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e3_airquality.png", dpi=140)
    print("saved figures/e3_airquality.png")


if __name__ == "__main__":
    main()
