"""E3: Real-data reproduction on UCI Air Quality (with proper -200 handling).

Reproduces the three usage modes from BinAgg's real_data_example.py:
  Option A: DP linear regression only
  Option B: DP regression + synthetic data sharing the SAME budget
  Option C: synthetic data only
across privacy budgets mu, comparing DP coefficients / CI widths against
non-private OLS, plus synthetic-vs-original moment errors.

PREPROCESSING (fixes the contamination documented earlier): this UCI dataset codes
missing values as the sentinel -200. We (1) drop the NMHC(GT) feature, which is
~90% missing (-200) in this dataset, and (2) remove rows containing -200 in any
remaining used column. With the data cleaned, the non-private OLS reference and the
synthetic-vs-original moment comparison are both trustworthy, so we now also report
the DP-vs-OLS coefficient agreement as a genuine utility metric.

Bounds come from DOMAIN KNOWLEDGE (from the BinAgg example), never from the data.
Outputs: results/e3_airquality.json/.csv, figures/e3_airquality.png
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from binagg import dp_linear_regression, generate_synthetic_data

from common import DATA, FIGURES, MU_GRID, RESULTS, ols, rel_l2, save_json

TARGET = "CO(GT)"
SEEDS = [42, 43, 44, 45, 46]

# NMHC(GT) is ~90% missing (-200) in this dataset -> drop the whole feature column
# rather than discard ~90% of the rows.
DROP_FEATURES = ["NMHC(GT)"]

# Domain-knowledge bounds per column (from BinAgg examples/real_data_example.py).
# Selected by column name so the order always matches the surviving feature set.
BOUNDS_BY_COL = {
    "PT08.S1(CO)": (0, 3000), "NMHC(GT)": (0, 1500), "C6H6(GT)": (0, 100),
    "PT08.S2(NMHC)": (0, 3000), "NOx(GT)": (0, 2000), "PT08.S3(NOx)": (0, 3000),
    "NO2(GT)": (0, 500), "PT08.S4(NO2)": (0, 3000), "PT08.S5(O3)": (0, 3000),
    "T": (-20, 50), "RH": (0, 100), "AH": (0, 3),
}
Y_BOUNDS = (0, 15)


def load_airquality():
    """Load, parse, and clean the UCI Air Quality CSV.

    Returns (X, y, feature_cols, info) where info records the cleaning effect.
    """
    data = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    data = data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    data = data.replace(",", ".", regex=True)
    data = data.apply(pd.to_numeric, errors="coerce")
    n_example = int(len(data.dropna()))            # baseline the example would keep (keeps -200)
    data = data.drop(columns=DROP_FEATURES)
    data = data.replace(-200, np.nan).dropna()     # treat the -200 sentinel as missing
    n_used = int(len(data))
    feature_cols = [c for c in data.columns if c != TARGET]
    X = data[feature_cols].values
    y = data[TARGET].values
    info = {"n_example_baseline": n_example, "n_used": n_used,
            "dropped_features": DROP_FEATURES}
    return X, y, feature_cols, info


def main() -> None:
    X, y, feature_cols, info = load_airquality()
    n, d = X.shape
    x_bounds = [BOUNDS_BY_COL[c] for c in feature_cols]
    print(f"AirQuality: n={n} (example baseline {info['n_example_baseline']}), "
          f"d={d}, dropped={info['dropped_features']}, target={TARGET}")

    # Data are now clean of -200, so the OLS reference and moment comparison are valid.
    beta_ols = ols(X, y)
    mean_y, std_y = float(y.mean()), float(y.std())
    # This real design is severely ill-conditioned (very different feature scales,
    # correlated sensors). We record it because it explains the noisy coefficient-level
    # metrics below; the median CI width is reported as a robust summary since a few
    # ill-conditioned directions inflate the mean.
    cond_xtx = float(np.linalg.cond(X.T @ X))

    rows = []
    coef_table = {"feature": feature_cols, "ols": beta_ols.round(5).tolist()}
    for mu in MU_GRID:
        ci_med, ci_mean, dp_coef_seeds, coef_rel = [], [], [], []
        meany_err, stdy_err = [], []
        for s in SEEDS:
            # Option B: regression + synthetic, shared budget
            reg, syn = dp_linear_regression(
                X, y, x_bounds=x_bounds, y_bounds=Y_BOUNDS, mu=mu,
                return_synthetic=True, clip_synthetic_output=True, random_state=s,
            )
            if np.all(np.isfinite(reg.coefficients)):
                dp_coef_seeds.append(reg.coefficients)
                coef_rel.append(rel_l2(reg.coefficients, beta_ols))
                w = reg.confidence_intervals[:, 1] - reg.confidence_intervals[:, 0]
                ci_med.append(np.nanmedian(w))
                ci_mean.append(np.nanmean(w))
            if syn.n_samples > 0:
                meany_err.append(abs(syn.y_synthetic.mean() - mean_y))
                stdy_err.append(abs(syn.y_synthetic.std() - std_y))
        dp_coef_seeds = np.array(dp_coef_seeds)
        row = {
            "mu": mu,
            "n_seeds_ok": int(len(dp_coef_seeds)),
            "ci_width_median": float(np.median(ci_med)) if ci_med else float("nan"),
            "ci_width_mean": float(np.mean(ci_mean)) if ci_mean else float("nan"),
            "dp_ols_coef_relL2_mean": float(np.mean(coef_rel)) if coef_rel else float("nan"),
            "synthetic_mean_y_err": float(np.mean(meany_err)) if meany_err else float("nan"),
            "synthetic_std_y_err": float(np.mean(stdy_err)) if stdy_err else float("nan"),
            "dp_coef_mean": dp_coef_seeds.mean(axis=0).round(5).tolist() if len(dp_coef_seeds) else None,
        }
        rows.append(row)
        coef_table[f"dp_mu{mu}"] = row["dp_coef_mean"]
        print(f"mu={mu:>4}: CIwidth median={row['ci_width_median']:.4f} mean={row['ci_width_mean']:.3f}  "
              f"DP-OLS relL2={row['dp_ols_coef_relL2_mean']:.3f}  "
              f"mean(y)err={row['synthetic_mean_y_err']:.4f}  "
              f"std(y)err={row['synthetic_std_y_err']:.4f}  (seeds ok={row['n_seeds_ok']})")

    # Option A vs C sanity (single seed, mu=1)
    res_a = dp_linear_regression(X, y, x_bounds=x_bounds, y_bounds=Y_BOUNDS, mu=1.0, random_state=42)
    syn_c = generate_synthetic_data(X, y, x_bounds=x_bounds, y_bounds=Y_BOUNDS, mu=1.0,
                                    clip_output=True, random_state=42)
    options_note = {
        "optionA_coef_mu1_seed42": res_a.coefficients.round(5).tolist(),
        "optionC_n_synthetic_mu1_seed42": int(syn_c.n_samples),
        "optionC_n_bins_used": int(syn_c.n_bins_used),
    }

    save_json("e3_airquality.json", {
        "n": int(n), "d": int(d), "target": TARGET, "features": feature_cols,
        "preprocessing": {
            "missing_sentinel": -200,
            "dropped_features": info["dropped_features"],
            "n_example_baseline": info["n_example_baseline"],
            "n_used": info["n_used"],
            "note": "NMHC(GT) dropped (~90% missing); rows with -200 in any remaining used column removed.",
        },
        "ols_coef": beta_ols.round(6).tolist(), "rows": rows, "options_note": options_note,
        "mean_y": mean_y, "std_y": std_y, "cond_xtx": cond_xtx,
        "moment_reference": "cleaned original (after -200 removal)",
        "coef_table": coef_table,
    })
    with open(RESULTS / "e3_airquality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mu", "ci_width_median", "ci_width_mean", "dp_ols_coef_relL2",
                    "synthetic_mean_y_err", "synthetic_std_y_err", "n_seeds_ok"])
        for r in rows:
            w.writerow([r["mu"], r["ci_width_median"], r["ci_width_mean"],
                        r["dp_ols_coef_relL2_mean"], r["synthetic_mean_y_err"],
                        r["synthetic_std_y_err"], r["n_seeds_ok"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = [r["mu"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1.plot(mus, [r["ci_width_median"] for r in rows], "o-")
    ax1.set_xlabel("privacy budget μ"); ax1.set_ylabel("median 95% CI width")
    ax1.set_title("E3: DP regression CI width (median) vs μ"); ax1.grid(alpha=0.3)

    ax2.plot(mus, [r["dp_ols_coef_relL2_mean"] for r in rows], "o-", color="tab:green")
    ax2.set_xlabel("privacy budget μ"); ax2.set_ylabel("relative L2 vs non-private OLS")
    ax2.set_title("E3: DP–OLS coefficient agreement vs μ"); ax2.grid(alpha=0.3)

    ax3.plot(mus, [r["synthetic_mean_y_err"] for r in rows], "o-", label="|mean(y) err|")
    ax3.plot(mus, [r["synthetic_std_y_err"] for r in rows], "s--", label="|std(y) err|")
    ax3.set_xlabel("privacy budget μ"); ax3.set_ylabel("synthetic vs original moment error")
    ax3.set_title("E3: synthetic moment error vs μ"); ax3.legend(); ax3.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e3_airquality.png", dpi=140)
    print("saved figures/e3_airquality.png")


if __name__ == "__main__":
    main()
