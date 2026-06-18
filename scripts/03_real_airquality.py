"""E3 (paper-faithful): DP linear-regression prediction error on UCI Air Quality.

Reproduces the paper's real-data protocol for D7 = Air Quality (Table 2, [1]):
  - data: n=9357, d=12 (parse ';'/comma decimals, to_numeric + dropna; the -200
    missing-value sentinel is KEPT and all 12 features are used, as in the paper)
  - metric: RelMSE = ||X b_hat - y||^2 / ||y||^2  (prediction error, NOT coefficient error)
  - privacy: mu-GDP, mu=1 (Table 2), also swept over {0.5,1,2,5}
  - bounds: non-private (data min/max per column) -- the paper uses non-private bounds
  - benchmark: non-private OLS. Paper D7 reported: OLS=0.441, BinAgg=0.463.

Smoke-validated: this setup reproduces OLS RelMSE = 0.441 and BinAgg ~ 0.45 at mu=1.

DEVIATIONS from the paper (logged; see docs/plans/E3-paper-faithful.md):
  - only BinAgg + OLS in the main table (AdaSSP is added separately in 06_adassp.py;
    DP-GD is outside scope -- the paper notes it is highly tuning-sensitive)
  - additionally sweep mu and report a cleaned-data variant (-200 removed,
    NMHC(GT) dropped; n=6941, d=11) as a robustness comparison (NOT the paper setting).

Outputs: results/e3_airquality.json/.csv, figures/e3_airquality.png
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from binagg import dp_linear_regression, generate_synthetic_data

from common import DATA, FIGURES, MU_GRID, RESULTS, ols, save_json

TARGET = "CO(GT)"
N_REPS = 100  # matches the paper's 100 repetitions
SEEDS = list(range(N_REPS))

# Paper-reported RelMSE for D7 (Air Quality), Table 2 of [1].
PAPER_D7 = {"OLS": 0.441, "BinAgg": 0.463, "AdaSSP": 0.682, "DP-GD": 0.852}


def load_raw():
    """Paper-faithful load: parse + dropna only; -200 kept, all 12 features used."""
    data = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    data = data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    data = data.replace(",", ".", regex=True)
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    feature_cols = [c for c in data.columns if c != TARGET]
    return data[feature_cols].values, data[TARGET].values, feature_cols


def load_clean():
    """Robustness variant (deviation): drop NMHC(GT) (~90% missing) and remove -200 rows."""
    data = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    data = data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    data = data.replace(",", ".", regex=True)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.drop(columns=["NMHC(GT)"]).replace(-200, np.nan).dropna()
    feature_cols = [c for c in data.columns if c != TARGET]
    return data[feature_cols].values, data[TARGET].values, feature_cols


def nonprivate_bounds(X, y):
    """Non-private bounds = per-column data min/max (the paper's choice for real data)."""
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    yb = (float(y.min()), float(y.max()))
    return xb, yb


def relmse(X, y, beta):
    return float(np.sum((X @ beta - y) ** 2) / np.sum(y ** 2))


def binagg_relmse_curve(X, y, label):
    xb, yb = nonprivate_bounds(X, y)
    beta_ols = ols(X, y)
    ols_rel = relmse(X, y, beta_ols)
    rows = []
    for mu in MU_GRID:
        errs = []
        for s in SEEDS:
            r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=mu, random_state=s)
            if np.all(np.isfinite(r.coefficients)):
                errs.append(relmse(X, y, r.coefficients))
        errs = np.array(errs)
        rows.append({"mu": mu, "binagg_relmse_mean": float(errs.mean()),
                     "binagg_relmse_std": float(errs.std()), "n_ok": int(errs.size)})
        print(f"  [{label}] mu={mu:>4}: BinAgg RelMSE={errs.mean():.3f}±{errs.std():.3f} "
              f"(OLS {ols_rel:.3f}, n_ok={errs.size})")
    return ols_rel, rows


def main() -> None:
    # --- Paper-faithful (primary) ---
    Xr, yr, fc_r = load_raw()
    print(f"[paper-faithful] n={Xr.shape[0]}, d={Xr.shape[1]} (-200 kept, all features)")
    ols_raw, rows_raw = binagg_relmse_curve(Xr, yr, "paper-faithful")

    # --- Robustness deviation: cleaned data ---
    Xc, yc, fc_c = load_clean()
    print(f"[clean variant] n={Xc.shape[0]}, d={Xc.shape[1]} (-200 removed, NMHC(GT) dropped)")
    ols_clean, rows_clean = binagg_relmse_curve(Xc, yc, "clean")

    # mu=1 vs paper Table 2 (D7)
    binagg_mu1 = next(r["binagg_relmse_mean"] for r in rows_raw if r["mu"] == 1.0)
    print(f"\n=== mu=1 vs paper D7 ===  OLS: {ols_raw:.3f} (paper {PAPER_D7['OLS']}) | "
          f"BinAgg: {binagg_mu1:.3f} (paper {PAPER_D7['BinAgg']})")

    # Synthetic-data sanity (Option C, mu=1, paper-faithful data)
    xb, yb = nonprivate_bounds(Xr, yr)
    syn = generate_synthetic_data(Xr, yr, x_bounds=xb, y_bounds=yb, mu=1.0,
                                  clip_output=True, random_state=42)

    save_json("e3_airquality.json", {
        "metric": "RelMSE = ||X*beta - y||^2 / ||y||^2 (prediction error)",
        "bounds": "non-private per-column data min/max",
        "n_reps": N_REPS,
        "paper_reported_D7": PAPER_D7,
        "paper_faithful": {"n": int(Xr.shape[0]), "d": int(Xr.shape[1]),
                           "features": fc_r, "ols_relmse": ols_raw, "rows": rows_raw},
        "clean_variant": {"n": int(Xc.shape[0]), "d": int(Xc.shape[1]),
                          "features": fc_c, "ols_relmse": ols_clean, "rows": rows_clean,
                          "note": "deviation: -200 removed, NMHC(GT) dropped (NOT the paper setting)"},
        "option_c_sanity_mu1": {"n_synthetic": int(syn.n_samples),
                                "n_bins_used": int(syn.n_bins_used)},
    })
    with open(RESULTS / "e3_airquality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mu", "binagg_relmse_mean", "binagg_relmse_std", "ols_relmse"])
        for r in rows_raw:
            w.writerow(["paper_faithful", r["mu"], r["binagg_relmse_mean"], r["binagg_relmse_std"], ols_raw])
        for r in rows_clean:
            w.writerow(["clean", r["mu"], r["binagg_relmse_mean"], r["binagg_relmse_std"], ols_clean])

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = [r["mu"] for r in rows_raw]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.axhline(ols_raw, color="k", ls=":", label=f"OLS (non-private) = {ols_raw:.3f}")
    ax1.axhline(PAPER_D7["BinAgg"], color="tab:red", ls="--", lw=1,
                label=f"paper BinAgg (D7) = {PAPER_D7['BinAgg']}")
    ax1.errorbar(mus, [r["binagg_relmse_mean"] for r in rows_raw],
                 yerr=[r["binagg_relmse_std"] for r in rows_raw], fmt="o-",
                 color="tab:blue", capsize=3, label="BinAgg (paper-faithful)")
    ax1.set_xlabel("privacy budget μ"); ax1.set_ylabel("RelMSE (prediction)")
    ax1.set_title("E3: prediction RelMSE vs μ (paper-faithful)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.bar([0, 1], [ols_raw, binagg_mu1], width=0.5, color=["#888", "tab:blue"], label="this study (μ=1)")
    ax2.bar([0.0, 1.0], [PAPER_D7["OLS"], PAPER_D7["BinAgg"]], width=0.18,
            color=["#444", "tab:red"], label="paper D7 (Table 2)")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["OLS", "BinAgg"])
    ax2.set_ylabel("RelMSE"); ax2.set_title("E3: μ=1 vs paper Table 2 (D7)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e3_airquality.png", dpi=140)
    print("saved figures/e3_airquality.png")


if __name__ == "__main__":
    main()
