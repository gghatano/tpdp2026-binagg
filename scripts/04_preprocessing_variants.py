"""E4 (supplementary): how intuitive preprocessing choices and a coefficient-level
metric behave on UCI Air Quality.

The main result (E3) follows the paper: keep the -200 sentinel, all 12 features,
predict with RelMSE. Here we additionally try the choices a practitioner would
intuitively reach for, as a supplement (NOT the paper setting):

  variants (preprocessing of the -200 missing sentinel):
    - keep      : paper-faithful (n=9357, d=12), -200 kept
    - drop_rows : drop NMHC(GT) (~90% missing) + remove rows with -200 (n=6941, d=11)
    - impute    : feature -200 -> column mean; drop rows whose TARGET is -200 (n=7674, d=12)

  metrics per variant x mu:
    - OLS RelMSE (non-private reference)
    - BinAgg RelMSE  (prediction error, robust)
    - DP-vs-OLS coefficient relative L2  (the intuitive but fragile coefficient metric)

bounds = non-private per-column data min/max (paper's real-data choice).
Outputs: results/e4_preprocessing.json/.csv, figures/e4_preprocessing.png
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from binagg import dp_linear_regression

from common import DATA, FIGURES, MU_GRID, RESULTS, ols, rel_l2, save_json

TARGET = "CO(GT)"
SEEDS = list(range(20))
SENTINEL = -200


def _base():
    data = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    data = data.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    data = data.replace(",", ".", regex=True)
    return data.apply(pd.to_numeric, errors="coerce").dropna()


def variant_keep():
    d = _base()
    fc = [c for c in d.columns if c != TARGET]
    return d[fc].values, d[TARGET].values, fc


def variant_drop_rows():
    d = _base().drop(columns=["NMHC(GT)"]).replace(SENTINEL, np.nan).dropna()
    fc = [c for c in d.columns if c != TARGET]
    return d[fc].values, d[TARGET].values, fc


def variant_impute():
    d = _base()
    d = d[d[TARGET] != SENTINEL].copy()           # cannot impute the prediction target
    for c in d.columns:
        if c == TARGET:
            continue
        col = d[c].replace(SENTINEL, np.nan)
        d[c] = col.fillna(col.mean())             # feature -200 -> column mean
    fc = [c for c in d.columns if c != TARGET]
    return d[fc].values, d[TARGET].values, fc


def nonprivate_bounds(X, y):
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    return xb, (float(y.min()), float(y.max()))


def relmse(X, y, beta):
    return float(np.sum((X @ beta - y) ** 2) / np.sum(y ** 2))


VARIANTS = [
    ("keep (paper)", variant_keep),
    ("drop_rows", variant_drop_rows),
    ("impute", variant_impute),
]


def main() -> None:
    results = {}
    for name, loader in VARIANTS:
        X, y, fc = loader()
        xb, yb = nonprivate_bounds(X, y)
        beta_ols = ols(X, y)
        ols_rel = relmse(X, y, beta_ols)
        rows = []
        for mu in MU_GRID:
            rm, cl = [], []
            for s in SEEDS:
                r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=mu, random_state=s)
                if np.all(np.isfinite(r.coefficients)):
                    rm.append(relmse(X, y, r.coefficients))
                    cl.append(rel_l2(r.coefficients, beta_ols))
            rm, cl = np.array(rm), np.array(cl)
            rows.append({"mu": mu,
                         "binagg_relmse_mean": float(rm.mean()), "binagg_relmse_std": float(rm.std()),
                         "coef_relL2_mean": float(cl.mean()), "coef_relL2_std": float(cl.std())})
        results[name] = {"n": int(X.shape[0]), "d": int(X.shape[1]),
                         "ols_relmse": ols_rel, "rows": rows}
        print(f"[{name:14s}] n={X.shape[0]:5d} d={X.shape[1]:2d} OLS RelMSE={ols_rel:.3f}")
        for r in rows:
            print(f"    mu={r['mu']:>4}: BinAgg RelMSE={r['binagg_relmse_mean']:.3f}  "
                  f"coef relL2={r['coef_relL2_mean']:.2f}±{r['coef_relL2_std']:.2f}")

    save_json("e4_preprocessing.json", {
        "metric_relmse": "||X*beta - y||^2 / ||y||^2 (prediction; robust)",
        "metric_coef": "||beta_dp - beta_ols||_2 / ||beta_ols||_2 (coefficient; fragile on ill-conditioned designs)",
        "bounds": "non-private per-column data min/max", "n_reps": len(SEEDS),
        "variants": results,
    })
    with open(RESULTS / "e4_preprocessing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "d", "mu", "ols_relmse", "binagg_relmse", "coef_relL2"])
        for name, v in results.items():
            for r in v["rows"]:
                w.writerow([name, v["n"], v["d"], r["mu"], v["ols_relmse"],
                            r["binagg_relmse_mean"], r["coef_relL2_mean"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = MU_GRID
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, v in results.items():
        ax1.plot(mus, [r["binagg_relmse_mean"] for r in v["rows"]], "o-", label=f"{name} (n={v['n']})")
    ax1.set_xlabel("privacy budget μ"); ax1.set_ylabel("BinAgg RelMSE (prediction)")
    ax1.set_title("E4: prediction RelMSE by preprocessing"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    for name, v in results.items():
        ax2.plot(mus, [r["coef_relL2_mean"] for r in v["rows"]], "s--", label=name)
    ax2.set_xlabel("privacy budget μ"); ax2.set_ylabel("DP–OLS coefficient relative L2")
    ax2.set_title("E4: coefficient-level metric (fragile)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e4_preprocessing.png", dpi=140)
    print("saved figures/e4_preprocessing.png")


if __name__ == "__main__":
    main()
