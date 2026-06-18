"""E9: coefficient-level comparison (OLS vs BinAgg corrected / naive) on real data.

A follow-up to E7/E8 requested as an additional experiment. So far the real-data
evaluation used only prediction RelMSE; here we compare at the COEFFICIENT level,
which is where DP noise and design ill-conditioning bite hardest (cf. the
supplementary standardization experiment and Issue #6).

For each dataset we treat the non-private OLS solution as the reference (there is no
"true" beta on real data) and report, over repeated DP runs (mu=1):
  - cond(X^T X): the design's condition number (ill-conditioning indicator)
  - relL2_corrected = ||beta_binagg - beta_ols|| / ||beta_ols||  (bias-corrected beta)
  - relL2_naive     = ||beta_naive  - beta_ols|| / ||beta_ols||  (uncorrected beta)
  - ci_cover_ols: fraction of coefficients whose BinAgg 95% CI contains the OLS value
  - ci_width: mean width of the BinAgg 95% CI

Each is computed twice: on the RAW design and on a STANDARDIZED design (per-column
z-score), since standardization lowers cond and is expected to stabilize the
coefficient comparison. Prediction RelMSE is unchanged by standardization, so it is
not the point here -- coefficient agreement is.

Datasets reuse the exact loaders from scripts 07/08 (same preprocessing as E7/E8):
D2 BUPA, D5 Parkinsons, D9 Superconductivity, and Wine (D6). Data come from the
pinned GitHub mirrors cached under data/cache/ (OpenML/UCI are blocked here).

Outputs: results/e9_coefficients.json/.csv, figures/e9_coefficients.png
"""
from __future__ import annotations

import csv
import importlib.util
import warnings
from pathlib import Path

import numpy as np

from binagg import dp_linear_regression

from common import FIGURES, RESULTS, ols, save_json

warnings.filterwarnings("ignore")

MU = 1.0
HERE = Path(__file__).resolve().parent


def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m07 = _load_module("07_more_datasets.py", "more_datasets")
_m08 = _load_module("08_wine_preprocessing.py", "wine_prep")


def _wine_design():
    X11, type_white, y, _ = _m08.load_combined()
    X = np.hstack([X11, type_white])  # 11 chem + type = d=12 (same as E5/E8)
    return X, y


# name -> (loader returning (X, y), n_reps)  [D9 fewer reps: 81-dim binning is costly]
DATASETS = {
    "D2 BUPA":            (lambda: _m07.load_bupa()[:2], 30),
    "D5 Parkinsons":      (lambda: _m07.load_parkinsons()[:2], 30),
    "D9 Superconduct.":   (lambda: _m07.load_superconductivity()[:2], 5),
    "D6 Wine":            (_wine_design, 30),
}


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def coeff_eval(X, y, n_reps):
    beta_ols = ols(X, y)
    nb = np.linalg.norm(beta_ols)
    cond = float(np.linalg.cond(X.T @ X))
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    yb = (float(y.min()), float(y.max()))
    rc, rn, cov, wid = [], [], [], []
    for s in range(n_reps):
        r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=MU, random_state=s)
        bc, bn = np.asarray(r.coefficients), np.asarray(r.naive_coefficients)
        if not np.all(np.isfinite(bc)):
            continue
        rc.append(np.linalg.norm(bc - beta_ols) / nb)
        rn.append(np.linalg.norm(bn - beta_ols) / nb)
        ci = np.asarray(r.confidence_intervals)
        cov.append(float(np.mean((ci[:, 0] <= beta_ols) & (beta_ols <= ci[:, 1]))))
        wid.append(float(np.mean(ci[:, 1] - ci[:, 0])))
    f = lambda a: (float(np.mean(a)), float(np.std(a)))  # noqa: E731
    return {
        "d": int(X.shape[1]), "cond_XtX": cond, "n_ok": len(rc),
        "relL2_corrected": f(rc), "relL2_naive": f(rn),
        "ci_cover_ols": f(cov), "ci_width": f(wid),
        "ols_coef_norm": float(nb),
    }


def main() -> None:
    results = {}
    for name, (loader, reps) in DATASETS.items():
        print(f"\n=== {name} ===")
        X, y = loader()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        raw = coeff_eval(X, y, reps)
        std = coeff_eval(standardize(X), y, reps)
        results[name] = {"raw": raw, "standardized": std, "n": int(X.shape[0])}
        for tag, r in (("raw", raw), ("standardized", std)):
            print(f"  [{tag:12s}] cond(XtX)={r['cond_XtX']:.2e}  "
                  f"relL2(corrected)={r['relL2_corrected'][0]:.3f}±{r['relL2_corrected'][1]:.3f}  "
                  f"relL2(naive)={r['relL2_naive'][0]:.3f}  "
                  f"CI_cover_OLS={r['ci_cover_ols'][0]:.2f}")

    payload = {
        "experiment": "E9: coefficient-level OLS vs BinAgg (corrected/naive), mu=1",
        "reference": "non-private OLS beta (no true beta on real data)",
        "metrics": {
            "relL2_corrected": "||beta_binagg - beta_ols|| / ||beta_ols||",
            "relL2_naive": "||beta_naive - beta_ols|| / ||beta_ols||",
            "ci_cover_ols": "fraction of coeffs whose BinAgg 95% CI contains beta_ols",
            "cond_XtX": "condition number of X^T X (ill-conditioning indicator)",
        },
        "note": ("prediction RelMSE is invariant to standardization; coefficient "
                 "agreement is not. Large relL2 on raw + small on standardized "
                 "indicates the gap is driven by design ill-conditioning, not by the "
                 "DP mechanism per se (cf. Issue #6)."),
        "datasets": results,
    }
    save_json("e9_coefficients.json", payload)

    with open(RESULTS / "e9_coefficients.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "design", "n", "d", "cond_XtX",
                    "relL2_corrected_mean", "relL2_corrected_std",
                    "relL2_naive_mean", "ci_cover_ols_mean", "ci_width_mean"])
        for name, r in results.items():
            for tag in ("raw", "standardized"):
                e = r[tag]
                w.writerow([name, tag, r["n"], e["d"], f"{e['cond_XtX']:.3e}",
                            f"{e['relL2_corrected'][0]:.4f}",
                            f"{e['relL2_corrected'][1]:.4f}",
                            f"{e['relL2_naive'][0]:.4f}",
                            f"{e['ci_cover_ols'][0]:.3f}",
                            f"{e['ci_width'][0]:.4f}"])

    # --- figure: relL2(corrected) raw vs standardized per dataset (log y) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    raw_v = [results[n]["raw"]["relL2_corrected"][0] for n in names]
    raw_e = [results[n]["raw"]["relL2_corrected"][1] for n in names]
    std_v = [results[n]["standardized"]["relL2_corrected"][0] for n in names]
    std_e = [results[n]["standardized"]["relL2_corrected"][1] for n in names]
    x = np.arange(len(names))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.5 * bw, raw_v, bw, yerr=raw_e, capsize=3, color="tab:red",
           label="raw design")
    ax.bar(x + 0.5 * bw, std_v, bw, yerr=std_e, capsize=3, color="tab:blue",
           label="standardized design")
    ax.axhline(1.0, color="#888", ls="--", lw=1, label="relL2 = 1 (β差が|β|並み)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(cond {results[n]['raw']['cond_XtX']:.0e})" for n in names],
                       fontsize=8)
    ax.set_ylabel("relL2(BinAgg corrected vs OLS)  [log]")
    ax.set_title("E9: coefficient agreement with OLS (mu=1) -- raw vs standardized")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "e9_coefficients.png", dpi=140)
    print("\nsaved figures/e9_coefficients.png")


if __name__ == "__main__":
    main()
