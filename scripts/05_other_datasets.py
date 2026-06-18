"""E5 (best-effort): DP linear-regression prediction error on 3 more UCI datasets.

Best-effort reproduction of the paper's real-data protocol (Table 2, [1]) for:
  - D4 = Abalone           (n=4177,  d=10): paper OLS=0.044 / BinAgg=0.059
  - D6 = Wine Quality      (n=6497,  d=12): paper OLS=0.016 / BinAgg=0.022
  - D8 = Appliances Energy (n=19735, d=27): paper OLS=0.438 / BinAgg=0.507

Setup is unified with E3 (03_real_airquality.py):
  - metric: RelMSE = ||X b_hat - y||^2 / ||y||^2  (prediction error, no intercept)
  - non-private benchmark: OLS via np.linalg.lstsq
  - BinAgg: from binagg import dp_linear_regression, mu=1.0, 30 reps (seed 0..29), mean+-std
  - bounds: non-private per-column data min/max; y_bounds = y min/max

Data via sklearn.datasets.fetch_openml(as_frame=True). Each loader logs the final
(n, d) and asserts/reports whether it matches the paper. On fetch failure a dataset
is skipped and reported.

NOTE: best-effort -- exact match to paper values is not guaranteed; preprocessing
assumptions are documented per loader.

Outputs: results/e5_other_datasets.json/.csv, figures/e5_other_datasets.png
"""
from __future__ import annotations

import csv
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from binagg import dp_linear_regression

from common import FIGURES, RESULTS, ols, save_json

warnings.filterwarnings("ignore")

N_REPS = 30
SEEDS = list(range(N_REPS))
MU = 1.0

PAPER = {
    "Abalone":    {"n": 4177,  "d": 10, "OLS": 0.044, "BinAgg": 0.059},
    "WineQuality": {"n": 6497, "d": 12, "OLS": 0.016, "BinAgg": 0.022},
    "Appliances": {"n": 19735, "d": 27, "OLS": 0.438, "BinAgg": 0.507},
}


def relmse(X, y, beta):
    return float(np.sum((X @ beta - y) ** 2) / np.sum(y ** 2))


def nonprivate_bounds(X, y):
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    yb = (float(y.min()), float(y.max()))
    return xb, yb


# ---------------------------------------------------------------------------
# Loaders. Each returns (X, y, feature_names, source_str, note_str) or raises.
# ---------------------------------------------------------------------------
def _try_fetch(candidates):
    """Try a list of fetch_openml kwarg dicts; return (frame, target_names, src)."""
    last = None
    for c in candidates:
        try:
            d = fetch_openml(as_frame=True, parser="auto", **c)
            return d, f"openml {c}"
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"all fetch candidates failed; last error: {last!r}")


def load_abalone():
    """Abalone: one-hot Sex (M/F/I -> 3 cols) + 7 numeric features = d=10; target=Rings."""
    d, src = _try_fetch([dict(name="abalone"), dict(data_id=183)])
    df = d.frame.copy()
    target_col = "Class_number_of_rings"
    y = df[target_col].astype(float).values
    feats = df.drop(columns=[target_col])
    # one-hot Sex (3 categories: M, F, I) + 7 numeric columns
    sex = pd.get_dummies(feats["Sex"].astype(str), prefix="Sex")
    numeric = feats.drop(columns=["Sex"]).astype(float)
    Xdf = pd.concat([sex.astype(float), numeric], axis=1)
    note = ("Sex one-hot (3 cols: M/F/I) + 7 numeric = 10; target=Rings; "
            "no missing values; one-hot NOT dropped-first (full 3-col encoding -> d=10).")
    return Xdf.values, y, list(Xdf.columns), src, note


def load_wine():
    """Wine Quality: stack red (n=1599) + white (n=4898), add type 0/1 col -> d=12; target=quality."""
    dr, src_r = _try_fetch([dict(name="wine-quality-red")])
    dw, src_w = _try_fetch([dict(name="wine-quality-white")])
    red = dr.frame.copy()
    white = dw.frame.copy()
    # red has named cols (target 'class'), white has V1..V11/Class in the same order
    feat_names = [c for c in red.columns if c != "class"]
    red_feats = red[feat_names].astype(float).values
    red_y = red["class"].astype(float).values
    white_feats = white.drop(columns=["Class"]).astype(float).values
    white_y = white["Class"].astype(float).values
    assert red_feats.shape[1] == white_feats.shape[1] == 11
    X11 = np.vstack([red_feats, white_feats])
    y = np.concatenate([red_y, white_y])
    # type: red=0, white=1
    type_col = np.concatenate([np.zeros(len(red_y)), np.ones(len(white_y))]).reshape(-1, 1)
    X = np.hstack([X11, type_col])
    names = feat_names + ["type_white"]
    note = ("red (1599) + white (4898) stacked (6497); 11 chem features + type {red=0,white=1} "
            "= 12; target=quality (cast to float).")
    return X, y, names, f"{src_r} + {src_w}", note


def load_appliances():
    """Appliances Energy: drop date (already absent in openml frame), target=Appliances; d=27."""
    d, src = _try_fetch([
        dict(name="Appliances_Energy_Prediction"),
        dict(name="Appliances"),
    ])
    df = d.frame.copy()
    target_col = "Appliances"
    y = df[target_col].astype(float).values
    Xdf = df.drop(columns=[target_col]).astype(float)
    # openml frame has no 'date' column; remaining cols: lights, T1..T9, RH_1..RH_9,
    # T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint, rv1, rv2 = 27
    note = ("openml frame has no date col; target=Appliances; all 27 remaining features "
            "kept (includes lights, rv1, rv2) -> d=27.")
    return Xdf.values, y, list(Xdf.columns), src, note


LOADERS = {
    "Abalone": load_abalone,
    "WineQuality": load_wine,
    "Appliances": load_appliances,
}


def evaluate(X, y):
    xb, yb = nonprivate_bounds(X, y)
    beta_ols = ols(X, y)
    ols_rel = relmse(X, y, beta_ols)
    errs = []
    for s in SEEDS:
        r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=MU, random_state=s)
        if np.all(np.isfinite(r.coefficients)):
            errs.append(relmse(X, y, r.coefficients))
    errs = np.array(errs)
    return ols_rel, float(errs.mean()), float(errs.std()), int(errs.size)


def main() -> None:
    results = {}
    for name, loader in LOADERS.items():
        paper = PAPER[name]
        print(f"\n=== {name} (paper n={paper['n']}, d={paper['d']}) ===")
        try:
            X, y, feats, src, note = loader()
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP: load failed: {e!r}")
            results[name] = {"status": "skipped", "error": repr(e), "paper": paper}
            continue
        n, d = X.shape
        n_match = (n == paper["n"])
        d_match = (d == paper["d"])
        print(f"  source: {src}")
        print(f"  final (n,d) = ({n},{d})  "
              f"n_match={n_match} d_match={d_match}")
        print(f"  note: {note}")
        ols_rel, ba_mean, ba_std, n_ok = evaluate(X, y)
        print(f"  OLS RelMSE   = {ols_rel:.4f}  (paper {paper['OLS']})")
        print(f"  BinAgg RelMSE= {ba_mean:.4f} +- {ba_std:.4f}  (paper {paper['BinAgg']}, "
              f"mu={MU}, n_ok={n_ok}/{N_REPS})")
        results[name] = {
            "status": "ok",
            "source": src,
            "preprocessing_note": note,
            "n": n, "d": d, "n_match": n_match, "d_match": d_match,
            "paper": paper,
            "ols_relmse": ols_rel,
            "binagg_relmse_mean": ba_mean,
            "binagg_relmse_std": ba_std,
            "binagg_n_ok": n_ok,
            "features": feats,
        }

    payload = {
        "experiment": "E5 best-effort: 3 more UCI datasets (Table 2 of [1])",
        "metric": "RelMSE = ||X*beta - y||^2 / ||y||^2 (prediction error, no intercept)",
        "bounds": "non-private per-column data min/max; y_bounds = y min/max",
        "binagg": {"mu": MU, "n_reps": N_REPS, "seeds": "0..29"},
        "datasets": results,
    }
    save_json("e5_other_datasets.json", payload)

    with open(RESULTS / "e5_other_datasets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "status", "n", "d", "paper_n", "paper_d",
                    "n_match", "d_match", "ols_relmse", "binagg_relmse_mean",
                    "binagg_relmse_std", "paper_ols", "paper_binagg"])
        for name, r in results.items():
            p = r["paper"]
            if r["status"] != "ok":
                w.writerow([name, r["status"], "", "", p["n"], p["d"], "", "",
                            "", "", "", p["OLS"], p["BinAgg"]])
            else:
                w.writerow([name, "ok", r["n"], r["d"], p["n"], p["d"],
                            r["n_match"], r["d_match"],
                            f"{r['ols_relmse']:.4f}", f"{r['binagg_relmse_mean']:.4f}",
                            f"{r['binagg_relmse_std']:.4f}", p["OLS"], p["BinAgg"]])

    # --- figure: grouped bars per dataset {OLS, BinAgg(this), paper OLS, paper BinAgg} ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [(n, r) for n, r in results.items() if r["status"] == "ok"]
    if ok:
        labels = [n for n, _ in ok]
        ours_ols = [r["ols_relmse"] for _, r in ok]
        ours_ba = [r["binagg_relmse_mean"] for _, r in ok]
        ours_ba_std = [r["binagg_relmse_std"] for _, r in ok]
        pap_ols = [r["paper"]["OLS"] for _, r in ok]
        pap_ba = [r["paper"]["BinAgg"] for _, r in ok]

        x = np.arange(len(labels))
        bw = 0.2
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - 1.5 * bw, ours_ols, bw, color="#888", label="OLS (this study)")
        ax.bar(x - 0.5 * bw, ours_ba, bw, yerr=ours_ba_std, capsize=3,
               color="tab:blue", label="BinAgg (this study, mu=1)")
        ax.bar(x + 0.5 * bw, pap_ols, bw, color="#444", label="OLS (paper)")
        ax.bar(x + 1.5 * bw, pap_ba, bw, color="tab:red", label="BinAgg (paper)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={r['n']},d={r['d']})" for l, (_, r) in zip(labels, ok)])
        ax.set_ylabel("RelMSE (prediction error)")
        ax.set_title("E5: prediction RelMSE on 3 UCI datasets (this study vs paper Table 2, mu=1)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "e5_other_datasets.png", dpi=140)
        print("\nsaved figures/e5_other_datasets.png")


if __name__ == "__main__":
    main()
