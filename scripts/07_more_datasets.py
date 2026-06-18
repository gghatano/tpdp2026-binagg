"""E7 (best-effort): identify + reproduce RelMSE for paper datasets D2/D5/D9.

Issue #3 asks to identify the still-unidentified paper datasets D1/D2/D3/D5/D9
from their (n, d) and the paper's references, and to best-effort reproduce their
prediction RelMSE.

In THIS reproduction environment OpenML and the UCI archive are network-blocked
(HTTP 403), so the E5 path (`sklearn.datasets.fetch_openml`) cannot be used.
GitHub raw mirrors ARE reachable, so each dataset is fetched from a pinned GitHub
mirror (cached under data/cache/, gitignored) and processed here.

Identification (by (n, d) + the dataset's role in DP linear-regression benchmarks):
  - D2 = BUPA Liver Disorders        (n=345,   d=6)   -- HIGH confidence
  - D5 = Parkinsons Telemonitoring   (n=5875,  d=21)  -- HIGH confidence (5875 rows)
  - D9 = Superconductivity           (n=21263, d=81)  -- HIGH confidence (exact (n,d))
  - D1 (182, 4) and D3 (2043, 8) remain UNIDENTIFIED (the paper appendix, the
    authoritative source for the (n,d)->dataset map, is on arXiv which is blocked
    in this environment). They are recorded as unidentified.

Protocol is unified with E3/E5 (03_real_airquality.py, 05_other_datasets.py):
  - metric: RelMSE = ||X b_hat - y||^2 / ||y||^2  (prediction error, no intercept)
  - non-private benchmark: OLS via np.linalg.lstsq
  - BinAgg: dp_linear_regression, mu=1.0, 30 reps (seed 0..29), mean+-std
  - bounds: non-private per-column data min/max; y_bounds = y min/max

NOTE: best-effort. The paper's per-dataset Table-2 RelMSE values for D2/D5/D9 are
not available here (appendix blocked), so we report our values and the matched
(n, d). Preprocessing assumptions needed to hit the paper's exact d are documented
per loader and are the main source of any discrepancy.

Outputs: results/e7_more_datasets.json/.csv, figures/e7_more_datasets.png
"""
from __future__ import annotations

import csv
import io
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from binagg import dp_linear_regression

from common import DATA, FIGURES, RESULTS, ols, save_json

warnings.filterwarnings("ignore")

N_REPS = 30
# Superconductivity (d=81) PrivTree binning is expensive and highly seed-dependent
# (some seeds build very deep trees), so it uses a documented reduced rep count.
REPS_OVERRIDE = {"Superconductivity": 5}
MU = 1.0
CACHE = DATA / "cache"

# Pinned GitHub raw mirrors (OpenML/UCI are blocked in this environment).
MIRRORS = {
    "BUPA": (
        "https://raw.githubusercontent.com/PepeAlex/BUPA_Liver_Disorders/"
        "master/bupa.data"
    ),
    "Parkinsons": (
        "https://raw.githubusercontent.com/pqrst/ParkinsonsDiseaseDataAnalysis/"
        "master/parkinsons_updrs.csv"
    ),
    "Superconductivity": (
        "https://raw.githubusercontent.com/saranggalada/ML_Superconductivity/"
        "main/superconduct/train.csv"
    ),
}

# Paper Table-2 values are unknown here (appendix on arXiv is network-blocked).
PAPER = {
    "BUPA":              {"id": "D2", "n": 345,   "d": 6,  "OLS": None, "BinAgg": None},
    "Parkinsons":        {"id": "D5", "n": 5875,  "d": 21, "OLS": None, "BinAgg": None},
    "Superconductivity": {"id": "D9", "n": 21263, "d": 81, "OLS": None, "BinAgg": None},
}

CACHE_FILES = {
    "BUPA": "bupa.data",
    "Parkinsons": "parkinsons_updrs.csv",
    "Superconductivity": "superconduct_train.csv",
}


def relmse(X, y, beta):
    return float(np.sum((X @ beta - y) ** 2) / np.sum(y ** 2))


def nonprivate_bounds(X, y):
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    yb = (float(y.min()), float(y.max()))
    return xb, yb


def fetch_text(name: str) -> str:
    """Return raw file text, using data/cache/ if present else the pinned mirror."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / CACHE_FILES[name]
    if local.exists():
        return local.read_text(encoding="utf-8")
    url = MIRRORS[name]
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
        text = r.read().decode("utf-8")
    local.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Loaders. Each returns (X, y, feature_names, source_str, note_str).
# ---------------------------------------------------------------------------
def load_bupa():
    """BUPA Liver Disorders: 7 cols, no header.

    cols = [mcv, alkphos, sgpt, sgot, gammagt, drinks, selector].
    To hit the paper's d=6 with a continuous target we predict `drinks` and use
    the other 6 columns as features (the 5 blood tests + the selector field).
    """
    cols = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
    df = pd.read_csv(io.StringIO(fetch_text("BUPA")), header=None, names=cols)
    df = df.astype(float)
    y = df["drinks"].values
    Xdf = df.drop(columns=["drinks"])  # 6 features incl. selector
    note = ("7 cols (no header): mcv,alkphos,sgpt,sgot,gammagt,drinks,selector. "
            "target=drinks; features = 5 blood tests + selector = 6 (best-effort "
            "encoding to reach d=6; selector is the original train/test split field).")
    return Xdf.values, y, list(Xdf.columns), MIRRORS["BUPA"], note


def load_parkinsons():
    """Parkinsons Telemonitoring: 22 cols incl. header.

    cols = [subject#, age, sex, test_time, motor_UPDRS, total_UPDRS, + 16 voice].
    To hit the paper's d=21 we predict total_UPDRS and use all 21 remaining
    columns as features (this necessarily includes subject# and motor_UPDRS).
    """
    df = pd.read_csv(io.StringIO(fetch_text("Parkinsons")))
    df = df.astype(float)
    y = df["total_UPDRS"].values
    Xdf = df.drop(columns=["total_UPDRS"])  # 21 features
    note = ("22 cols: subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,+16 voice. "
            "target=total_UPDRS; features = all 21 remaining cols (best-effort to "
            "reach d=21; this includes the subject# id and motor_UPDRS, which is "
            "strongly correlated with the target).")
    return Xdf.values, y, list(Xdf.columns), MIRRORS["Parkinsons"], note


def load_superconductivity():
    """Superconductivity: 82 cols incl. header; last col critical_temp is target."""
    df = pd.read_csv(io.StringIO(fetch_text("Superconductivity")))
    df = df.astype(float)
    y = df["critical_temp"].values
    Xdf = df.drop(columns=["critical_temp"])  # 81 features
    note = ("82 cols (header): 81 extracted features + critical_temp. "
            "target=critical_temp; all 81 features kept -> d=81 (unambiguous).")
    return Xdf.values, y, list(Xdf.columns), MIRRORS["Superconductivity"], note


LOADERS = {
    "BUPA": load_bupa,
    "Parkinsons": load_parkinsons,
    "Superconductivity": load_superconductivity,
}

# Identified but the data could not be fetched in this environment (UCI/Kaggle
# blocked, no reachable GitHub mirror). Recorded, not run.
IDENTIFIED_UNAVAILABLE = {
    "D1": {"n": 182, "d": 4, "name": "LT-FS-ID: Intrusion Detection in WSNs",
           "source": "UCI dataset 715 (archive.ics.uci.edu/dataset/715)",
           "note": ("regression: predict #k-barriers from 4 features (area, sensing "
                    "range, transmission range, #sensor nodes); (n,d)=(182,4) exact "
                    "match. UCI/Kaggle/author-site all network-blocked here and no "
                    "GitHub raw mirror found, so RelMSE not reproduced.")},
}

# Datasets we could not identify -> recorded, not run.
UNIDENTIFIED = {
    "D3": {"n": 2043, "d": 8,
           "note": "not identified; (n,d) candidates inconclusive; needs paper appendix"},
}


def evaluate(X, y, n_reps):
    xb, yb = nonprivate_bounds(X, y)
    beta_ols = ols(X, y)
    ols_rel = relmse(X, y, beta_ols)
    errs = []
    for s in range(n_reps):
        r = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=MU, random_state=s)
        if np.all(np.isfinite(r.coefficients)):
            errs.append(relmse(X, y, r.coefficients))
    errs = np.array(errs)
    return ols_rel, float(errs.mean()), float(errs.std()), int(errs.size)


def main() -> None:
    results = {}
    for name, loader in LOADERS.items():
        paper = PAPER[name]
        print(f"\n=== {name} ({paper['id']}: paper n={paper['n']}, d={paper['d']}) ===")
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
        print(f"  final (n,d) = ({n},{d})  n_match={n_match} d_match={d_match}")
        print(f"  note: {note}")
        n_reps = REPS_OVERRIDE.get(name, N_REPS)
        ols_rel, ba_mean, ba_std, n_ok = evaluate(X, y, n_reps)
        print(f"  OLS RelMSE    = {ols_rel:.4f}")
        print(f"  BinAgg RelMSE = {ba_mean:.4f} +- {ba_std:.4f} "
              f"(mu={MU}, n_ok={n_ok}/{n_reps})")
        results[name] = {
            "status": "ok",
            "paper_id": paper["id"],
            "source": src,
            "preprocessing_note": note,
            "n": n, "d": d, "n_match": n_match, "d_match": d_match,
            "paper": paper,
            "ols_relmse": ols_rel,
            "binagg_relmse_mean": ba_mean,
            "binagg_relmse_std": ba_std,
            "binagg_n_ok": n_ok,
            "n_reps": n_reps,
            "features": feats,
        }

    payload = {
        "experiment": "E7 best-effort: identify + reproduce D2/D5/D9 (Issue #3)",
        "environment_note": ("OpenML and UCI are network-blocked (HTTP 403) in this "
                             "reproduction environment; datasets are fetched from "
                             "pinned GitHub raw mirrors instead of fetch_openml."),
        "metric": "RelMSE = ||X*beta - y||^2 / ||y||^2 (prediction error, no intercept)",
        "bounds": "non-private per-column data min/max; y_bounds = y min/max",
        "binagg": {"mu": MU, "n_reps_default": N_REPS,
                   "n_reps_override": REPS_OVERRIDE, "seeds": "0..n_reps-1"},
        "datasets": results,
        "identified_data_unavailable": IDENTIFIED_UNAVAILABLE,
        "unidentified": UNIDENTIFIED,
    }
    save_json("e7_more_datasets.json", payload)

    with open(RESULTS / "e7_more_datasets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "paper_id", "status", "n", "d", "paper_n", "paper_d",
                    "n_match", "d_match", "ols_relmse", "binagg_relmse_mean",
                    "binagg_relmse_std"])
        for name, r in results.items():
            p = r["paper"]
            if r["status"] != "ok":
                w.writerow([name, p["id"], r["status"], "", "", p["n"], p["d"],
                            "", "", "", "", ""])
            else:
                w.writerow([name, r["paper_id"], "ok", r["n"], r["d"], p["n"], p["d"],
                            r["n_match"], r["d_match"],
                            f"{r['ols_relmse']:.4f}", f"{r['binagg_relmse_mean']:.4f}",
                            f"{r['binagg_relmse_std']:.4f}"])
        for did, info in IDENTIFIED_UNAVAILABLE.items():
            w.writerow([did, did, "identified_data_unavailable", "", "",
                        info["n"], info["d"], "", "", "", "", ""])
        for did, info in UNIDENTIFIED.items():
            w.writerow([did, did, "unidentified", "", "", info["n"], info["d"],
                        "", "", "", "", ""])

    # --- figure: OLS vs BinAgg RelMSE per identified dataset ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [(n, r) for n, r in results.items() if r["status"] == "ok"]
    if ok:
        labels = [f"{r['paper_id']}={n}\n(n={r['n']},d={r['d']})" for n, r in ok]
        ours_ols = [r["ols_relmse"] for _, r in ok]
        ours_ba = [r["binagg_relmse_mean"] for _, r in ok]
        ours_ba_std = [r["binagg_relmse_std"] for _, r in ok]

        x = np.arange(len(labels))
        bw = 0.35
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.bar(x - 0.5 * bw, ours_ols, bw, color="#888", label="OLS (this study)")
        ax.bar(x + 0.5 * bw, ours_ba, bw, yerr=ours_ba_std, capsize=3,
               color="tab:blue", label="BinAgg (this study, mu=1)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RelMSE (prediction error)")
        ax.set_title("E7: identified D2/D5/D9 -- prediction RelMSE (best-effort, mu=1)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "e7_more_datasets.png", dpi=140)
        print("\nsaved figures/e7_more_datasets.png")


if __name__ == "__main__":
    main()
