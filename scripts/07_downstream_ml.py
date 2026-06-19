"""E4 (best-effort): downstream-ML utility of BinAgg synthetic data (paper Section 5.2.2).

Reproduces the paper's "Synthetic Data Generation" evaluation (Table 3, [1]):
train non-linear regressors on BinAgg synthetic data, evaluate prediction error
on a held-out REAL test set (train-on-synthetic / test-on-real).

Protocol (paper Section 5.2.2):
  - data:    D4 Abalone(4177,10) / D6 Wine(6497,12) / D7 AirQuality(9357,12) /
             D8 Appliances(19735,27)  -- the (n,d)-identifiable subset of D1-D8.
  - split:   80/20 train/test (seed-fixed). Synthetic generated from the TRAIN
             partition only (no test leakage); evaluation on the REAL test set.
  - models:  RandomForest, SVR, MLP (exact), + HistGradientBoosting as an
             XGBoost stand-in (DEVIATION, see docs/plans/E4-downstream-ml.md).
  - metric:  RelMSE_var = ||y_pred - y_test||^2 / ||y_test - mean(y_test)||^2
             ( = 1 - R^2 ).  See the metric note below for why this (not the
             ||y||^2 normalization of Table 2) matches the Table 3 scale.
  - privacy: eps=1, delta=1/n^1.1, converted to mu-GDP via mu_from_eps_delta
             (paper uses eps=1; BinAgg is mu-GDP). mu ~ 0.28-0.31 here.
  - reps:    10 synthetic datasets per dataset (paper generates 10).

METRIC NOTE (interpretation; logged as a deviation):
  Table 2 (linear regression, our E3/E5) uses RelMSE = ||y_pred-y||^2/||y||^2 and
  we matched it (Abalone OLS 0.044). But Table 3's "Original" column is on a
  completely different scale (Abalone 0.487, AirQuality 0.489). Empirically only
  the VARIANCE normalization MSE/Var(y) (= 1-R^2, equiv. to standardizing the
  target) reproduces that scale for BOTH datasets (Abalone ~0.43, AirQuality
  ~0.40 vs paper 0.487/0.489), whereas ||y||^2 gives 0.044 vs 0.33 -- internally
  inconsistent. So we report RelMSE_var as the PRIMARY downstream metric (matched
  to Table 3) and keep the ||y||^2-normalized value for continuity with E3/E5.

DEVIATIONS (logged; docs/plans/E4-downstream-ml.md):
  - downstream RelMSE normalized by Var(y) (=1-R^2) to match Table 3's scale (above).
  - XGBoost -> HistGradientBoostingRegressor (avoids a heavy new dependency).
  - competing DP-SDG methods (AIM/DP-CTGAN/...) NOT reimplemented; paper Table 3
    values are quoted for context only (best-effort; separate issue).
  - SVR/MLP use a StandardScaler fit on the training partition (Pipeline, no leak);
    paper does not specify scaling. Tree models need none.
  - SVR/MLP TRAINING set capped at FIT_CAP=5000 (fixed-seed subsample) for tractability
    on large datasets; tree models use the full training set; test set never capped.

Outputs: results/e4_downstream_ml.json/.csv, results/figures/e4_downstream_ml.png
"""
from __future__ import annotations

import csv
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from binagg import generate_synthetic_data, mu_from_eps_delta

from common import DATA, FIGURES, RESULTS, save_json

warnings.filterwarnings("ignore")

N_REPS = 10          # paper generates 10 synthetic datasets per method
TEST_SIZE = 0.20     # paper's 80/20 split
SPLIT_SEED = 0       # fixed train/test split, shared by Original and BinAgg
EPS = 1.0            # paper's privacy budget (Table 3)
# SVR/MLP are O(n^2)~ and intractable on the largest datasets; cap their TRAINING
# size by a fixed-seed subsample (tree models always use the full training set).
# DEVIATION: logged in docs/plans/E4-downstream-ml.md. Test set is never subsampled.
FIT_CAP = 5000
CAP_MODELS = {"SVR", "MLP"}

# Paper Table 3 (Average RelMSE across the 4 downstream models), Original/BinAgg columns.
PAPER_T3 = {
    "Abalone":    {"Original": 0.487, "BinAgg": 0.731},
    "WineQuality": {"Original": 0.628, "BinAgg": 1.195},
    "AirQuality": {"Original": 0.489, "BinAgg": 0.584},
    "Appliances": {"Original": 0.683, "BinAgg": 1.490},
}


def relmse_var(y_true, y_pred) -> float:
    """Primary downstream metric: MSE / Var(y) = 1 - R^2 (matches paper Table 3 scale)."""
    sse = np.sum((y_pred - y_true) ** 2)
    return float(sse / np.sum((y_true - y_true.mean()) ** 2))


def relmse_ynorm(y_true, y_pred) -> float:
    """||y||^2-normalized RelMSE (Table 2 / E3/E5 definition), kept for continuity."""
    return float(np.sum((y_pred - y_true) ** 2) / np.sum(y_true ** 2))


def make_models():
    """4 downstream regressors (paper: XGBoost/RF/SVR/MLP; GB stands in for XGBoost)."""
    return {
        "GradBoost(XGB-stand-in)": HistGradientBoostingRegressor(random_state=0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
        "SVR": make_pipeline(StandardScaler(), SVR()),  # scaled (deviation)
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0),
        ),
    }


# ---------------------------------------------------------------------------
# Loaders (replicated from E3/E5 so this script is self-contained).
# Each returns (X, y, feature_names, source_str, note_str).
# ---------------------------------------------------------------------------
def _try_fetch(candidates):
    last = None
    for c in candidates:
        try:
            d = fetch_openml(as_frame=True, parser="auto", **c)
            return d, f"openml {c}"
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"all fetch candidates failed; last error: {last!r}")


def load_airquality():
    """D7: parse ';'/comma decimals, to_numeric + dropna; -200 kept, all 12 features."""
    df = pd.read_csv(DATA / "AirQualityUCI.csv", sep=";")
    df = df.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"])
    df = df.replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").dropna()
    target = "CO(GT)"
    feats = [c for c in df.columns if c != target]
    note = "-200 missing sentinel KEPT; all 12 features (paper-faithful)."
    return df[feats].values, df[target].values, feats, "local AirQualityUCI.csv", note


def load_abalone():
    d, src = _try_fetch([dict(name="abalone"), dict(data_id=183)])
    df = d.frame.copy()
    y = df["Class_number_of_rings"].astype(float).values
    feats = df.drop(columns=["Class_number_of_rings"])
    sex = pd.get_dummies(feats["Sex"].astype(str), prefix="Sex").astype(float)
    numeric = feats.drop(columns=["Sex"]).astype(float)
    Xdf = pd.concat([sex, numeric], axis=1)
    note = "Sex one-hot (3 cols) + 7 numeric = 10; target=Rings."
    return Xdf.values, y, list(Xdf.columns), src, note


def load_wine():
    dr, src_r = _try_fetch([dict(name="wine-quality-red")])
    dw, src_w = _try_fetch([dict(name="wine-quality-white")])
    red, white = dr.frame.copy(), dw.frame.copy()
    feat_names = [c for c in red.columns if c != "class"]
    red_feats = red[feat_names].astype(float).values
    red_y = red["class"].astype(float).values
    white_feats = white.drop(columns=["Class"]).astype(float).values
    white_y = white["Class"].astype(float).values
    X11 = np.vstack([red_feats, white_feats])
    y = np.concatenate([red_y, white_y])
    type_col = np.concatenate([np.zeros(len(red_y)), np.ones(len(white_y))]).reshape(-1, 1)
    X = np.hstack([X11, type_col])
    note = "red+white stacked (6497); 11 chem + type = 12; target=quality."
    return X, y, feat_names + ["type_white"], f"{src_r} + {src_w}", note


def load_appliances():
    d, src = _try_fetch([dict(name="Appliances_Energy_Prediction"), dict(name="Appliances")])
    df = d.frame.copy()
    y = df["Appliances"].astype(float).values
    Xdf = df.drop(columns=["Appliances"]).astype(float)
    note = "no date col; all 27 features (incl rv1/rv2); target=Appliances."
    return Xdf.values, y, list(Xdf.columns), src, note


LOADERS = {
    "Abalone": load_abalone,
    "WineQuality": load_wine,
    "AirQuality": load_airquality,
    "Appliances": load_appliances,
}


def nonprivate_bounds(X, y):
    xb = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(X.shape[1])]
    return xb, (float(y.min()), float(y.max()))


def fit_eval_all(X_train, y_train, X_test, y_test):
    """Train each model on (X_train,y_train); return {model: {"var":.., "ynorm":..}}."""
    out = {}
    for name, model in make_models().items():
        try:
            Xt, yt = X_train, y_train
            if name in CAP_MODELS and len(y_train) > FIT_CAP:
                idx = np.random.default_rng(12345).choice(len(y_train), FIT_CAP, replace=False)
                Xt, yt = X_train[idx], y_train[idx]
            model.fit(Xt, yt)
            p = model.predict(X_test)
            out[name] = {"var": relmse_var(y_test, p), "ynorm": relmse_ynorm(y_test, p)}
        except Exception as e:  # noqa: BLE001
            out[name] = {"var": float("nan"), "ynorm": float("nan")}
            print(f"      [{name}] FAILED: {e!r}")
    return out


def evaluate_dataset(name, X, y):
    n, d = X.shape
    # delta uses the FULL dataset n (paper: delta = 1/n^1.1); convert eps=1 -> mu-GDP.
    delta = float(n ** -1.1)
    mu = float(mu_from_eps_delta(EPS, delta))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    print(f"  n={n} d={d}  split train={len(y_tr)}/test={len(y_te)}  "
          f"eps={EPS} delta={delta:.2e} -> mu={mu:.3f}")

    models = list(make_models().keys())

    # --- Original (non-private): train on REAL train, test on REAL test ---
    orig = fit_eval_all(X_tr, y_tr, X_te, y_te)
    orig_mean = float(np.nanmean([orig[m]["var"] for m in models]))
    orig_mean_ynorm = float(np.nanmean([orig[m]["ynorm"] for m in models]))
    print(f"  Original  RelMSE_var(avg) = {orig_mean:.3f}  (paper {PAPER_T3[name]['Original']})  "
          f"per-model={ {k: round(orig[k]['var'],3) for k in models} }")

    # --- BinAgg synthetic: generate from TRAIN partition, train models, test on REAL test ---
    xb, yb = nonprivate_bounds(X_tr, y_tr)
    per_rep = []          # list of {model: {"var":..,"ynorm":..}} per synthetic dataset
    for s in range(N_REPS):
        syn = generate_synthetic_data(
            X_tr, y_tr, x_bounds=xb, y_bounds=yb, mu=mu,
            clip_output=True, random_state=1000 + s)
        Xs, ys = syn.X_synthetic, syn.y_synthetic
        if Xs is None or len(ys) < d + 2:
            print(f"      rep {s}: too few synthetic samples ({0 if Xs is None else len(ys)}), skip")
            continue
        per_rep.append(fit_eval_all(np.asarray(Xs), np.asarray(ys), X_te, y_te))
    # aggregate: per-model mean over reps, then mean over models (paper Table 3 cell)
    binagg_per_model = {m: float(np.nanmean([r[m]["var"] for r in per_rep])) for m in models}
    binagg_per_model_std = {m: float(np.nanstd([r[m]["var"] for r in per_rep])) for m in models}
    binagg_mean = float(np.nanmean(list(binagg_per_model.values())))
    binagg_mean_ynorm = float(np.nanmean(
        [np.nanmean([r[m]["ynorm"] for r in per_rep]) for m in models]))
    print(f"  BinAgg    RelMSE_var(avg) = {binagg_mean:.3f}  (paper {PAPER_T3[name]['BinAgg']})  "
          f"reps_ok={len(per_rep)}/{N_REPS}")
    print(f"            per-model={ {k: round(binagg_per_model[k],3) for k in models} }")

    return {
        "n": n, "d": d, "n_train": len(y_tr), "n_test": len(y_te),
        "eps": EPS, "delta": delta, "mu": mu, "n_reps_ok": len(per_rep),
        "paper": PAPER_T3[name],
        "original_per_model": {m: orig[m]["var"] for m in models},
        "original_mean": orig_mean, "original_mean_ynorm": orig_mean_ynorm,
        "binagg_per_model_mean": binagg_per_model,
        "binagg_per_model_std": binagg_per_model_std,
        "binagg_mean": binagg_mean, "binagg_mean_ynorm": binagg_mean_ynorm,
    }


def main() -> None:
    t0 = time.time()
    results = {}
    for name, loader in LOADERS.items():
        print(f"\n=== {name} ===")
        try:
            X, y, feats, src, note = loader()
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP load: {e!r}")
            results[name] = {"status": "skipped", "error": repr(e), "paper": PAPER_T3[name]}
            continue
        r = evaluate_dataset(name, X, y)
        r.update({"status": "ok", "source": src, "preprocessing_note": note, "features": feats})
        results[name] = r

    payload = {
        "experiment": "E4 best-effort: downstream-ML utility of synthetic data (paper Section 5.2.2, Table 3)",
        "metric": "PRIMARY RelMSE_var = MSE/Var(y_test) = 1-R^2 (matches paper Table 3 scale); "
                  "secondary RelMSE_ynorm = ||y_pred-y||^2/||y||^2 (Table 2 / E3-E5 def, for continuity)",
        "metric_note": "Table 3's Original scale (Abalone 0.487, AirQuality 0.489) is reproduced only by "
                       "the variance normalization, not the ||y||^2 one (0.044 vs 0.33). See script header.",
        "protocol": {
            "split": "80/20 train/test, synthetic from train only, eval on real test",
            "models": list(make_models().keys()),
            "privacy": f"eps={EPS}, delta=1/n^1.1 -> mu via mu_from_eps_delta",
            "n_reps": N_REPS,
        },
        "deviations": "XGBoost->HistGradientBoosting; competitors quoted from paper only; SVR/MLP scaled.",
        "datasets": results,
    }
    save_json("e4_downstream_ml.json", payload)

    with open(RESULTS / "e4_downstream_ml.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "status", "n", "d", "mu",
                    "original_relmse_var", "binagg_relmse_var", "paper_original", "paper_binagg",
                    "original_relmse_ynorm", "binagg_relmse_ynorm"])
        for name, r in results.items():
            if r.get("status") != "ok":
                p = r["paper"]
                w.writerow([name, r.get("status", "?"), "", "", "", "", "", p["Original"], p["BinAgg"], "", ""])
            else:
                w.writerow([name, "ok", r["n"], r["d"], f"{r['mu']:.3f}",
                            f"{r['original_mean']:.4f}", f"{r['binagg_mean']:.4f}",
                            r["paper"]["Original"], r["paper"]["BinAgg"],
                            f"{r['original_mean_ynorm']:.4f}", f"{r['binagg_mean_ynorm']:.4f}"])

    # --- figure: grouped bars per dataset {Original(this), BinAgg(this), Original(paper), BinAgg(paper)} ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [(n, r) for n, r in results.items() if r.get("status") == "ok"]
    if ok:
        labels = [n for n, _ in ok]
        x = np.arange(len(labels))
        bw = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 1.5 * bw, [r["original_mean"] for _, r in ok], bw, color="#bbb", label="Original (this study)")
        ax.bar(x - 0.5 * bw, [r["binagg_mean"] for _, r in ok], bw, color="tab:blue", label="BinAgg (this study)")
        ax.bar(x + 0.5 * bw, [r["paper"]["Original"] for _, r in ok], bw, color="#444", label="Original (paper)")
        ax.bar(x + 1.5 * bw, [r["paper"]["BinAgg"] for _, r in ok], bw, color="tab:red", label="BinAgg (paper)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={r['n']},d={r['d']})" for l, (_, r) in zip(labels, ok)])
        ax.set_ylabel("RelMSE (downstream prediction, avg over 4 models)")
        ax.set_title("E4: downstream-ML utility of synthetic data (this study vs paper Table 3, eps=1)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "e4_downstream_ml.png", dpi=140)
        print("\nsaved figures/e4_downstream_ml.png")

    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
