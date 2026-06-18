"""Shared helpers for the BinAgg reproduction experiments (Issue #16)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA = ROOT / "data"

# Simulation ground truth (shared by E1/E2). Bounds are set from the KNOWN
# generating distribution (public domain knowledge), not from sampled data.
TRUE_BETA = np.array([1.5, -2.0, 0.5])
X_LOW, X_HIGH = 0.0, 10.0
SIM_X_BOUNDS = [(X_LOW, X_HIGH)] * 3
SIM_Y_BOUNDS = (-30.0, 30.0)
MU_GRID = [0.5, 1.0, 2.0, 5.0]


def make_sim_data(n: int, seed: int):
    """X ~ Uniform(0,10)^3, y = X@beta + N(0,1)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_LOW, X_HIGH, size=(n, 3))
    y = X @ TRUE_BETA + rng.normal(0.0, 1.0, size=n)
    return X, y


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def rel_l2(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    return float(np.linalg.norm(beta_hat - beta_true) / np.linalg.norm(beta_true))


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
