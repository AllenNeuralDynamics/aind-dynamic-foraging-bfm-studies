"""I9: is v "a choice trace scaled by outcome"? Kernels and a direct test of the description.

    python analysis/i9_v_kernel.py

Target x(t) on every trial with ≥ K previous trials in its session: the GRU's value coordinate
v (I2 axis, state before choice t), and for contrast Hattori's and Bari's ΔQ = Q_R − Q_L (same
trials, per-mouse fits). History: c_k = ±1 (choice k trials back, R = +1), r_k = 1 if it was
rewarded, k = 1…K. Three descriptions, fitted on 80% of sessions, scored by held-out R²:
  free        x = β0 + Σ_k a_k c_k + b_k c_k r_k                       (2K + 1 parameters)
              → rewarded-choice kernel a_k + b_k, unrewarded-choice kernel a_k
  separable   x = β0 + A Σ_k λ^(k−1) c_k (1 + κ r_k)                   (one choice trace,
              outcome scales each choice by 1 + κ; 4 parameters)
  two-rate    x = β0 + A Σ_k λ^(k−1) c_k + B Σ_k μ^(k−1) c_k r_k        (choice trace and a reward
              trace with its own decay μ; 5 parameters)
"Separable ≈ free" means the description is sufficient; "two-rate > separable" means the reward
part has its own timescale. Kernels are shown normalised by each variable's lag-1 rewarded
weight. Writes ``i9_v_kernel.json`` and ``fig_I9_v_kernel.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

K = 10
N_FIT = 600_000
SEED = 17
TARGETS = {"GRU v": "#222222", "Hattori ΔQ": "#55A868", "Bari ΔQ": "#4C72B0"}


def history(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with K previous trials in the session; C[:, k−1] = c_k, R[:, k−1] = r_k."""
    g = t.groupby("ses_idx", sort=False)
    c = 2.0 * t["action"].to_numpy() - 1.0
    r = (t["reward"].to_numpy() > 0).astype(float)
    pos = g.cumcount().to_numpy()
    rows = np.flatnonzero(pos >= K)
    C = np.stack([c[rows - k] for k in range(1, K + 1)], axis=1)
    R = np.stack([r[rows - k] for k in range(1, K + 1)], axis=1)
    return rows, C, R


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.mean((y - pred) ** 2) / np.var(y))


def fit_all(y: np.ndarray, C: np.ndarray, R: np.ndarray, train: np.ndarray) -> dict:
    lags = np.arange(K)
    out = {}
    # Free.
    X = np.column_stack([np.ones(len(y)), C, C * R])
    coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
    out["free"] = {"heldout_r2": r2(y[~train], X[~train] @ coef), "n_params": 2 * K + 1,
                   "unrewarded": coef[1:K + 1], "rewarded": coef[1:K + 1] + coef[K + 1:]}
    # Separable: one trace, outcome scales.
    def sep(p, C_, R_):
        b0, A, lam, kap = p
        return b0 + A * ((C_ * (1 + kap * R_)) @ lam ** lags)
    p = least_squares(lambda p: sep(p, C[train], R[train]) - y[train], [0.0, 0.3, 0.75, 0.5],
                      bounds=([-np.inf, -np.inf, 0.0, -0.99], [np.inf, np.inf, 0.999, 50])).x
    out["separable"] = {"heldout_r2": r2(y[~train], sep(p, C[~train], R[~train])), "n_params": 4,
                        "params": dict(zip(["b0", "A", "lambda", "kappa"], p.tolist())),
                        "unrewarded": p[1] * p[2] ** lags, "rewarded": p[1] * (1 + p[3]) * p[2] ** lags}
    # Two-rate: choice trace + reward trace.
    def two(p, C_, R_):
        b0, A, lam, B, mu = p
        return b0 + A * (C_ @ lam ** lags) + B * ((C_ * R_) @ mu ** lags)
    p2 = least_squares(lambda p: two(p, C[train], R[train]) - y[train], [0.0, 0.3, 0.75, 0.2, 0.5],
                       bounds=([-np.inf, -np.inf, 0.0, -np.inf, 0.0], [np.inf, np.inf, 0.999, np.inf, 0.999])).x
    out["two-rate"] = {"heldout_r2": r2(y[~train], two(p2, C[~train], R[~train])), "n_params": 5,
                       "params": dict(zip(["b0", "A", "lambda", "B", "mu"], p2.tolist())),
                       "unrewarded": p2[1] * p2[2] ** lags, "rewarded": p2[1] * p2[2] ** lags + p2[3] * p2[4] ** lags}
    return out


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "action", "reward"])
    rows, C, R = history(t)
    ys = {"GRU v": value_coordinate()}
    for name, m in (("Hattori ΔQ", "hattori"), ("Bari ΔQ", "bari")):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["q_rl_left", "q_rl_right"])
        ys[name] = (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy()
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(rows), size=min(N_FIT, len(rows)), replace=False))
    rows, C, R = rows[sub], C[sub], R[sub]
    ses = t["ses_idx"].to_numpy()[rows]
    uniq = np.unique(ses)
    train = ~np.isin(ses, rng.choice(uniq, size=len(uniq) // 5, replace=False))
    fits = {name: fit_all(y[rows].astype(np.float64), C, R, train) for name, y in ys.items()}
    for name, f in fits.items():
        print(name, {m: round(v["heldout_r2"], 4) for m, v in f.items()}, f["separable"]["params"], f["two-rate"]["params"], flush=True)

    lags = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 4, figsize=(30, 7), gridspec_kw={"width_ratios": [1.2, 1, 1, 0.9]})
    ax = axes[0]
    f = fits["GRU v"]
    ax.plot(lags, f["free"]["rewarded"], color="#0072B2", marker="o", lw=2.5, label="rewarded choice (free fit)")
    ax.plot(lags, f["free"]["unrewarded"], color="#D55E00", marker="o", lw=2.5, label="unrewarded choice (free fit)")
    for model, ls in (("separable", "--"), ("two-rate", ":")):
        ax.plot(lags, f[model]["rewarded"], color="#0072B2", ls=ls, lw=1.8, label=f"rewarded ({model})")
        ax.plot(lags, f[model]["unrewarded"], color="#D55E00", ls=ls, lw=1.8, label=f"unrewarded ({model})")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("trials back (k)")
    ax.set_ylabel("weight on v (state units per ±1 choice)")
    ax.set_title("What v integrates: choice kernels by outcome (GRU)", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for name, color in TARGETS.items():
        fr = fits[name]["free"]
        ax.plot(lags, fr["rewarded"] / fr["rewarded"][0], color=color, marker="o", lw=2.2, label=f"{name}: rewarded")
        ax.plot(lags, fr["unrewarded"] / fr["rewarded"][0], color=color, marker="s", lw=1.6, ls="--", label=f"{name}: unrewarded")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("trials back (k)")
    ax.set_ylabel("weight / lag-1 rewarded weight")
    ax.set_title("Kernels, normalised (free fits)", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    for name, color in TARGETS.items():
        fr = fits[name]["free"]
        ax.plot(lags, fr["unrewarded"] / fr["rewarded"], color=color, marker="o", lw=2.2, label=name)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("trials back (k)")
    ax.set_ylabel("unrewarded / rewarded weight")
    ax.set_title("Is the outcome a fixed scale? (flat = separable)", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[3]
    models = ["separable", "two-rate", "free"]
    x = np.arange(len(TARGETS))
    for j, (m, shade) in enumerate(zip(models, ("#E69F00", "#4C72B0", "#999999"))):
        vals = [fits[n][m]["heldout_r2"] for n in TARGETS]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=shade, label=f"{m} ({fits['GRU v'][m]['n_params']} params)")
        for xi, v in zip(x + (j - 1) * 0.27, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, list(TARGETS))
    lo = min(fits[n][m]["heldout_r2"] for n in TARGETS for m in models)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.set_ylabel(f"held-out R² (history {K} back)")
    ax.set_title("Does the description suffice?", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I9_v_kernel.png"))
    plt.close(fig)

    s12.write_json("i9_v_kernel.json", {
        "K": K, "n_trials": int(len(rows)), "split": "80/20 by session", "encoding": "c_k = ±1 (R = +1), r_k ∈ {0, 1}",
        "fits": fits,
    }, "analysis/i9_v_kernel.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
