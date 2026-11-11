"""B7: what the hidden state linearly encodes — one ridge decoder per variable from all 128 units.

    python analysis/b7_decoders.py

For each variable: 5-fold cross-validated R² (folds grouped by session, so no session is
in both train and test); for history lags also a *unique* R² — the lag-k target after
regressing out every lag < k, since choices persist and lag-5 is partly lag-1 again.

Where a variable lives is read from the decoder's activation pattern a = Σw (Haufe et al.
2014), not from w: ridge weights load most on low-variance directions, so w points away
from the PCs that carry the signal. Per PC, the share of the decoded signal's variance is
λ_k (w·v_k)² / Σ λ_j (w·v_j)². Cosine similarity between variables is between patterns.
The target correlation matrix is reported alongside — patterns can align because targets do.

Timing: row t's state has seen trials up to t−1, so "1 back" is trial t−1.
Writes ``fig_B7_decoders.png`` and ``b7_decoders.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_TRIALS = 400_000
N_LAGS = 5
ALPHA = 1.0
N_PCS_SHOWN = 10


def targets(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("ses_idx", sort=False)
    choice = 2.0 * d["action"] - 1.0  # right +1, left −1
    reward = (d["reward"] > 0).astype(float)
    out = {}
    for k in range(1, N_LAGS + 1):
        c_k = choice.groupby(d["ses_idx"], sort=False).shift(k)
        r_k = reward.groupby(d["ses_idx"], sort=False).shift(k)
        out[f"choice {k} back"] = c_k
        out[f"reward {k} back"] = r_k
        out[f"choice×reward {k} back"] = c_k * r_k
    out["P(rew|R) − P(rew|L)"] = d["reward_probabilityR"] - d["reward_probabilityL"]
    out["P(rew|R) + P(rew|L)"] = d["reward_probabilityR"] + d["reward_probabilityL"]
    position = g.cumcount()
    for side, name in ((0, "left"), (1, "right")):
        chosen_at = pd.Series(np.where(d["action"] == side, position, np.nan), index=d.index)
        last = chosen_at.groupby(d["ses_idx"], sort=False).transform(lambda s: s.shift(1).ffill())
        out[f"trials since {name} chosen (≤{s12.CAP})"] = np.clip(position - last, None, s12.CAP)
    out["model logit(right) (check)"] = d["logit_diff"]
    return pd.DataFrame(out, index=d.index)


def residualise(y: np.ndarray, others: np.ndarray) -> np.ndarray:
    """y minus its least-squares fit (with intercept) on ``others``."""
    design = np.column_stack([np.ones(len(y)), others])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def cv_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pred = np.empty_like(y)
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        pred[test] = Ridge(alpha=ALPHA).fit(x[train], y[train]).predict(x[test])
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    s12.apply_presentation_style()
    d = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "logit_diff",
                         "reward_probabilityL", "reward_probabilityR"])
    y_all = targets(d)
    sub = s12.subsample_rows(len(d), N_TRIALS, seed=1)
    pca = s12.load_pca()
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)[sub].astype(np.float64)
    x = states - pca["mean"]
    groups = d["ses_idx"].to_numpy()[sub]
    components = pca["components"]

    eigvals = np.clip(pca["eigvals"], 0, None)
    history = [c for c in y_all.columns if " back" in c]
    rows, patterns = [], {}
    for name in y_all.columns:
        y = y_all[name].to_numpy(dtype=float)[sub]
        ok = np.isfinite(y)
        lag = int(name.split()[-2]) if name in history else None
        lower = [c for c in history if int(c.split()[-2]) < lag] if lag else []
        if lower:
            ok &= np.isfinite(y_all[lower].to_numpy(dtype=float)[sub]).all(axis=1)
        r2 = cv_r2(x[ok], y[ok], groups[ok])
        unique = (cv_r2(x[ok], residualise(y[ok], y_all[lower].to_numpy(dtype=float)[sub][ok]), groups[ok])
                  if lower else None)
        w = Ridge(alpha=ALPHA).fit(x[ok], y[ok]).coef_
        per_pc = eigvals * (components @ w) ** 2  # decoded-signal variance carried by each PC
        share = per_pc / per_pc.sum()
        pattern = eigvals * (components @ w)  # Σw in PC coordinates
        patterns[name] = pattern / np.linalg.norm(pattern)
        rows.append({"variable": name, "cv_r2": r2, "unique_cv_r2": unique, "n": int(ok.sum()),
                     "controlled_for": lower,
                     "signal_share_by_pc": share[:N_PCS_SHOWN].round(4).tolist(),
                     "signal_share_rest": float(share[N_PCS_SHOWN:].sum()),
                     "signal_share_top4_pcs": float(share[:4].sum())})
    names = list(patterns)
    cosine = np.array([[patterns[a] @ patterns[b] for b in names] for a in names])
    target_corr = y_all.iloc[sub].corr().loc[names, names].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})
    r2 = [r["cv_r2"] for r in rows]
    axes[0].barh(range(len(names)), r2, color="#4C72B0", label="R²")
    unique = [r["unique_cv_r2"] if r["unique_cv_r2"] is not None else np.nan for r in rows]
    axes[0].barh(range(len(names)), unique, color="#DD8452", height=0.45, label="unique R² (beyond lags < k)")
    axes[0].legend(fontsize=10, frameon=False, loc="lower right")
    axes[0].set_yticks(range(len(names)), names, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("cross-validated R²")
    axes[0].set_title("Decodable from the hidden state?", fontsize=13)
    shares = np.array([r["signal_share_by_pc"] + [r["signal_share_rest"]] for r in rows])
    im = axes[1].imshow(shares, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(N_PCS_SHOWN + 1), [f"PC{i + 1}" for i in range(N_PCS_SHOWN)] + ["rest"], fontsize=10)
    axes[1].set_yticks(range(len(names)), [""] * len(names))
    axes[1].set_title("Share of the decoded signal carried by each PC", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    im = axes[2].imshow(cosine, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(names)), names, rotation=90, fontsize=9)
    axes[2].set_yticks(range(len(names)), [""] * len(names))
    axes[2].set_title("Cosine similarity between activation patterns", fontsize=13)
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B7_decoders.png"))
    plt.close(fig)

    s12.write_json("b7_decoders.json", {
        "n_trials": int(len(sub)), "subsample_seed": 1, "ridge_alpha": ALPHA,
        "cv": "5-fold GroupKFold by session", "states": "centred by the B3 PCA mean, raw units",
        "location": "activation pattern Σw (Haufe et al. 2014); per-PC share of decoded-signal variance",
        "decoders": rows, "variables": names,
        "pattern_cosine_similarity": cosine.round(4), "target_correlation": np.round(target_corr, 4),
    }, "analysis/b7_decoders.py")


if __name__ == "__main__":
    main()
