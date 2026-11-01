"""B7: what C3's hidden state linearly encodes — one ridge decoder per variable from all 128 units.

    python analysis/b7_decoders.py

Study 50's method (5-fold CV grouped by session; unique R² for history lags beyond every
shorter lag; location from the activation pattern Σw, Haufe et al. 2014) on the timestep
sequence, so ignored trials are steps. Encodings: choice k back = +1 right, −1 left, 0
ignored; reward k back = 1 if rewarded (0 when ignored); ignored k back = 1 / 0. Added
engagement variables: trials since the last ignore, ignore rate over the last 10 steps,
trial position. Row t's state has seen steps up to t−1, so "1 back" is step t−1.
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
import _pk02 as s13  # noqa: E402

N_TRIALS = 400_000
N_LAGS = 5
ALPHA = 1.0
N_PCS_SHOWN = 10
RECENT = 10
HISTORY_KINDS = ("choice", "reward", "choice×reward", "ignored")


def targets(d: pd.DataFrame) -> pd.DataFrame:
    ses = d["ses_idx"]
    ignored = (d["action"] == s13.IGNORE).astype(float)
    choice = np.where(d["action"] == s13.IGNORE, 0.0, 2.0 * d["action"] - 1.0)
    choice = pd.Series(choice, index=d.index)
    reward = ((d["reward"] > 0) & (d["action"] != s13.IGNORE)).astype(float)
    out = {}
    for k in range(1, N_LAGS + 1):
        c_k = choice.groupby(ses, sort=False).shift(k)
        r_k = reward.groupby(ses, sort=False).shift(k)
        out[f"choice {k} back"] = c_k
        out[f"reward {k} back"] = r_k
        out[f"choice×reward {k} back"] = c_k * r_k
        out[f"ignored {k} back"] = ignored.groupby(ses, sort=False).shift(k)
    out["P(rew|R) − P(rew|L)"] = d["reward_probabilityR"] - d["reward_probabilityL"]
    out["P(rew|R) + P(rew|L)"] = d["reward_probabilityR"] + d["reward_probabilityL"]
    position = d.groupby("ses_idx", sort=False).cumcount()
    for side, name in ((0, "left"), (1, "right"), (s13.IGNORE, "ignore")):
        at = pd.Series(np.where(d["action"] == side, position, np.nan), index=d.index)
        last = at.groupby(ses, sort=False).transform(lambda s: s.shift(1).ffill())
        label = f"trials since {name} chosen" if side != s13.IGNORE else "trials since last ignore"
        out[f"{label} (≤{s13.CAP})"] = np.clip(position - last, None, s13.CAP)
    out[f"ignore rate, last {RECENT} steps"] = ignored.groupby(ses, sort=False).transform(
        lambda s: s.shift(1).rolling(RECENT, min_periods=RECENT).mean())
    out["trial position"] = position.astype(float)
    out["model logit(R | engaged) (check)"] = d["logit_diff"].astype(float)
    out["model logit(engaged / ignore) (check)"] = d["engage_logit"]
    return pd.DataFrame(out, index=d.index)


def residualise(y: np.ndarray, others: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), others])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def cv_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pred = np.empty_like(y)
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        pred[test] = Ridge(alpha=ALPHA).fit(x[train], y[train]).predict(x[test])
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def lag_of(name: str) -> int | None:
    parts = name.split()
    return int(parts[-2]) if parts[-1] == "back" else None


def main() -> None:
    s13.apply_presentation_style()
    d = s13.load_trials(["ses_idx", "action", "reward", "logit_diff", "choice_logit_0", "choice_logit_1",
                         "choice_logit_2", "reward_probabilityL", "reward_probabilityR"])
    y_all = targets(d)
    sub = s13.subsample_rows(len(d), N_TRIALS, seed=1)
    pca = s13.load_pca()
    states = s13.load_trials(s13.STATE_COLUMNS).to_numpy(dtype=np.float32)[sub].astype(np.float64)
    x = states - pca["mean"]
    groups = d["ses_idx"].to_numpy()[sub]
    components = pca["components"]
    eigvals = np.clip(pca["eigvals"], 0, None)
    history = [c for c in y_all.columns if lag_of(c) is not None]

    rows, patterns = [], {}
    for name in y_all.columns:
        y = y_all[name].to_numpy(dtype=float)[sub]
        ok = np.isfinite(y)
        lag = lag_of(name)
        lower = [c for c in history if lag and lag_of(c) < lag]
        if lower:
            ok &= np.isfinite(y_all[lower].to_numpy(dtype=float)[sub]).all(axis=1)
        r2 = cv_r2(x[ok], y[ok], groups[ok])
        unique = (cv_r2(x[ok], residualise(y[ok], y_all[lower].to_numpy(dtype=float)[sub][ok]), groups[ok])
                  if lower else None)
        w = Ridge(alpha=ALPHA).fit(x[ok], y[ok]).coef_
        per_pc = eigvals * (components @ w) ** 2
        share = per_pc / per_pc.sum()
        pattern = eigvals * (components @ w)
        patterns[name] = pattern / np.linalg.norm(pattern)
        rows.append({"variable": name, "cv_r2": r2, "unique_cv_r2": unique, "n": int(ok.sum()),
                     "controlled_for": lower,
                     "signal_share_by_pc": share[:N_PCS_SHOWN].round(4).tolist(),
                     "signal_share_rest": float(share[N_PCS_SHOWN:].sum()),
                     "signal_share_top4_pcs": float(share[:4].sum())})
    names = list(patterns)
    cosine = np.array([[patterns[a] @ patterns[b] for b in names] for a in names])
    target_corr = y_all.iloc[sub].corr().loc[names, names].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(26, 13), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})
    axes[0].barh(range(len(names)), [r["cv_r2"] for r in rows], color="#4C72B0", label="R²")
    unique = [r["unique_cv_r2"] if r["unique_cv_r2"] is not None else np.nan for r in rows]
    axes[0].barh(range(len(names)), unique, color="#DD8452", height=0.45, label="unique R² (beyond lags < k)")
    axes[0].legend(fontsize=10, frameon=False, loc="lower right")
    axes[0].set_yticks(range(len(names)), names, fontsize=10)
    # Same row extent as the heatmap next to it, so each bar lines up with its row there.
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("cross-validated R²")
    axes[0].set_title("Decodable from C3's hidden state?", fontsize=13)
    shares = np.array([r["signal_share_by_pc"] + [r["signal_share_rest"]] for r in rows])
    im = axes[1].imshow(shares, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(N_PCS_SHOWN + 1), [f"PC{i + 1}" for i in range(N_PCS_SHOWN)] + ["rest"], fontsize=10)
    axes[1].set_yticks(range(len(names)), [""] * len(names))
    axes[1].set_title("Share of the decoded signal carried by each PC", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    im = axes[2].imshow(cosine, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    axes[2].set_yticks(range(len(names)), [""] * len(names))
    axes[2].set_title("Cosine similarity between activation patterns", fontsize=13)
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.tight_layout()
    axes[0].set_position([axes[0].get_position().x0, axes[1].get_position().y0,
                          axes[0].get_position().width, axes[1].get_position().height])
    fig.savefig(s13.figure_path("fig_B7_decoders.png"))
    plt.close(fig)

    s13.write_json("b7_decoders.json", {
        "n_timesteps": int(len(sub)), "subsample_seed": 1, "ridge_alpha": ALPHA,
        "cv": "5-fold GroupKFold by session", "states": "centred by the B3 PCA mean, raw units",
        "encoding": "choice +1 R / −1 L / 0 ignored; reward 0 when ignored; steps include ignored trials",
        "location": "activation pattern Σw (Haufe et al. 2014); per-PC share of decoded-signal variance",
        "decoders": rows, "variables": names,
        "pattern_cosine_similarity": cosine.round(4), "target_correlation": np.round(target_corr, 4),
    }, "analysis/b7_decoders.py")


if __name__ == "__main__":
    main()
