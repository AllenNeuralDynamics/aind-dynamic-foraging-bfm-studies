"""I6: the I4 transition analysis on the RL models — does Q-learning have the GRU's update structure?

    python analysis/i6_rl_transitions.py

Per transition group (input × stay/switch, ``_transitions``), fit x(t+1) = x* + λ(x(t) − x*)
for: the GRU's value coordinate (I2 axis) and the per-mouse RL fits' ΔQ = Q_R − Q_L (Hattori
L2F1, Bari L1F1_CK1), pooled over the 614 mice. ΔQ alone misses Bari's choice kernel and both
models' softmax β/bias, so the same fit is also run on every model's policy logit
log P(R)/P(L), which puts all three on one scale. Per group: slope λ, target x*, R², and the
median over mice of the per-mouse λ (pooled fits mix mice with different learning rates).
Q-learning with equal learning and forgetting rates predicts λ_stay = λ_switch and ΔQ
targets of ±1 (rewarded) and 0 (unrewarded); deviations measure what the GRU adds.
Writes ``i6_rl_transitions.json`` and ``fig_I6_rl_transitions.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
import _transitions as tr  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

RL = {"hattori": "Hattori (L2F1)", "bari": "Bari (L1F1_CK1)"}
N_BINS = 20
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
FIT_GROUPS = [g for g, name in enumerate(tr.GROUP_NAMES) if not name.endswith("(first)")]


def group_r2(x: np.ndarray, lay: dict, lam: np.ndarray, target: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(lay["has_next"])
    r2 = np.full(tr.N_GROUPS, np.nan)
    for g in FIT_GROUPS:
        sel = idx[lay["group"][idx] == g]
        pred = target[g] + lam[g] * (x[sel] - target[g])
        r2[g] = 1 - np.var(x[sel + 1] - pred) / np.var(x[sel + 1])
    return r2


def analyse(x: np.ndarray, lay: dict, by_mouse: dict) -> dict:
    lam, target, n = tr.fit_groups(x, lay)
    per_mouse = np.array([tr.fit_groups(x, lay, rows=rows)[0] for rows in by_mouse.values()])
    return {"lam": lam, "target": target, "n": n, "r2": group_r2(x, lay, lam, target),
            "lam_median_per_mouse": np.nanmedian(per_mouse, axis=0)}


def binned(ax, x: np.ndarray, lay: dict, fit: dict, title: str, xlabel: str) -> None:
    idx = np.flatnonzero(lay["has_next"])
    lo, hi = np.percentile(x, [1, 99])
    edges = np.linspace(lo, hi, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for g in FIT_GROUPS:
        name, kind = tr.GROUP_NAMES[g].split(" (")
        sel = idx[lay["group"][idx] == g]
        b = np.digitize(x[sel], edges) - 1
        ok = (b >= 0) & (b < N_BINS)
        means = pd.Series(x[sel + 1][ok]).groupby(b[ok]).agg(["mean", "size"])
        means = means[means["size"] >= 50]
        ls = "-" if kind == "stay)" else "--"
        ax.plot(centres[means.index], means["mean"], color=COLORS[name], ls=ls, marker="o", ms=3, lw=1.6,
                label=f"{tr.GROUP_NAMES[g]}: λ {fit['lam'][g]:.2f}, x* {fit['target'][g]:+.2f}, R² {fit['r2'][g]:.2f}")
    ax.plot([lo, hi], [lo, hi], color="0.6", ls=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("next trial")
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=7.5)


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "logit_diff"])
    lay = tr.session_layout(t)
    by_mouse = pd.Series(np.arange(len(t))).groupby(t["subject_id"].to_numpy()).indices
    variables = {"GRU": {"state": value_coordinate(), "logit": t["logit_diff"].to_numpy(dtype=np.float64)}}
    for name in RL:
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet")
        if not ((rl["ses_idx"].to_numpy() == t["ses_idx"].to_numpy()).all()
                and (rl["trial_idx"].to_numpy() == t["trial_idx"].to_numpy()).all()):
            raise AssertionError(f"{name} rollout rows are not aligned with the trial table")
        p = np.clip(rl["p_rl_right"].to_numpy(dtype=np.float64), 1e-10, 1 - 1e-10)
        variables[name] = {"state": (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy(dtype=np.float64),
                           "logit": np.log(p / (1 - p))}
    fits = {m: {k: analyse(x, lay, by_mouse) for k, x in d.items()} for m, d in variables.items()}

    titles = {"GRU": "GRU", **RL}
    state_label = {"GRU": "value coordinate v(t)", "hattori": "ΔQ(t) = Q_R − Q_L", "bari": "ΔQ(t) = Q_R − Q_L"}
    fig, axes = plt.subplots(2, 4, figsize=(32, 13), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for j, m in enumerate(fits):
        binned(axes[0, j], variables[m]["state"], lay, fits[m]["state"], f"{titles[m]}: carried variable", state_label[m])
        binned(axes[1, j], variables[m]["logit"], lay, fits[m]["logit"], f"{titles[m]}: policy logit", "logit P(R)/P(L) at t")
    labels = [tr.GROUP_NAMES[g] for g in FIT_GROUPS]
    y = np.arange(len(labels))
    width = 0.27
    for k, (m, color) in enumerate(zip(fits, ("#222222", "#C44E52", "#4C72B0"))):
        f = fits[m]["logit"]
        axes[0, 3].barh(y + (k - 1) * width, f["lam_median_per_mouse"][FIT_GROUPS], height=width, color=color, label=titles[m])
        scale = f["target"][tr.GROUP_NAMES.index("R rewarded (stay)")]
        axes[1, 3].barh(y + (k - 1) * width, f["target"][FIT_GROUPS] / scale, height=width, color=color, label=titles[m])
    for ax, xlabel, title in ((axes[0, 3], "λ (median of per-mouse fits)", "Retention per group (policy logit)"),
                              (axes[1, 3], "target / target of R rewarded (stay)", "Targets, normalised (policy logit)")):
        ax.set_yticks(y, labels, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I6_rl_transitions.png"))
    plt.close(fig)

    s12.write_json("i6_rl_transitions.json", {
        "groups": tr.GROUP_NAMES, "fit_groups": FIT_GROUPS, "models": {m: titles[m] for m in fits},
        "variables": {"state": "GRU: I2 value coordinate; RL: Q_R − Q_L", "logit": "log P(R)/P(L)"},
        "fits": fits, "min_per_group": tr.MIN_PER_GROUP,
    }, "analysis/i6_rl_transitions.py", extra_groups=s12.rl_groups())
    for m in fits:
        for k in ("state", "logit"):
            f = fits[m][k]
            print(m, k, {tr.GROUP_NAMES[g]: (round(f["lam"][g], 2), round(f["lam_median_per_mouse"][g], 2),
                                              round(f["target"][g], 2), round(f["r2"][g], 3)) for g in FIT_GROUPS})


if __name__ == "__main__":
    main()
