"""I9b: I9's history-kernel fits of v (and RL ΔQ) per mouse.

    python analysis/i9b_v_kernel_per_mouse.py

For every mouse, on its trials with ≥ K previous trials in the session, the three I9 fits
(free, separable, two-rate) of the GRU's v and of its own Hattori and Bari ΔQ. Per mouse:
  λ, κ (separable), λ, μ, B/A (two-rate), held-out R² (alternate sessions held out);
  early / late outcome ratio from the free fit: Σ unrewarded / Σ rewarded weights over lags
  1–3 and over lags 5–8 (sums are steadier than single-lag ratios); "ratio rises" = late > early.
Split-half reliability (odd vs even sessions; Spearman across mice, Spearman–Brown) for each
parameter. Relations across mice to F7's named embedding axes and the animals' own features.
Writes ``i9b_v_kernel_per_mouse.json``, ``_cache/i9b_mouse_kernels_<key>.parquet`` and
``fig_I9b_v_kernel_per_mouse.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402
from i9_v_kernel import K, TARGETS, history  # noqa: E402

MIN_ROWS = 1500
EARLY, LATE = slice(0, 3), slice(4, 8)
AXIS_NAMES = ["determinism / stickiness", "side bias", "slow learning / long choice memory", "dynamics-only"]


def fit_mouse(y: np.ndarray, C: np.ndarray, R: np.ndarray, test: np.ndarray) -> dict:
    lags = np.arange(K)
    out = {}
    if len(y) < MIN_ROWS or test.sum() < 200 or (~test).sum() < 500 or np.var(y) == 0:
        return out
    tr_ = ~test
    r2 = lambda yt, p: float(1 - np.mean((yt - p) ** 2) / np.var(yt))  # noqa: E731
    X = np.column_stack([np.ones(len(y)), C, C * R])
    coef, *_ = np.linalg.lstsq(X[tr_], y[tr_], rcond=None)
    unrew, rew = coef[1:K + 1], coef[1:K + 1] + coef[K + 1:]
    out.update({"free_r2": r2(y[test], X[test] @ coef), "rew": rew, "unrew": unrew,
                "ratio_early": float(unrew[EARLY].sum() / rew[EARLY].sum()) if rew[EARLY].sum() > 0 else np.nan,
                "ratio_late": float(unrew[LATE].sum() / rew[LATE].sum()) if rew[LATE].sum() > 0 else np.nan})

    def sep(p, C_, R_):
        return p[0] + p[1] * ((C_ * (1 + p[3] * R_)) @ p[2] ** lags)
    try:
        p = least_squares(lambda p: sep(p, C[tr_], R[tr_]) - y[tr_], [0.0, np.std(y) / 3, 0.75, 0.5],
                          bounds=([-np.inf, -np.inf, 0.0, -0.99], [np.inf, np.inf, 0.999, 50])).x
        out.update({"sep_r2": r2(y[test], sep(p, C[test], R[test])), "sep_lambda": p[2], "sep_kappa": p[3]})
    except Exception:  # noqa: BLE001
        pass

    def two(p, C_, R_):
        return p[0] + p[1] * (C_ @ p[2] ** lags) + p[3] * ((C_ * R_) @ p[4] ** lags)
    try:
        p = least_squares(lambda p: two(p, C[tr_], R[tr_]) - y[tr_], [0.0, np.std(y) / 3, 0.75, np.std(y) / 5, 0.5],
                          bounds=([-np.inf, -np.inf, 0.0, -np.inf, 0.0], [np.inf, np.inf, 0.999, np.inf, 0.999])).x
        out.update({"two_r2": r2(y[test], two(p, C[test], R[test])), "two_lambda": p[2], "two_mu": p[4],
                    "two_B_over_A": p[3] / p[1] if p[1] != 0 else np.nan})
    except Exception:  # noqa: BLE001
        pass
    return out


def reliability(a: pd.Series, b: pd.Series) -> float:
    both = a.dropna().index.intersection(b.dropna().index)
    r = spearmanr(a.loc[both], b.loc[both])[0]
    return float(2 * r / (1 + r)) if r > -1 else np.nan


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "action", "reward", "subject_session_index"])
    rows, C, R = history(t)
    ys = {"GRU v": value_coordinate()[rows]}
    for name, m in (("Hattori ΔQ", "hattori"), ("Bari ΔQ", "bari")):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["q_rl_left", "q_rl_right"])
        ys[name] = (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy()[rows]
    subj = t["subject_id"].to_numpy()[rows]
    order = t.drop_duplicates("ses_idx").sort_values(["subject_id", "subject_session_index"])
    half = pd.Series(order.groupby("subject_id").cumcount().to_numpy() % 2, index=order["ses_idx"].to_numpy())
    h = t["ses_idx"].map(half).to_numpy()[rows]
    by_mouse = pd.Series(np.arange(len(rows))).groupby(subj).indices

    jobs = []
    for m, ix in by_mouse.items():
        for name, y in ys.items():
            jobs.append((m, name, "all", ix, h[ix] == 1))            # held out: odd sessions
            for part in (0, 1):                                      # split halves, each with its own held-out third
                sub = ix[h[ix] == part]
                jobs.append((m, name, f"half{part}", sub, (np.arange(len(sub)) % 3) == 0))
    res = Parallel(n_jobs=24, batch_size=16)(delayed(fit_mouse)(ys[name][ix].astype(np.float64), C[ix], R[ix], test)
                                              for _, name, _, ix, test in jobs)
    recs = []
    for (m, name, part, _, _), r in zip(jobs, res):
        if r:
            recs.append({"subject_id": m, "target": name, "part": part, **{k: v for k, v in r.items() if k not in ("rew", "unrew")},
                         **{f"rew_{k + 1}": r["rew"][k] for k in range(K)}, **{f"unrew_{k + 1}": r["unrew"][k] for k in range(K)}})
    df = pd.DataFrame(recs)
    df.to_parquet(s12.CACHE / f"i9b_mouse_kernels_{s12.MODEL_KEY}.parquet")
    params = ["sep_lambda", "sep_kappa", "two_lambda", "two_mu", "ratio_early", "ratio_late"]
    summary = {}
    for name in TARGETS:
        a = df.query("target == @name and part == 'all'").set_index("subject_id")
        h0 = df.query("target == @name and part == 'half0'").set_index("subject_id")
        h1 = df.query("target == @name and part == 'half1'").set_index("subject_id")
        rises = (a["ratio_late"] > a["ratio_early"]).mean()
        summary[name] = {"n_mice": int(len(a)),
                         **{p: {"median": float(a[p].median()), "q10_q90": a[p].quantile([0.1, 0.9]).round(4).tolist(),
                                "reliability": reliability(h0[p], h1[p])} for p in params},
                         "share_ratio_rises": float(rises), "share_mu_below_lambda": float((a["two_mu"] < a["two_lambda"]).mean()),
                         "r2_median": {k: float(a[k].median()) for k in ("free_r2", "sep_r2", "two_r2")}}
        print(name, {k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                     for k, v in summary[name].items()}, flush=True)

    # Relations to the named embedding axes and the animals' own features (GRU v parameters).
    axes = pd.read_parquet(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.parquet")
    axes.columns = AXIS_NAMES
    feats = pd.read_parquet(s12.CACHE / f"f7_mouse_features_{s12.MODEL_KEY}.parquet").query("part == 'all' and source == 'animal'").set_index("subject_id")
    v = df.query("target == 'GRU v' and part == 'all'").set_index("subject_id")
    vparams = ["sep_lambda", "sep_kappa", "two_lambda", "two_mu", "ratio_early", "ratio_late"]
    animal_cols = ["determinism", "stickiness ratio", "choice memory span", "reward memory span", "win-stay", "lose-stay", "switch rate"]
    mice = v.index.intersection(axes.index).intersection(feats.index)
    corr_axes = pd.DataFrame({ax: [spearmanr(v.loc[mice, p], axes.loc[mice, ax], nan_policy="omit")[0] for p in vparams] for ax in AXIS_NAMES}, index=vparams)
    corr_animal = pd.DataFrame({f: [spearmanr(v.loc[mice, p], feats.loc[mice, f], nan_policy="omit")[0] for p in vparams] for f in animal_cols}, index=vparams)
    print(corr_axes.round(2).to_string())
    print(corr_animal.round(2).to_string())

    fig, axs = plt.subplots(2, 4, figsize=(32, 13))
    lags = np.arange(1, K + 1)
    for k, (name, color) in enumerate(TARGETS.items()):
        ax = axs[0, k]
        a = df.query("target == @name and part == 'all'")
        for kind, ls in (("rew", "-"), ("unrew", "--")):
            mat = a[[f"{kind}_{j}" for j in lags]].to_numpy()
            norm = a[["rew_1"]].to_numpy()
            mat = mat / np.where(np.abs(norm) > 1e-9, norm, np.nan)
            q = np.nanpercentile(mat, [25, 50, 75], axis=0)
            ax.fill_between(lags, q[0], q[2], color=color, alpha=0.15)
            ax.plot(lags, q[1], color=color, ls=ls, lw=2.4, marker="o", label=f"{'rewarded' if kind == 'rew' else 'unrewarded'} (median, IQR)")
        pooled = s12.ANALYSIS / "i9_v_kernel.json"
        import json  # noqa: PLC0415
        fr = json.loads(pooled.read_text())["fits"][name]["free"]
        ax.plot(lags, np.array(fr["rewarded"]) / fr["rewarded"][0], color="0.5", lw=1.2, ls="-", label="pooled fit (I9)")
        ax.plot(lags, np.array(fr["unrewarded"]) / fr["rewarded"][0], color="0.5", lw=1.2, ls="--")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("trials back")
        ax.set_ylabel("weight / the mouse's lag-1 rewarded weight")
        ax.set_title(f"{name}: per-mouse kernels ({len(a)} mice)", fontsize=13)
        ax.legend(frameon=False, fontsize=9)
    ax = axs[0, 3]
    for name, color in TARGETS.items():
        a = df.query("target == @name and part == 'all'")
        ax.scatter(a["ratio_early"], a["ratio_late"], s=7, alpha=0.4, color=color,
                   label=f"{name}: rises in {100 * summary[name]['share_ratio_rises']:.0f}% of mice")
    lim = [-0.2, 2.5]
    ax.plot(lim, lim, color="0.5", ls=":", lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("outcome ratio, lags 1–3 (Σ unrewarded / Σ rewarded)")
    ax.set_ylabel("outcome ratio, lags 5–8")
    ax.set_title("Does the ratio rise within each mouse? (above the diagonal = yes)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 0]
    for name, color in TARGETS.items():
        a = df.query("target == @name and part == 'all'")
        ax.scatter(a["two_lambda"], a["two_mu"], s=7, alpha=0.4, color=color,
                   label=f"{name}: μ < λ in {100 * summary[name]['share_mu_below_lambda']:.0f}%")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("choice-trace decay λ (two-rate)")
    ax.set_ylabel("reward-trace decay μ (two-rate)")
    ax.set_title("Choice vs reward memory per mouse (below the diagonal: reward fades faster)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    for ax, (mat, title) in zip(axs[1, 1:3], ((corr_axes, "GRU v kernel parameters × named embedding axes (Spearman)"),
                                             (corr_animal, "GRU v kernel parameters × the animals' own features"))):
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.iat[i, j]:+.2f}", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)
    ax = axs[1, 3]
    y = np.arange(len(params))
    for j, (name, color) in enumerate(TARGETS.items()):
        ax.barh(y + (j - 1) * 0.27, [summary[name][p]["reliability"] for p in params], 0.27, color=color, label=name)
    ax.set_yticks(y, params, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("split-half reliability across mice (Spearman–Brown)")
    ax.set_title("How reliable are the per-mouse parameters?", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I9b_v_kernel_per_mouse.png"))
    plt.close(fig)

    s12.write_json("i9b_v_kernel_per_mouse.json", {
        "K": K, "min_rows": MIN_ROWS, "early_lags": "1–3", "late_lags": "5–8", "summary": summary,
        "spearman_v_params_vs_named_axes": corr_axes.to_dict(), "spearman_v_params_vs_animal_features": corr_animal.to_dict(),
        "held_out": "all: odd sessions held out; halves: every third session-row block of each half held out",
    }, "analysis/i9b_v_kernel_per_mouse.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
