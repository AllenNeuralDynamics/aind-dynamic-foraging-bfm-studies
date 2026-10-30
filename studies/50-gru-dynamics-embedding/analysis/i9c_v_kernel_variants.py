"""I9c: constrained kernel variants of v — lag 1 absorbed by cluster offsets, and two outcome traces.

    python analysis/i9c_v_kernel_variants.py

Same target and history as I9 (v, and Hattori / Bari ΔQ for contrast; c_k = ±1, r_k ∈ {0, 1},
k = 1…K). Two families of descriptions, each with and without "cluster offsets":
  choice + reward   x = Σ_k A λ^(k−k0) c_k + B μ^(k−k0) c_k r_k          (I9's two-rate)
  two outcomes      x = Σ_k A_rew λ_rew^(k−k0) c_k r_k + A_unr λ_unr^(k−k0) c_k (1 − r_k)
  free              one weight per lag and outcome
With cluster offsets, the last trial (lag 1) enters only through an offset for each of its four
choice × outcome combinations, and the kernels run over lags 2…K (k0 = 2). This removes lag 1,
which v's construction partly partialled out (I2), and scores what the older history explains:
"within-cluster R²" = 1 − residual variance / variance around the cluster means. Without
offsets k0 = 1 and lag 1 is part of the kernel. All fits: 80/20 split by session (pooled), odd
sessions held out (per mouse, ≥ MIN_ROWS trials). Writes ``i9c_v_kernel_variants.json``,
``_cache/i9c_mouse_variants_<key>.parquet`` and ``fig_I9c_v_kernel_variants.png``.
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

N_FIT = 600_000
SEED = 20
MIN_ROWS = 1500
AXIS_NAMES = ["determinism / stickiness", "side bias", "slow learning / long choice memory", "dynamics-only"]
MODELS = ["free", "choice + reward", "two outcomes"]


def design(C: np.ndarray, R: np.ndarray, offsets: bool):
    """Base columns (intercept or 4 cluster dummies) and the lag block used by the kernels."""
    if offsets:
        cl = (2 * (C[:, 0] > 0) + R[:, 0]).astype(int)
        base = np.eye(4)[cl]
        return base, C[:, 1:], R[:, 1:], cl
    return np.ones((len(C), 1)), C, R, np.zeros(len(C), int)


def fit(y: np.ndarray, C: np.ndarray, R: np.ndarray, test: np.ndarray, offsets: bool) -> dict:
    base, Cl, Rl, cl = design(C, R, offsets)
    nl = Cl.shape[1]
    lags = np.arange(nl)
    tr_ = ~test
    # Variance around cluster means (test rows), for within-cluster R².
    cm = pd.Series(y[tr_]).groupby(cl[tr_]).mean()
    resid_base = y[test] - cm.reindex(cl[test]).fillna(y[tr_].mean()).to_numpy()
    denom = np.mean(resid_base ** 2) if offsets else np.var(y[test])
    score = lambda p: float(1 - np.mean((y[test] - p) ** 2) / denom)  # noqa: E731
    out = {}
    X = np.column_stack([base, Cl, Cl * Rl])
    coef, *_ = np.linalg.lstsq(X[tr_], y[tr_], rcond=None)
    nb = base.shape[1]
    unr, rew = coef[nb:nb + nl], coef[nb:nb + nl] + coef[nb + nl:]
    out["free"] = {"r2": score(X[test] @ coef), "rew": rew, "unr": unr}

    def solve(model, p0, lo, hi):
        def pred(p, rows):
            b = base[rows] @ p[:nb]
            q = p[nb:]
            if model == "choice + reward":
                return b + q[0] * (Cl[rows] @ q[1] ** lags) + q[2] * ((Cl[rows] * Rl[rows]) @ q[3] ** lags)
            return b + q[0] * ((Cl[rows] * Rl[rows]) @ q[1] ** lags) + q[2] * ((Cl[rows] * (1 - Rl[rows])) @ q[3] ** lags)
        tri, tei = np.flatnonzero(tr_), np.flatnonzero(test)
        p = least_squares(lambda p: pred(p, tri) - y[tri], np.r_[np.zeros(nb), p0],
                          bounds=(np.r_[np.full(nb, -np.inf), lo], np.r_[np.full(nb, np.inf), hi])).x
        return p[nb:], score(pred(p, tei))
    sd = np.std(y)
    q, r2 = solve("choice + reward", [sd / 3, 0.75, sd / 5, 0.5], [-np.inf, 0, -np.inf, 0], [np.inf, 0.999, np.inf, 0.999])
    out["choice + reward"] = {"r2": r2, "A": q[0], "lambda": q[1], "B": q[2], "mu": q[3],
                              "rew": q[0] * q[1] ** lags + q[2] * q[3] ** lags, "unr": q[0] * q[1] ** lags}
    q, r2 = solve("two outcomes", [sd / 3, 0.7, sd / 4, 0.75], [-np.inf, 0, -np.inf, 0], [np.inf, 0.999, np.inf, 0.999])
    out["two outcomes"] = {"r2": r2, "A_rew": q[0], "lambda_rew": q[1], "A_unr": q[2], "lambda_unr": q[3],
                           "rew": q[0] * q[1] ** lags, "unr": q[2] * q[3] ** lags}
    return out


def flat(res: dict) -> dict:
    out = {}
    for m, d in res.items():
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                out[f"{m} | {k}"] = float(v)
    return out


def mouse_job(y, C, R, test):
    if len(y) < MIN_ROWS or test.sum() < 200 or (~test).sum() < 500:
        return None
    try:
        return {f"{'offsets' if o else 'no offsets'} | {k}": v for o in (False, True) for k, v in flat(fit(y, C, R, test, o)).items()}
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "action", "reward", "subject_session_index"])
    rows, C, R = history(t)
    ys = {"GRU v": value_coordinate()[rows]}
    for name, m in (("Hattori ΔQ", "hattori"), ("Bari ΔQ", "bari")):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["q_rl_left", "q_rl_right"])
        ys[name] = (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy()[rows]
    ses = t["ses_idx"].to_numpy()[rows]
    subj = t["subject_id"].to_numpy()[rows]

    # Pooled fits.
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(rows), size=min(N_FIT, len(rows)), replace=False))
    uniq = np.unique(ses[sub])
    test = np.isin(ses[sub], rng.choice(uniq, size=len(uniq) // 5, replace=False))
    pooled = {name: {o: fit(y[sub].astype(np.float64), C[sub], R[sub], test, o == "offsets") for o in ("no offsets", "offsets")}
              for name, y in ys.items()}
    for name, d in pooled.items():
        for o, res in d.items():
            print(name, o, {m: round(v["r2"], 4) for m, v in res.items()},
                  {k: round(v, 3) for k, v in res["two outcomes"].items() if k not in ("rew", "unr", "r2")},
                  {k: round(v, 3) for k, v in res["choice + reward"].items() if k not in ("rew", "unr", "r2")}, flush=True)

    # Per mouse (odd sessions held out).
    order = t.drop_duplicates("ses_idx").sort_values(["subject_id", "subject_session_index"])
    odd = pd.Series(order.groupby("subject_id").cumcount().to_numpy() % 2 == 1, index=order["ses_idx"].to_numpy())
    is_odd = pd.Series(ses).map(odd).to_numpy()
    by_mouse = pd.Series(np.arange(len(rows))).groupby(subj).indices
    jobs = [(m, name, ix) for m, ix in by_mouse.items() for name in ys]
    res = Parallel(n_jobs=24, batch_size=8)(delayed(mouse_job)(ys[name][ix].astype(np.float64), C[ix], R[ix], is_odd[ix])
                                            for _, name, ix in jobs)
    df = pd.DataFrame([{"subject_id": m, "target": name, **r} for (m, name, _), r in zip(jobs, res) if r])
    df.to_parquet(s12.CACHE / f"i9c_mouse_variants_{s12.MODEL_KEY}.parquet")
    per_mouse = {}
    for name in TARGETS:
        a = df.query("target == @name")
        d = {}
        for o in ("no offsets", "offsets"):
            d[o] = {m: float(a[f"{o} | {m} | r2"].median()) for m in MODELS}
            d[o]["share_two_outcomes_beats_choice_reward"] = float((a[f"{o} | two outcomes | r2"] > a[f"{o} | choice + reward | r2"]).mean())
            d[o]["lambda_rew_median"] = float(a[f"{o} | two outcomes | lambda_rew"].median())
            d[o]["lambda_unr_median"] = float(a[f"{o} | two outcomes | lambda_unr"].median())
            d[o]["share_lambda_unr_gt_rew"] = float((a[f"{o} | two outcomes | lambda_unr"] > a[f"{o} | two outcomes | lambda_rew"]).mean())
            d[o]["share_mu_below_lambda"] = float((a[f"{o} | choice + reward | mu"] < a[f"{o} | choice + reward | lambda"]).mean())
        per_mouse[name] = d
        print(name, per_mouse[name], flush=True)

    axes = pd.read_parquet(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.parquet")
    axes.columns = AXIS_NAMES
    v = df.query("target == 'GRU v'").set_index("subject_id")
    mice = v.index.intersection(axes.index)
    keys = ["offsets | two outcomes | lambda_rew", "offsets | two outcomes | lambda_unr", "offsets | two outcomes | A_rew",
            "offsets | two outcomes | A_unr", "offsets | choice + reward | lambda", "offsets | choice + reward | mu"]
    corr = pd.DataFrame({ax: [spearmanr(v.loc[mice, k], axes.loc[mice, ax], nan_policy="omit")[0] for k in keys] for ax in AXIS_NAMES},
                        index=[k.replace("offsets | ", "") for k in keys])
    print(corr.round(2).to_string())

    fig, axs = plt.subplots(2, 4, figsize=(32, 13))
    lags_off, lags_all = np.arange(2, K + 1), np.arange(1, K + 1)
    for k, (name, color) in enumerate(TARGETS.items()):
        ax = axs[0, k]
        for o, lags, alpha in (("no offsets", lags_all, 0.45), ("offsets", lags_off, 1.0)):
            fr = pooled[name][o]["free"]
            ax.plot(lags, fr["rew"], color=color, lw=2.4, alpha=alpha, marker="o", label=f"rewarded, {o}")
            ax.plot(lags, fr["unr"], color=color, lw=2.4, alpha=alpha, marker="s", ls="--", label=f"unrewarded, {o}")
        to = pooled[name]["offsets"]["two outcomes"]
        ax.plot(lags_off, to["rew"], color="k", lw=1.2, ls=":", label="two-outcome fit (offsets)")
        ax.plot(lags_off, to["unr"], color="k", lw=1.2, ls=":")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("trials back")
        ax.set_ylabel("weight")
        ax.set_title(f"{name}: free kernels, with vs without lag-1 cluster offsets", fontsize=12)
        ax.legend(frameon=False, fontsize=8.5)
    ax = axs[0, 3]
    labels = [f"{o}\n{m}" for o in ("no offsets", "offsets") for m in MODELS]
    x = np.arange(len(labels))
    for j, (name, color) in enumerate(TARGETS.items()):
        vals = [pooled[name][o][m]["r2"] for o in ("no offsets", "offsets") for m in MODELS]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=color, label=name)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("held-out R² (with offsets: within-cluster R²)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Pooled fits: which description suffices?", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 0]
    for name, color in TARGETS.items():
        a = df.query("target == @name")
        ax.scatter(a["offsets | two outcomes | lambda_rew"], a["offsets | two outcomes | lambda_unr"], s=7, alpha=0.4, color=color,
                   label=f"{name}: λ_unrew > λ_rew in {100 * per_mouse[name]['offsets']['share_lambda_unr_gt_rew']:.0f}%")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":")
    ax.set_xlabel("rewarded-choice decay λ_rew (two outcomes, offsets)")
    ax.set_ylabel("unrewarded-choice decay λ_unrew")
    ax.set_title("Per mouse: do unrewarded choices persist longer?", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 1]
    for name, color in TARGETS.items():
        a = df.query("target == @name")
        ax.scatter(a["offsets | choice + reward | r2"], a["offsets | two outcomes | r2"], s=7, alpha=0.4, color=color,
                   label=f"{name}: two outcomes better in {100 * per_mouse[name]['offsets']['share_two_outcomes_beats_choice_reward']:.0f}%")
    lo = df[["offsets | choice + reward | r2", "offsets | two outcomes | r2"]].quantile(0.02).min()
    ax.plot([lo, 1], [lo, 1], color="0.5", ls=":")
    ax.set_xlabel("within-cluster R², choice + reward")
    ax.set_ylabel("within-cluster R², two outcomes")
    ax.set_title("Per mouse: which family fits better (held-out)?", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 2]
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(AXIS_NAMES)), AXIS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=10)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("GRU v (offsets): parameters × named embedding axes", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axs[1, 3]
    a = df.query("target == 'GRU v'")
    for o, color in (("no offsets", "#999999"), ("offsets", "#222222")):
        for j, m in enumerate(MODELS):
            vals = a[f"{o} | {m} | r2"].dropna()
            ax.boxplot(vals, positions=[j + (0.2 if o == "offsets" else -0.2)], widths=0.35, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": color, "alpha": 0.5})
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_ylabel("held-out R² per mouse (grey: no offsets; black: within-cluster, offsets)")
    ax.set_title("GRU v per mouse", fontsize=12)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I9c_v_kernel_variants.png"))
    plt.close(fig)

    s12.write_json("i9c_v_kernel_variants.json", {
        "K": K, "models": MODELS, "pooled": pooled, "per_mouse": per_mouse,
        "spearman_v_params_vs_named_axes": corr.to_dict(), "n_pooled": int(len(sub)), "min_rows": MIN_ROWS,
    }, "analysis/i9c_v_kernel_variants.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
