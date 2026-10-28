"""I9d: longer history (K = 20) and a three-trace model for v's kernel.

    python analysis/i9d_v_kernel_long.py

Follows I9c's "offsets" setup (lag 1 absorbed by four choice × outcome cluster offsets; kernels
over lags 2…K; within-cluster R²), with two changes:
  1. K = 20 instead of 10, so the last lag no longer collects all older history (I9c's lag-10
     rise). K = 10 is refitted on the same rows (sessions' trials ≥ 20) for a like-for-like test.
  2. A six-parameter "three traces" model:
       x = Σ_k A_rew λ_rew^(k−2) c_k r_k + A_unr λ_unr^(k−2) c_k (1 − r_k) + A_c λ_c^(k−2) c_k
     two outcome traces plus a shared choice trace. It nests both four-parameter families
     ("two outcomes": A_c = 0; "choice + reward": A_unr = 0, so the choice trace is A λ and the
     rewarded trace is the reward bonus B μ).
Scored by held-out within-cluster R² (pooled 80/20 by session; per mouse, odd sessions held out)
and per-mouse BIC on the training rows (n log(RSS/n) + p log n, p counting the 4 offsets).
Writes ``i9d_v_kernel_long.json``, ``_cache/i9d_mouse_long_<key>.parquet`` and
``fig_I9d_v_kernel_long.png``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402
from i9_v_kernel import TARGETS  # noqa: E402

K_LONG, K_SHORT = 20, 10
N_FIT = 600_000
SEED = 21
MIN_ROWS = 1500
MODELS = ["free", "choice + reward", "two outcomes", "three traces"]
N_PAR = {"choice + reward": 4, "two outcomes": 4, "three traces": 6}


def history(t: pd.DataFrame, k_max: int):
    """Rows with k_max previous trials in the session; C[:, k−1] = c_k, R[:, k−1] = r_k."""
    c = 2.0 * t["action"].to_numpy() - 1.0
    r = (t["reward"].to_numpy() > 0).astype(float)
    pos = t.groupby("ses_idx", sort=False).cumcount().to_numpy()
    rows = np.flatnonzero(pos >= k_max)
    C = np.stack([c[rows - k] for k in range(1, k_max + 1)], axis=1)
    R = np.stack([r[rows - k] for k in range(1, k_max + 1)], axis=1)
    return rows, C, R


def kernels(model: str, q: np.ndarray, lags: np.ndarray):
    if model == "choice + reward":
        return q[0] * q[1] ** lags + q[2] * q[3] ** lags, q[0] * q[1] ** lags
    if model == "two outcomes":
        return q[0] * q[1] ** lags, q[2] * q[3] ** lags
    ch = q[4] * q[5] ** lags
    return q[0] * q[1] ** lags + ch, q[2] * q[3] ** lags + ch


def fit(y: np.ndarray, C: np.ndarray, R: np.ndarray, test: np.ndarray, k: int) -> dict:
    """Offsets setup over lags 2…k. Returns held-out within-cluster R², train BIC and kernels."""
    cl = (2 * (C[:, 0] > 0) + R[:, 0]).astype(int)
    base = np.eye(4)[cl]
    Cl, Rl = C[:, 1:k], R[:, 1:k]
    lags = np.arange(k - 1)
    tr_ = ~test
    cm = pd.Series(y[tr_]).groupby(cl[tr_]).mean()
    denom = np.mean((y[test] - cm.reindex(cl[test]).fillna(y[tr_].mean()).to_numpy()) ** 2)
    n_tr = int(tr_.sum())
    score = lambda p: float(1 - np.mean((y[test] - p) ** 2) / denom)  # noqa: E731
    bic = lambda rss, p: float(n_tr * np.log(rss / n_tr) + (p + 4) * np.log(n_tr))  # noqa: E731
    out = {}
    X = np.column_stack([base, Cl, Cl * Rl])
    coef, *_ = np.linalg.lstsq(X[tr_], y[tr_], rcond=None)
    unr = coef[4:4 + len(lags)]
    out["free"] = {"r2": score(X[test] @ coef), "bic": bic(np.sum((X[tr_] @ coef - y[tr_]) ** 2), 2 * len(lags)),
                   "rew": unr + coef[4 + len(lags):], "unr": unr}
    CR, CU = Cl * Rl, Cl * (1 - Rl)

    def pred(model, p, rows):
        b, q = base[rows] @ p[:4], p[4:]
        if model == "choice + reward":
            return b + q[0] * (Cl[rows] @ q[1] ** lags) + q[2] * (CR[rows] @ q[3] ** lags)
        x = b + q[0] * (CR[rows] @ q[1] ** lags) + q[2] * (CU[rows] @ q[3] ** lags)
        return x + q[4] * (Cl[rows] @ q[5] ** lags) if model == "three traces" else x

    tri, tei = np.flatnonzero(tr_), np.flatnonzero(test)
    sd = np.std(y)
    starts = {"choice + reward": [[sd / 3, 0.75, sd / 5, 0.5]],
              "two outcomes": [[sd / 3, 0.7, sd / 4, 0.75]],
              # Several starts: the slow choice trace can sit above or below the outcome traces.
              "three traces": [[sd / 3, 0.6, sd / 5, 0.6, sd / 10, 0.9], [sd / 3, 0.7, sd / 4, 0.8, 0.0, 0.5],
                               [sd / 4, 0.5, sd / 6, 0.5, sd / 6, 0.85]]}
    for model, p0s in starts.items():
        npar = N_PAR[model]
        lo = np.r_[np.full(4, -np.inf), np.tile([-np.inf, 0.0], npar // 2)]
        hi = np.r_[np.full(4, np.inf), np.tile([np.inf, 0.999], npar // 2)]
        best = None
        for p0 in p0s:
            r = least_squares(lambda p: pred(model, p, tri) - y[tri], np.r_[cm.reindex(range(4)).fillna(0).to_numpy(), p0],
                              bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
        q = best.x[4:]
        rew, un = kernels(model, q, lags)
        out[model] = {"r2": score(pred(model, best.x, tei)), "bic": bic(2 * best.cost, npar), "params": q, "rew": rew, "unr": un}
    return out


def flat(res: dict) -> dict:
    out = {}
    for m, d in res.items():
        out[f"{m} | r2"], out[f"{m} | bic"] = d["r2"], d["bic"]
        for j, v in enumerate(d.get("params", [])):
            out[f"{m} | p{j}"] = float(v)
    return out


def mouse_job(y, C, R, test):
    if len(y) < MIN_ROWS or test.sum() < 200 or (~test).sum() < 500:
        return None
    try:
        return {f"K{k} | {key}": v for k in (K_SHORT, K_LONG) for key, v in flat(fit(y, C, R, test, k)).items()}
    except Exception:  # noqa: BLE001
        return None


def jsonable(res: dict) -> dict:
    return {m: {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for m, d in res.items()}


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "action", "reward", "subject_session_index"])
    rows, C, R = history(t, K_LONG)
    ys = {"GRU v": value_coordinate()[rows]}
    for name, m in (("Hattori ΔQ", "hattori"), ("Bari ΔQ", "bari")):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["q_rl_left", "q_rl_right"])
        ys[name] = (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy()[rows]
    ses = t["ses_idx"].to_numpy()[rows]
    subj = t["subject_id"].to_numpy()[rows]

    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(rows), size=min(N_FIT, len(rows)), replace=False))
    uniq = np.unique(ses[sub])
    test = np.isin(ses[sub], rng.choice(uniq, size=len(uniq) // 5, replace=False))
    pooled = {name: {f"K{k}": fit(y[sub].astype(np.float64), C[sub], R[sub], test, k) for k in (K_SHORT, K_LONG)}
              for name, y in ys.items()}
    for name, d in pooled.items():
        for k, res in d.items():
            print(name, k, {m: round(v["r2"], 4) for m, v in res.items()},
                  {m: np.round(v["params"], 3).tolist() for m, v in res.items() if "params" in v}, flush=True)

    order = t.drop_duplicates("ses_idx").sort_values(["subject_id", "subject_session_index"])
    odd = pd.Series(order.groupby("subject_id").cumcount().to_numpy() % 2 == 1, index=order["ses_idx"].to_numpy())
    is_odd = pd.Series(ses).map(odd).to_numpy()
    by_mouse = pd.Series(np.arange(len(rows))).groupby(subj).indices
    jobs = [(m, name, ix) for m, ix in by_mouse.items() for name in ys]
    res = Parallel(n_jobs=24, batch_size=8)(delayed(mouse_job)(ys[name][ix].astype(np.float64), C[ix], R[ix], is_odd[ix])
                                            for _, name, ix in jobs)
    df = pd.DataFrame([{"subject_id": m, "target": name, **r} for (m, name, _), r in zip(jobs, res) if r])
    df.to_parquet(s12.CACHE / f"i9d_mouse_long_{s12.MODEL_KEY}.parquet")

    per_mouse = {}
    for name in TARGETS:
        a = df.query("target == @name")
        d = {"n_mice": int(len(a))}
        for k in (f"K{K_SHORT}", f"K{K_LONG}"):
            e = {"r2_median": {m: float(a[f"{k} | {m} | r2"].median()) for m in MODELS}}
            bics = a[[f"{k} | {m} | bic" for m in MODELS]].to_numpy()
            e["bic_best_share"] = {m: float(np.mean(np.argmin(bics, 1) == j)) for j, m in enumerate(MODELS)}
            e["r2_best_share_constrained"] = {m: float(np.mean(np.argmax(a[[f"{k} | {mm} | r2" for mm in MODELS[1:]]].to_numpy(), 1) == j))
                                              for j, m in enumerate(MODELS[1:])}
            e["two_outcomes_beats_choice_reward"] = float((a[f"{k} | two outcomes | r2"] > a[f"{k} | choice + reward | r2"]).mean())
            e["lambda_rew_median"] = float(a[f"{k} | two outcomes | p1"].median())
            e["lambda_unr_median"] = float(a[f"{k} | two outcomes | p3"].median())
            e["share_lambda_unr_gt_rew"] = float((a[f"{k} | two outcomes | p3"] > a[f"{k} | two outcomes | p1"]).mean())
            e["three_traces_param_medians"] = {n: float(a[f"{k} | three traces | p{j}"].median())
                                               for j, n in enumerate(["A_rew", "lambda_rew", "A_unr", "lambda_unr", "A_c", "lambda_c"])}
            d[k] = e
        per_mouse[name] = d
        print(name, d, flush=True)

    # Figure: row 1 — pooled K = 20 kernels for each target (free vs fits); row 2 — model comparison.
    lags = np.arange(2, K_LONG + 1)
    fig, axs = plt.subplots(2, 4, figsize=(32, 13))
    styles = {"choice + reward": ("#CC79A7", "--"), "two outcomes": ("#000000", ":"), "three traces": ("#009E73", "-.")}
    for k, (name, color) in enumerate(TARGETS.items()):
        ax = axs[0, k]
        r20, r10 = pooled[name][f"K{K_LONG}"], pooled[name][f"K{K_SHORT}"]
        ax.plot(lags, r20["free"]["rew"], color=color, lw=2.6, marker="o", label="free, rewarded (K = 20)")
        ax.plot(lags, r20["free"]["unr"], color=color, lw=2.6, marker="s", ls="--", label="free, unrewarded (K = 20)")
        ax.plot(np.arange(2, K_SHORT + 1), r10["free"]["unr"], color=color, lw=1.2, alpha=0.5, marker="s", ls="--",
                label="free, unrewarded (K = 10, same rows)")
        for m, (c_, ls) in styles.items():
            ax.plot(lags, r20[m]["rew"], color=c_, lw=1.4, ls=ls, label=f"{m} (R² {r20[m]['r2']:.4f})")
            ax.plot(lags, r20[m]["unr"], color=c_, lw=1.4, ls=ls)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("trials back")
        ax.set_ylabel("weight")
        ax.set_title(f"{name}: kernels over lags 2–20 (lag 1 in offsets)", fontsize=12)
        ax.legend(frameon=False, fontsize=8.5)
    ax = axs[0, 3]
    a = pooled["GRU v"][f"K{K_LONG}"]
    for m, (c_, ls) in styles.items():
        rew, unr = a[m]["rew"], a[m]["unr"]
        ax.plot(lags, rew - a["free"]["rew"], color=c_, ls=ls, lw=2, marker="o", ms=4, label=f"{m}: rewarded")
        ax.plot(lags, unr - a["free"]["unr"], color=c_, ls=ls, lw=2, marker="s", ms=4, alpha=0.6, label=f"{m}: unrewarded")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("trials back")
    ax.set_ylabel("fitted − free weight")
    ax.set_title("GRU v (K = 20): where each constrained model misses the free kernel", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    ax = axs[1, 0]
    x = np.arange(len(MODELS))
    for j, (name, color) in enumerate(TARGETS.items()):
        base = pooled[name][f"K{K_LONG}"]["free"]["r2"]
        ax.bar(x + (j - 1) * 0.27, [base - pooled[name][f"K{K_LONG}"][m]["r2"] for m in MODELS], 0.27, color=color, label=name)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("gap to free fit (within-cluster R², pooled held-out)")
    ax.set_title("Pooled K = 20: how much each model leaves on the table", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 1]
    for j, (name, color) in enumerate(TARGETS.items()):
        sh = per_mouse[name][f"K{K_LONG}"]["bic_best_share"]
        ax.bar(x + (j - 1) * 0.27, [sh[m] for m in MODELS], 0.27, color=color, label=name)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("share of mice where the model has the lowest BIC")
    ax.set_title("Per mouse K = 20: BIC winner", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 2]
    v = df.query("target == 'GRU v'")
    for kk, mk in ((f"K{K_SHORT}", "o"), (f"K{K_LONG}", "x")):
        ax.scatter(v[f"{kk} | two outcomes | p1"], v[f"{kk} | two outcomes | p3"], s=8, alpha=0.4, marker=mk,
                   label=f"{kk}: median λ_rew {per_mouse['GRU v'][kk]['lambda_rew_median']:.2f}, "
                         f"λ_unr {per_mouse['GRU v'][kk]['lambda_unr_median']:.2f}; "
                         f"λ_unr > λ_rew {100 * per_mouse['GRU v'][kk]['share_lambda_unr_gt_rew']:.0f}%")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":")
    ax.set_xlabel("λ_rew (two outcomes)")
    ax.set_ylabel("λ_unr")
    ax.set_title("GRU v per mouse: does K = 20 change the two decays?", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1, 3]
    for j, m in enumerate(MODELS):
        for kk, off, col in ((f"K{K_SHORT}", -0.2, "#999999"), (f"K{K_LONG}", 0.2, "#222222")):
            ax.boxplot(v[f"{kk} | {m} | r2"].dropna(), positions=[j + off], widths=0.35, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": col, "alpha": 0.5})
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("held-out within-cluster R² per mouse (grey K = 10, black K = 20)")
    ax.set_title("GRU v per mouse", fontsize=12)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I9d_v_kernel_long.png"))
    plt.close(fig)

    s12.write_json("i9d_v_kernel_long.json", {
        "K_short": K_SHORT, "K_long": K_LONG, "models": MODELS, "n_params_kernel": N_PAR,
        "three_traces_params": ["A_rew", "lambda_rew", "A_unr", "lambda_unr", "A_c", "lambda_c"],
        "pooled": {n: {k: jsonable(r) for k, r in d.items()} for n, d in pooled.items()}, "per_mouse": per_mouse,
        "n_pooled": int(len(sub)), "n_rows_eligible": int(len(rows)), "min_rows": MIN_ROWS, "seed": SEED,
    }, "analysis/i9d_v_kernel_long.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
