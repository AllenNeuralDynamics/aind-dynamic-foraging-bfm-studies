"""F7: a behavioural feature library, how much of it the embedding carries, and named embedding axes.

    python analysis/f7_feature_library.py

1. Features, per mouse and per source — the animal's choices, and the GRU's predicted
   P(right) on the same trials (soft labels; model-free features use expected values):
     determinism      mean |history-driven logit| of the mouse's Su2022 fit (intercept excluded):
                      the overall scale of history dependence, kept as its own feature
     stickiness ratio UnrC(1) / RewC(1): how much of the tendency to repeat survives no reward
     win-stay, lose-stay   P(stay | previous rewarded / unrewarded)
     reward memory span    Σ lags 2–5 / lag 1 of the reward effect (RewC − UnrC)
     choice memory span    Σ lags 2–5 / lag 1 of persistence ((RewC + UnrC) / 2)
     bias             Su2022 intercept (rightward)
     matching sensitivity / bias   Baum fit log(C_R/C_L) = s·log(I_R/I_L) + b over blocks of
                      constant reward probabilities (incomes are the animal's)
     poorer-side rate P(choosing the poorer side) ≥ 10 trials after a probability change, |ΔP| ≥ 0.3
     switch rate      P(switch)
     return after failed switch    P(back to the old side | t−1 switched and was unrewarded)
     stay through losses           P(stay | ≥ 2 losses in a row on a side just switched to)
   Reliability: each feature recomputed on odd and even sessions; Spearman across mice,
   Spearman–Brown corrected (the ceiling for any predictor of the full-data feature).
2. Embedding mapping: rank R² from the 4-d embedding (F3's ``decode``), as a share of the
   feature's reliability; and after removing the determinism scale (rank-residual on rank
   determinism).
3. Named axes: standardised GRU-source features are regressed on the whitened embedding; the
   loading matrix (features × 4) is varimax-rotated into four orthogonal embedding axes, each
   named by its top-loading features; every mouse gets coordinates (z) on these axes.
Writes ``f7_feature_library.json``, ``_cache/f7_mouse_features_<key>.parquet``,
``fig_F7a_feature_library.png`` and ``fig_F7b_named_axes.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

SEED = 15
N_BACK = 10
MIN_ROWS = 800
STEADY, MIN_DP = 10, 0.3
FEATURES = ["determinism", "stickiness ratio", "win-stay", "lose-stay", "reward memory span", "choice memory span", "bias",
            "matching sensitivity", "matching bias", "poorer-side rate", "switch rate", "return after failed switch",
            "stay through losses"]


def varimax(phi: np.ndarray, gamma: float = 1.0, iters: int = 200, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    p, k = phi.shape
    r = np.eye(k)
    d = 0
    for _ in range(iters):
        lam = phi @ r
        u, s, vh = np.linalg.svd(phi.T @ (lam ** 3 - (gamma / p) * lam @ np.diag(np.diag(lam.T @ lam))))
        r = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return phi @ r, r


def kernel_derived(coef: np.ndarray, x_rows: np.ndarray) -> dict:
    rew, unr = (coef[i * N_BACK:(i + 1) * N_BACK][::-1] for i in range(2))
    eff, per = rew - unr, (rew + unr) / 2
    hist_logit = x_rows @ coef[:-1]
    return {"determinism": float(np.mean(np.abs(hist_logit))),
            "stickiness ratio": float(unr[0] / rew[0]) if rew[0] > 0.1 else np.nan,
            "reward memory span": float(eff[1:5].sum() / eff[0]) if eff[0] > 0.05 else np.nan,
            "choice memory span": float(per[1:5].sum() / per[0]) if per[0] > 0.05 else np.nan,
            "bias": float(coef[-1])}


def model_free(d: pd.DataFrame, p_choice: np.ndarray) -> dict:
    """d: one mouse's (subset of) trials in order; p_choice: P(right) used (choice 0/1 or GRU p)."""
    a, r, ses = d["action"].to_numpy(), d["reward"].to_numpy() > 0, d["ses_idx"].to_numpy()
    n = len(d)
    first = np.r_[True, ses[1:] != ses[:-1]]
    prev_a = np.r_[-1, a[:-1]]
    prev_r = np.r_[False, r[:-1]]
    ok = ~first
    p_stay = np.where(prev_a == 1, p_choice, 1 - p_choice)
    out = {"win-stay": float(p_stay[ok & prev_r].mean()) if (ok & prev_r).sum() > 30 else np.nan,
           "lose-stay": float(p_stay[ok & ~prev_r].mean()) if (ok & ~prev_r).sum() > 30 else np.nan,
           "switch rate": float(1 - p_stay[ok].mean())}
    # Matching (Baum) over blocks of constant probabilities.
    pl, pr = d["reward_probabilityL"].to_numpy(), d["reward_probabilityR"].to_numpy()
    block = np.cumsum(first | np.r_[True, (pl[1:] != pl[:-1]) | (pr[1:] != pr[:-1])])
    df = pd.DataFrame({"b": block, "cR": p_choice, "cL": 1 - p_choice, "iR": r & (a == 1), "iL": r & (a == 0)})
    g = df.groupby("b").sum()
    g = g[(g["cR"] + g["cL"]) >= 20]
    if len(g) >= 5:
        y = np.log((g["cR"] + 0.5) / (g["cL"] + 0.5))
        x = np.log((g["iR"] + 0.5) / (g["iL"] + 0.5))
        s, b = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (np.nan, np.nan)
        out["matching sensitivity"], out["matching bias"] = float(s), float(b)
    else:
        out["matching sensitivity"] = out["matching bias"] = np.nan
    since = pd.Series(np.ones(n)).groupby(block).cumsum().to_numpy() - 1
    steady = (since >= STEADY) & (np.abs(pr - pl) >= MIN_DP)
    p_poor = np.where(pr > pl, 1 - p_choice, p_choice)
    out["poorer-side rate"] = float(p_poor[steady].mean()) if steady.sum() > 50 else np.nan
    # Failed-switch return and staying through losses after a switch.
    ok2 = np.zeros(n, bool)
    ok2[2:] = ~first[2:] & ~first[1:-1]
    switched = np.zeros(n, bool)
    switched[2:] = a[1:-1] != a[:-2]
    rows = ok2 & switched & ~prev_r
    old = np.r_[-1, -1, a[:-2]]
    p_ret = np.where(old == 1, p_choice, 1 - p_choice)
    out["return after failed switch"] = float(p_ret[rows].mean()) if rows.sum() > 20 else np.nan
    new_seg = first | np.r_[True, (a[1:] != a[:-1]) | r[:-1]]
    streak = np.where(r, 0, pd.Series(np.ones(n)).groupby(np.cumsum(new_seg)).cumsum().to_numpy())
    n_streak = np.r_[0, np.where(first[1:], 0, streak[:-1])].astype(int)
    idx = np.arange(n)
    start, before = idx - n_streak, idx - n_streak - 1
    valid = (n_streak >= 2) & (before >= 0)
    valid[valid] &= ses[before[valid]] == ses[idx[valid]]
    after_switch = np.zeros(n, bool)
    after_switch[valid] = a[before[valid]] != a[start[valid]]
    sel = valid & after_switch
    out["stay through losses"] = float(p_stay[sel].mean()) if sel.sum() > 20 else np.nan
    return out


def main() -> None:
    from c1_kernels import design, fit_rows  # noqa: PLC0415
    from f3_embedding_features import decode  # noqa: PLC0415

    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "p_model1_right",
                         "reward_probabilityL", "reward_probabilityR", "subject_session_index"])
    order = t.drop_duplicates("ses_idx")[["subject_id", "ses_idx", "subject_session_index"]].sort_values(["subject_id", "subject_session_index"])
    order["half"] = order.groupby("subject_id").cumcount() % 2  # alternate sessions → odd / even
    half_of = order.set_index("ses_idx")["half"]
    t["half"] = t["ses_idx"].map(half_of).to_numpy()

    # Kernel fits: all / half 0 / half 1, animal and GRU.
    x, td = design(N_BACK)
    td_half = td["ses_idx"].map(half_of).to_numpy()
    targets = {"animal": (td["action"].to_numpy().astype(float), False), "gru": (td["p_model1_right"].to_numpy(), True)}
    jobs = []
    for subject, rows in td.groupby("subject_id", sort=False).indices.items():
        for part in ("all", 0, 1):
            rr = rows if part == "all" else rows[td_half[rows] == part]
            if len(rr) >= MIN_ROWS:
                jobs += [(subject, part, src, rr) for src in targets]
    coefs = Parallel(n_jobs=24)(delayed(fit_rows)(x, targets[src][0], rr, targets[src][1]) for _, _, src, rr in jobs)
    kern = {(s, part, src): kernel_derived(c, x[rr]) for (s, part, src, rr), c in zip(jobs, coefs)}
    print("kernel fits", len(coefs), flush=True)

    rows_out = []
    for subject, d in t.groupby("subject_id", sort=False):
        for part in ("all", 0, 1):
            dd = d if part == "all" else d[d["half"] == part]
            for src in ("animal", "gru"):
                pc = dd["action"].to_numpy().astype(float) if src == "animal" else dd["p_model1_right"].to_numpy()
                rec = {"subject_id": subject, "part": str(part), "source": src, **model_free(dd, pc)}
                rec.update(kern.get((subject, part, src), {}))
                rows_out.append(rec)
    feats = pd.DataFrame(rows_out)
    feats.to_parquet(s12.CACHE / f"f7_mouse_features_{s12.MODEL_KEY}.parquet")

    emb = s12.subject_embeddings()
    rng = np.random.default_rng(SEED)
    summary = {}
    full = {src: feats.query("part == 'all' and source == @src").set_index("subject_id")[FEATURES] for src in ("animal", "gru")}
    for src in ("animal", "gru"):
        h0 = feats.query("part == '0' and source == @src").set_index("subject_id")[FEATURES]
        h1 = feats.query("part == '1' and source == @src").set_index("subject_id")[FEATURES]
        mice = emb.index.intersection(full[src].index)
        xe = emb.loc[mice].to_numpy()
        det = full[src].loc[mice, "determinism"].to_numpy()
        for f in FEATURES:
            both = h0[f].dropna().index.intersection(h1[f].dropna().index)
            r_half = spearmanr(h0.loc[both, f], h1.loc[both, f])[0]
            rel = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
            y = full[src].loc[mice, f].to_numpy(dtype=float)
            dec = decode(xe, y, rng)
            entry = {"split_half_spearman": float(r_half), "reliability_spearman_brown": float(rel),
                     "embedding_rank_r2": dec["rank_r2"], "null_95": dec["null_95"],
                     "share_of_reliable": float(dec["rank_r2"] / rel ** 2) if rel > 0 else np.nan, "n": dec["n"]}
            if f != "determinism":
                ok = np.isfinite(y) & np.isfinite(det)
                ry, rd = rankdata(y[ok]) / ok.sum(), rankdata(det[ok]) / ok.sum()
                resid = ry - np.polyval(np.polyfit(rd, ry, 1), rd)
                yy = np.full(len(y), np.nan)
                yy[ok] = resid
                entry["embedding_rank_r2_beyond_determinism"] = decode(xe, yy, rng)["rank_r2"]
                entry["spearman_with_determinism"] = float(spearmanr(y[ok], det[ok])[0])
            summary[f"{src} | {f}"] = entry
        print(src, {f: (round(summary[f'{src} | {f}']['reliability_spearman_brown'], 2), round(summary[f'{src} | {f}']['embedding_rank_r2'], 2))
                    for f in FEATURES}, flush=True)

    # Named axes: varimax on loadings of GRU features on the whitened embedding.
    g = full["gru"].loc[emb.index.intersection(full["gru"].index)]
    xe = emb.loc[g.index].to_numpy()
    mu, cov = xe.mean(0), np.cov(xe, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    white = (xe - mu) @ evecs / np.sqrt(evals)  # whitened embedding, mice × 4 (unit variance, uncorrelated)
    z = (g - g.mean()) / g.std()
    z = z.fillna(0.0)
    load = np.array([np.linalg.lstsq(np.column_stack([white, np.ones(len(white))]), z[f].to_numpy(), rcond=None)[0][:4] for f in FEATURES])
    rot_load, rot = varimax(load)
    sign = np.sign(rot_load[np.argmax(np.abs(rot_load), axis=0), np.arange(4)])
    rot_load, rot = rot_load * sign, rot * sign
    coords = white @ rot  # each mouse's coordinate on the four named axes (unit variance)
    order_ax = np.argsort(-np.sum(rot_load ** 2, axis=0))
    rot_load, rot, coords = rot_load[:, order_ax], rot[:, order_ax], coords[:, order_ax]
    names = []
    for k in range(4):
        top = np.argsort(-np.abs(rot_load[:, k]))[:3]
        names.append(" / ".join(("+" if rot_load[i, k] > 0 else "−") + FEATURES[i] for i in top))
    r2_axis = [float(np.sum(rot_load[:, k] ** 2)) for k in range(4)]
    animal_corr = {names[k]: {f: float(spearmanr(coords[:, k], full["animal"].loc[g.index, f], nan_policy="omit")[0]) for f in FEATURES}
                   for k in range(4)}
    pd.DataFrame(coords, index=g.index, columns=[f"axis{k + 1}" for k in range(4)]).to_parquet(
        s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.parquet")
    np.savez(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.npz", mean=mu, whiten=evecs / np.sqrt(evals), rotation=rot,
             features=np.array(FEATURES), loadings=rot_load)
    print("axes", names, flush=True)

    # Figures.
    fig, axes = plt.subplots(1, 3, figsize=(30, 9), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    for ax, src in zip(axes[:2], ("animal", "gru")):
        y = np.arange(len(FEATURES))
        rel = [summary[f"{src} | {f}"]["reliability_spearman_brown"] for f in FEATURES]
        r2 = [summary[f"{src} | {f}"]["embedding_rank_r2"] for f in FEATURES]
        beyond = [summary[f"{src} | {f}"].get("embedding_rank_r2_beyond_determinism", np.nan) for f in FEATURES]
        ax.barh(y - 0.27, np.square(rel), 0.27, color="0.75", label="reliability² (ceiling for rank R²)")
        ax.barh(y, r2, 0.27, color="#E69F00", label="rank R² from the embedding")
        ax.barh(y + 0.27, beyond, 0.27, color="#4C72B0", label="… after removing determinism")
        ax.set_yticks(y, FEATURES, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("variance explained across mice (rank)")
        ax.set_title(f"{'Animals' if src == 'animal' else 'GRU'}: reliability vs what the embedding carries", fontsize=13)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax = axes[2]
    cmat = full["animal"].rank().corr().to_numpy()
    im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=9)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{cmat[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Animals: Spearman between features across mice", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F7a_feature_library.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(30, 9), gridspec_kw={"width_ratios": [0.9, 1, 1]})
    ax = axes[0]
    im = ax.imshow(rot_load, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=11)
    ax.set_xticks(range(4), [f"axis {k + 1}\n(R² sum {r2_axis[k]:.1f})" for k in range(4)], fontsize=10)
    for i in range(len(FEATURES)):
        for j in range(4):
            ax.text(j, i, f"{rot_load[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Varimax-rotated embedding axes: loadings of the GRU features", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for ax, (i, j) in zip(axes[1:], ((0, 1), (2, 3))):
        sc = ax.scatter(coords[:, i], coords[:, j], s=9, c=full["animal"].loc[g.index, "determinism"].rank(pct=True), cmap="viridis")
        ax.set_xlabel(f"axis {i + 1}: {names[i]}", fontsize=10)
        ax.set_ylabel(f"axis {j + 1}: {names[j]}", fontsize=10)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_title("Each mouse on the named axes (colour: animal determinism rank)", fontsize=12)
    fig.colorbar(sc, ax=axes[1:], fraction=0.02)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F7b_named_axes.png"))
    plt.close(fig)

    s12.write_json("f7_feature_library.json", {
        "features": FEATURES, "n_mice": int(len(g)), "per_feature": summary,
        "named_axes": {"names": names, "loadings": {f: rot_load[i].tolist() for i, f in enumerate(FEATURES)},
                       "variance_explained_sum": r2_axis, "spearman_with_animal_features": animal_corr,
                       "method": "GRU features (z) regressed on the whitened embedding; varimax rotation of the 13 × 4 loadings"},
        "reliability": "odd vs even sessions within each mouse; Spearman across mice, Spearman–Brown corrected",
        "definitions": __doc__.split("Reliability:")[0],
    }, "analysis/f7_feature_library.py")
    for k in range(4):
        print(f"axis {k + 1}", names[k], {f: round(v, 2) for f, v in zip(FEATURES, rot_load[:, k])})


if __name__ == "__main__":
    main()
