"""F12: does a mouse's context trajectory over training track how its behaviour changes?

    python analysis/f12_context_learning_examples.py

The context for mouse m in session s is c(m, s) = embedding(m) + δ(embedding(m), phase(s)),
phase = s / N_m (F4c): a smooth, phase-driven path fixed by the embedding — the network's
expected learning curve for this kind of mouse. It cannot react to what happened in a session.
Uses the eval-session contexts (``_cache/contexts_<key>.npz``), F10's prototype basis transform,
and the A1 trial table, restricted to each mouse's dominant task.
1. Per eval session, for the animal and for the GRU's P(right):
     switch rate      P(choice ≠ previous choice);
     side preference  P(right) − 0.5;
     adaptation       P(favoured side) at trials 12–19 after a reward-probability change (pR − pL
                      jumps by ≥ JUMP, no other change for QUIET trials) minus its level over the
                      PRE trials before (NaN without such a change).
2. Context-predicted behaviour: across mice, a linear map from the mouse's mean context to its
   animal feature (dominant-task sessions pooled; 5-fold CV R² reported), applied to every
   session's context — the predicted learning curve, in the feature's own units.
3. Within each mouse with ≥ MIN_SES dominant-task eval sessions: Spearman between its
   context-predicted and its observed per-session feature. Null: the same mouse's observed
   series against another random mouse's predicted curve, interpolated at the same training
   phases (keeps the shared training trend, breaks the mouse-specific path); N_NULL draws of
   the population median.
4. Example: the mouse with the largest ρ × (predicted range / between-mouse SD). Only switch
   rate beats the null, so only it is drawn; side preference and adaptation are reported in the
   JSON as negative results. Picking the best of ~200 mice is selection; the population result
   (JSON, report) is the evidence. Panels: the context path over the mice coloured by their
   switch-rate rank; the per-session switch rate; run-length survival (F11's method, session
   bootstrap SEM) in the first vs last quarter of the mouse's eval sessions.
Writes ``f12_context_learning_examples.json`` and ``fig_F12_context_learning_examples.png``.
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
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from f11_prototype_examples import run_survival  # noqa: E402

SEED, N_NULL = 12, 200
MIN_SES, MIN_ADAPT_SES = 15, 10
JUMP, PRE, QUIET = 0.3, 5, 20
FEATS = ["switch rate", "side preference", "adaptation"]
PROTO = {"switch rate": "switch rate", "side preference": "bias", "adaptation": "learning rate"}
PLANE = {"switch rate": (0, 2), "side preference": (1, 2), "adaptation": (0, 1)}
AXIS = ["axis 1 (switch rate)", "axis 2 (learning rate ⊥)", "axis 3 (bias ⊥)", "axis 4"]


def session_features(g: pd.DataFrame) -> dict:
    a = g["action"].to_numpy()
    p = g["p_model1_right"].to_numpy()
    prev = a[:-1]
    out = {"animal switch rate": float(np.mean(a[1:] != prev)),
           "gru switch rate": float(np.mean(np.where(prev == 1, 1 - p[1:], p[1:]))),
           "animal side preference": float(np.mean(a == 1) - 0.5), "gru side preference": float(np.mean(p) - 0.5)}
    dp = (g["reward_probabilityR"] - g["reward_probabilityL"]).to_numpy()
    rows_a, rows_g = [], []
    for t0 in np.flatnonzero(dp[1:] != dp[:-1]) + 1:
        if abs(dp[t0] - dp[t0 - 1]) < JUMP or t0 < PRE or t0 + QUIET > len(dp):
            continue
        if np.any(dp[t0:t0 + QUIET] != dp[t0]) or np.any(dp[t0 - PRE:t0] != dp[t0 - 1]):
            continue
        right = dp[t0] > dp[t0 - 1]
        fa = (a[t0 - PRE:t0 + QUIET] == (1 if right else 0)).astype(float)
        fg = p[t0 - PRE:t0 + QUIET] if right else 1 - p[t0 - PRE:t0 + QUIET]
        rows_a.append(fa[PRE + 12:].mean() - fa[:PRE].mean())
        rows_g.append(fg[PRE + 12:].mean() - fg[:PRE].mean())
    out["animal adaptation"] = float(np.mean(rows_a)) if rows_a else np.nan
    out["gru adaptation"] = float(np.mean(rows_g)) if rows_g else np.nan
    out["n changes"] = len(rows_a)
    return out


def main() -> None:
    s12.apply_presentation_style()
    rng = np.random.default_rng(SEED)
    z = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz", allow_pickle=True)
    ctx = pd.DataFrame(z["contexts"], index=pd.Index(z["session_ids"], name="ses_idx"), columns=[f"c{k}" for k in range(4)])
    outputs = s12.CACHE / "models" / s12.MODEL_KEY / "run" / "outputs"
    cmap = json.loads((outputs / "session_context_map.json").read_text())
    pos = {sid: (e["subject_id"], i + 1, len(e["ordered_session_ids"])) for e in cmap["per_subject"]
           for i, sid in enumerate(e["ordered_session_ids"])}
    ctx["subject_id"] = [pos[s][0] for s in ctx.index]
    ctx["phase"] = [pos[s][1] / pos[s][2] for s in ctx.index]
    tr = np.load(s12.CACHE / f"f10_prototype_transform_{s12.MODEL_KEY}.npz")
    coords = ((ctx[[f"c{k}" for k in range(4)]].to_numpy() - tr["mean"]) @ tr["whiten"] @ tr["basis"].T) / tr["scale"]
    for k in range(4):
        ctx[f"axis{k + 1}"] = coords[:, k]
    basis_mice = pd.read_parquet(s12.CACHE / f"f10_prototype_basis_{s12.MODEL_KEY}.parquet")

    t = s12.load_trials(["subject_id", "ses_idx", "action", "reward", "p_model1_right", "reward_probabilityL",
                         "reward_probabilityR", "task"])
    t = t[t["action"].isin([0, 1])]
    dom = t.groupby("subject_id")["task"].agg(lambda s: s.value_counts().index[0])
    t = t[t["task"].to_numpy() == dom.reindex(t["subject_id"]).to_numpy()]
    ses = pd.DataFrame({k: session_features(g) for k, g in t.groupby("ses_idx", sort=False) if len(g) >= 100}).T
    ses = ses.join(ctx, how="inner")
    ses["task"] = dom.reindex(ses["subject_id"]).to_numpy()
    print("sessions", len(ses), "| mice", ses["subject_id"].nunique(), flush=True)

    # Cross-mouse map: mean context → animal feature (in the feature's units).
    cdims = [f"c{k}" for k in range(4)]
    mouse = ses.groupby("subject_id")[cdims + [f"animal {f}" for f in FEATS]].mean()
    folds = KFold(5, shuffle=True, random_state=SEED)
    maps, cv_r2 = {}, {}
    for f in FEATS:
        m = mouse.dropna(subset=[f"animal {f}"])
        y = m[f"animal {f}"].to_numpy()
        pred = cross_val_predict(LinearRegression(), m[cdims].to_numpy(), y, cv=folds)
        cv_r2[f] = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        maps[f] = LinearRegression().fit(m[cdims].to_numpy(), y)
        ses[f"pred {f}"] = maps[f].predict(ses[cdims].to_numpy())
    sd = {f: float(mouse[f"animal {f}"].std()) for f in FEATS}
    print("cross-mouse CV R²", {f: round(v, 2) for f, v in cv_r2.items()}, flush=True)

    # Within-mouse agreement and the phase-matched null.
    counts = ses.groupby("subject_id").size()
    mice = counts.index[counts >= MIN_SES]
    by = {m: g.sort_values("phase") for m, g in ses[ses["subject_id"].isin(mice)].groupby("subject_id")}
    within, null_med = {}, {}
    for f in FEATS:
        rho, rng_pred = {}, {}
        for m, g in by.items():
            ok = g[f"animal {f}"].notna()
            if ok.sum() < (MIN_ADAPT_SES if f == "adaptation" else MIN_SES) or g[f"pred {f}"].nunique() < 3:
                continue
            rho[m] = spearmanr(g.loc[ok, f"pred {f}"], g.loc[ok, f"animal {f}"])[0]
            rng_pred[m] = float(np.ptp(g[f"pred {f}"]) / sd[f])
        keys = list(rho)
        meds = []
        for _ in range(N_NULL):
            vals = []
            for m in keys:
                o = keys[rng.integers(len(keys))]
                while o == m:
                    o = keys[rng.integers(len(keys))]
                g, h = by[m], by[o]
                ok = g[f"animal {f}"].notna()
                pred_o = np.interp(g.loc[ok, "phase"], h["phase"], h[f"pred {f}"])
                vals.append(spearmanr(pred_o, g.loc[ok, f"animal {f}"])[0])
            meds.append(np.nanmedian(vals))
        within[f] = pd.DataFrame({"rho": pd.Series(rho), "pred_range_sd": pd.Series(rng_pred)})
        null_med[f] = np.array(meds)
        w = within[f]
        print(f, "| mice", len(w), "| median ρ", round(w.rho.median(), 3), "| null median 95%", round(np.percentile(meds, 95), 3),
              "| share ρ>0", round((w.rho > 0).mean(), 2), flush=True)

    # Figure.
    # Only features whose population beats the null get an example; the others are reported.
    fig, axes = plt.subplots(1, 3, figsize=(25, 8.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.2, 1]})
    out = {}
    for f in FEATS:
        w = within[f]
        score = w["rho"] * w["pred_range_sd"]
        m = score.idxmax()
        g = by[m]
        out[f] = {"example_candidate": {"subject_id": m, "rho": float(w.at[m, "rho"]), "pred_range_sd": float(w.at[m, "pred_range_sd"]),
                                        "n_sessions": int(len(g)), "task": g["task"].iloc[0]},
                  "n_mice": int(len(w)), "median_rho": float(w["rho"].median()), "share_rho_positive": float((w["rho"] > 0).mean()),
                  "null_median_mean": float(null_med[f].mean()), "null_median_95": float(np.percentile(null_med[f], 95)),
                  "beats_null": bool(w["rho"].median() > np.percentile(null_med[f], 95)),
                  "cross_mouse_cv_r2": cv_r2[f], "between_mouse_sd": sd[f]}
        if f != "switch rate":
            continue
        out[f]["example_quarters"] = None  # filled below
        row = 0
        pa, pb = PLANE[f]
        ax = axes[0]
        ax.scatter(basis_mice[f"axis{pa + 1}"], basis_mice[f"axis{pb + 1}"], s=30, c=basis_mice["animal: switch rate"].rank(pct=True),
                   cmap="RdBu_r", vmin=0, vmax=1, alpha=0.6, lw=0)
        pts = g[[f"axis{pa + 1}", f"axis{pb + 1}"]].to_numpy()
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap="viridis", lw=4)
        lc.set_array(g["phase"].to_numpy()[:-1])
        ax.add_collection(lc)
        ax.scatter(*pts[0], s=220, color=plt.get_cmap("viridis")(0.0), edgecolors="k", zorder=5, label="first session")
        ax.scatter(*pts[-1], s=220, marker="s", color=plt.get_cmap("viridis")(1.0), edgecolors="k", zorder=5, label="last session")
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS[pa], fontsize=21)
        ax.set_ylabel(AXIS[pb], fontsize=21)
        ax.set_title(f"mouse {m}", fontsize=24)
        if row == 0:
            ax.legend(frameon=False, fontsize=17, loc="lower right")
        ax = axes[1]
        x = g["phase"].to_numpy()
        ax.plot(x, g[f"animal {f}"], "o", color="k", ms=9, label="animal, per session")
        ax.plot(x, g[f"gru {f}"], "-", color="C1", lw=1.8, alpha=0.9, label="GRU, per session")
        ax.plot(x, g[f"pred {f}"], "-", color="C2", lw=5, label="predicted from the context alone")
        ax.set_xlabel("training phase (session / number of sessions)", fontsize=21)
        ax.set_ylabel(f, fontsize=21)
        ax.set_title(f"{g['task'].iloc[0]}, {len(g)} sessions", fontsize=24)
        if row == 0:
            ax.legend(frameon=False, fontsize=21)
        # Run-length survival, first vs last quarter of the mouse's eval sessions (F11's method).
        ax = axes[2]
        q = max(1, int(np.ceil(len(g) / 4)))
        n = np.arange(1, 41)
        tm = t[t["subject_id"] == m]
        quarters = {}
        for tag, ids, col in (("first 25%", g.index[:q], plt.get_cmap("viridis")(0.0)),
                              ("last 25%", g.index[-q:], plt.get_cmap("viridis")(0.8))):
            surv, sem = run_survival(tm[tm["ses_idx"].isin(ids)])
            quarters[tag] = {"sessions": list(ids), "run_survival": surv.tolist(), "run_survival_sem": sem.tolist()}
            ax.fill_between(n, np.maximum(surv - sem, 1e-4), surv + sem, color=col, alpha=0.25, lw=0)
            ax.semilogy(n, surv, "o-", ms=4, lw=2, color=col, label=f"{tag} of sessions ({len(ids)})")
        ax.set_xlabel("run length n (same-side choices)", fontsize=21)
        ax.set_ylabel("P(run ≥ n)", fontsize=21)
        ax.set_title("how long it stays on a side", fontsize=24)
        ax.legend(frameon=False, fontsize=18)
        for a in axes:
            a.tick_params(labelsize=18)
        out[f]["example_quarters"] = quarters
    fig.suptitle("Context path and switch-rate change over training", fontsize=28)
    fig.savefig(s12.figure_path("fig_F12_context_learning_examples.png"), bbox_inches="tight")
    plt.close(fig)
    s12.write_json("f12_context_learning_examples.json", {
        "min_sessions": MIN_SES, "n_null": N_NULL, "features": out}, "analysis/f12_context_learning_examples.py")


if __name__ == "__main__":
    main()
