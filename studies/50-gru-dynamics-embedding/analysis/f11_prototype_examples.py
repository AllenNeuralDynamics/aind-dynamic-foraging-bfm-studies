"""F11: two example mice along each prototype gradient (switch rate, learning rate, bias), same task.

    python analysis/f11_prototype_examples.py

Uses F10's prototype basis (``_cache/f10_prototype_basis_<key>.parquet``: basis coordinates and
each prototype's embedding-predicted percentile) and the A1 trial table.
Selection, per prototype, among mice whose dominant task is TASK with ≥ MIN_TRIALS trials in it:
  low  — embedding-predicted percentile ≤ 0.20 and the animal's own feature percentile ≤ 0.30;
  high — predicted ≥ 0.80 and animal ≥ 0.70;
  both — the other two prototypes' predicted percentiles within [0.20, 0.80];
  pair — maximises (animal percentile gap) − 1.5 × Σ |difference in the other two predicted percentiles|.
For each pair (rows): the mice in the F10f_best plane; one representative TASK session per mouse
(≥ MIN_SES trials, closest to the mouse's own switch rate and mean P(right) in TASK; first
WINDOW trials shown); and a summary over all the mouse's TASK trials that targets the prototype:
  switch rate   — survival of run lengths, P(run ≥ n), ± SEM from N_BOOT session bootstraps;
  learning rate — P(choosing the side a block change favoured), from PRE trials before to QUIET
                  after every change of pR − pL by ≥ JUMP with no other change in that window. For
                  this pair both mice need ≥ MIN_EVENTS such changes. The panel also gives the
                  population check: across the pool, Spearman of each mouse's late rise
                  (trials 12–19 minus the pre-change level) with its embedding-predicted learning rate;
  bias          — P(right) against pR − pL.
Everything shown is the animal's own choices; the GRU's P(right) is drawn in the session panels.
Writes ``f11_prototype_examples.json`` and ``fig_F11_prototype_examples.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

TASK = "Uncoupled Baiting"
MIN_TRIALS, MIN_SES, WINDOW = 3000, 400, 400
PROTOS = ["switch rate", "learning rate (CTT)", "bias"]
PLANE = {"switch rate": (0, 2), "learning rate (CTT)": (0, 1), "bias": (1, 2)}
AXIS = ["axis 1 (switch rate)", "axis 2 (learning rate ⊥)", "axis 3 (bias ⊥)", "axis 4"]
COL = {"low": "#2166ac", "high": "#b2182b"}
JUMP, PRE, QUIET, MIN_EVENTS = 0.3, 5, 20, 40
SEED, N_BOOT = 11, 500
BIAS_GRID = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])


def ses_date(ses: str) -> str:
    return ses.split("_")[-2] if ses.count("_") >= 3 else ses


def pick_pair(basis: pd.DataFrame, pool: pd.Index, name: str) -> tuple[str, str]:
    b = basis.loc[pool]
    animal = basis[f"animal: {name}"].rank(pct=True).loc[pool]
    others = [o for o in PROTOS if o != name]
    mid = np.all([(b[f"pred pct: {o}"] >= 0.2) & (b[f"pred pct: {o}"] <= 0.8) for o in others], axis=0)
    lo = b.index[mid & (b[f"pred pct: {name}"] <= 0.2) & (animal <= 0.3)]
    hi = b.index[mid & (b[f"pred pct: {name}"] >= 0.8) & (animal >= 0.7)]
    best, pair = -np.inf, None
    for i in lo:
        for j in hi:
            score = (animal[j] - animal[i]) - 1.5 * sum(abs(b.at[i, f"pred pct: {o}"] - b.at[j, f"pred pct: {o}"]) for o in others)
            if score > best:
                best, pair = score, (i, j)
    return pair


def session_stats(d: pd.DataFrame) -> pd.DataFrame:
    a = d["action"].to_numpy()
    first = np.r_[True, d["ses_idx"].to_numpy()[1:] != d["ses_idx"].to_numpy()[:-1]]
    sw = np.r_[False, a[1:] != a[:-1]] & ~first
    return pd.DataFrame({"ses_idx": d["ses_idx"], "sw": sw, "ok": ~first, "right": a == 1}).groupby("ses_idx").agg(
        n=("ok", "size"), sw=("sw", "sum"), ok=("ok", "sum"), right=("right", "mean")).assign(rate=lambda g: g.sw / g.ok)


def representative_session(d: pd.DataFrame) -> str:
    st = session_stats(d)
    target = np.array([st.sw.sum() / st.ok.sum(), d["action"].eq(1).mean()])
    cand = st[st.n >= MIN_SES]
    z = ((cand[["rate", "right"]] - target) / st[["rate", "right"]].std().replace(0, 1)).pow(2).sum(1)
    return z.idxmin()


def run_survival(d: pd.DataFrame, n_max: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Pooled P(run ≥ n), and its SEM by bootstrapping sessions (runs within a session are not
    independent, so the session is the resampling unit)."""
    per_ses = []
    for _, g in d.groupby("ses_idx", sort=False):
        a = g["action"].to_numpy()
        runs = np.diff(np.flatnonzero(np.r_[True, a[1:] != a[:-1], True]))
        per_ses.append(np.array([len(runs)] + [(runs >= n).sum() for n in range(1, n_max + 1)]))
    c = np.array(per_ses)  # sessions × (number of runs, runs ≥ 1 … runs ≥ n_max)
    curve = c[:, 1:].sum(0) / c[:, 0].sum()
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        cb = c[rng.integers(0, len(c), len(c))]
        boot.append(cb[:, 1:].sum(0) / cb[:, 0].sum())
    return curve, np.std(boot, axis=0)


def change_curves(d: pd.DataFrame) -> np.ndarray:
    """Per block change: choices of the favoured side from PRE trials before to QUIET after."""
    rows = []
    for _, g in d.groupby("ses_idx", sort=False):
        dp = (g["reward_probabilityR"] - g["reward_probabilityL"]).to_numpy()
        a = g["action"].to_numpy()
        for t0 in np.flatnonzero(dp[1:] != dp[:-1]) + 1:
            if abs(dp[t0] - dp[t0 - 1]) < JUMP or t0 < PRE or t0 + QUIET > len(dp):
                continue
            if np.any(dp[t0:t0 + QUIET] != dp[t0]) or np.any(dp[t0 - PRE:t0] != dp[t0 - 1]):
                continue
            rows.append((a[t0 - PRE:t0 + QUIET] == (1 if dp[t0] > dp[t0 - 1] else 0)).astype(float))
    return np.array(rows).reshape(-1, PRE + QUIET)


def bias_curve(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """P(right) at each value of pR − pL. TASK uses probabilities 0.1 / 0.4 / 0.7, so the
    differences are 0, ±0.3, ±0.6; snap to that grid (a binned edge at ±0.3 would split one
    condition by float rounding). The < 1% of trials off the grid go to the nearest value."""
    dp = (d["reward_probabilityR"] - d["reward_probabilityL"]).to_numpy()
    k = np.clip(np.rint(dp / 0.3), -2, 2).astype(int) + 2
    r = (d["action"] == 1).to_numpy(float)
    mean = np.array([r[k == i].mean() if (k == i).sum() >= 30 else np.nan for i in range(5)])
    sem = np.array([r[k == i].std() / np.sqrt((k == i).sum()) if (k == i).sum() >= 30 else np.nan for i in range(5)])
    return mean, sem


def main() -> None:
    s12.apply_presentation_style()
    basis = pd.read_parquet(s12.CACHE / f"f10_prototype_basis_{s12.MODEL_KEY}.parquet")
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "p_model1_right",
                         "reward_probabilityL", "reward_probabilityR", "task"])
    t = t[t["action"].isin([0, 1])]
    dom = t.groupby("subject_id")["task"].agg(lambda s: s.value_counts().index[0])
    n_task = t[t["task"] == TASK].groupby("subject_id").size()
    pool = basis.index.intersection(dom.index[dom == TASK]).intersection(n_task.index[n_task >= MIN_TRIALS])
    tt = t[(t["task"] == TASK) & t["subject_id"].isin(pool)]
    events = {m: change_curves(d) for m, d in tt.groupby("subject_id", sort=False)}
    n_events = pd.Series({m: len(e) for m, e in events.items()})
    late = pd.Series({m: e[:, PRE + 12:].mean() - e[:, :PRE].mean() for m, e in events.items() if len(e) >= 20})
    rho_late = spearmanr(late, basis.loc[late.index, "pred pct: learning rate (CTT)"])[0]
    rho_late_fit = spearmanr(late, basis.loc[late.index, "animal: learning rate (CTT)"], nan_policy="omit")[0]
    print("pool", len(pool), "| late rise vs predicted / fitted learning rate", round(rho_late, 2), round(rho_late_fit, 2), flush=True)

    fig, axes = plt.subplots(3, 4, figsize=(36, 22), layout="constrained", gridspec_kw={"width_ratios": [1, 1.55, 1.55, 1.1]})
    out = {}
    pct_all = basis[[f"animal: {p}" for p in PROTOS]].rank(pct=True)
    for row, name in enumerate(PROTOS):
        label = name.replace(" (CTT)", "")
        lo, hi = pick_pair(basis, pool if name != "learning rate (CTT)" else pool.intersection(n_events.index[n_events >= MIN_EVENTS]), name)
        entry = {}
        # Embedding panel.
        ax = axes[row, 0]
        pa, pb = PLANE[name]
        r = pct_all[f"animal: {name}"]
        ax.scatter(basis[f"axis{pa + 1}"], basis[f"axis{pb + 1}"], s=30, c=r, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.6, lw=0)
        for tag, m, mk in (("low", lo, "v"), ("high", hi, "^")):
            ax.scatter(basis.at[m, f"axis{pa + 1}"], basis.at[m, f"axis{pb + 1}"], s=420, marker=mk, color=COL[tag],
                       edgecolors="k", linewidths=1.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS[pa], fontsize=21)
        ax.set_ylabel(AXIS[pb], fontsize=21)
        ax.set_title(label, fontsize=25)
        curves = {}
        for col, (tag, m) in enumerate((("low", lo), ("high", hi)), start=1):
            d = tt[tt["subject_id"] == m]
            ses = representative_session(d)
            g = d[d["ses_idx"] == ses].iloc[:WINDOW]
            x = np.arange(len(g))
            ax = axes[row, col]
            ax.plot(x, g["reward_probabilityR"], color="0.35", lw=1.6, drawstyle="steps-post", label="P(reward | right)")
            ax.plot(x, g["reward_probabilityL"], color="0.65", lw=1.6, ls="--", drawstyle="steps-post", label="P(reward | left)")
            right = (g["action"] == 1).to_numpy()
            rew = (g["reward"] > 0).to_numpy()
            ax.plot(x, pd.Series(right.astype(float)).rolling(10, min_periods=1).mean(), color=COL[tag], lw=2.4,
                    label="animal P(right), 10-trial mean")
            ax.plot(x, g["p_model1_right"].to_numpy(), color="C1", lw=1.0, alpha=0.8, label="GRU P(right)")
            for side, y in ((True, 1.08), (False, -0.08)):
                sel = right == side
                ax.vlines(x[sel & rew], y - 0.04, y + 0.04, color="k", lw=1.2)
                ax.vlines(x[sel & ~rew], y - 0.02, y + 0.02, color="0.6", lw=1.0)
            ax.set_ylim(-0.15, 1.15)
            ax.set_xlim(0, WINDOW)
            ax.set_xlabel("trial in session", fontsize=21)
            ax.set_ylabel("P(right) / P(reward)", fontsize=21)
            val = basis.at[m, f"animal: {name}"]
            ax.set_title(f"{tag} {label}: mouse {m}\nsession {ses_date(ses)}", fontsize=23, color=COL[tag])
            if name == "switch rate":
                surv, surv_sem = run_survival(d)
                curves[tag] = {"run_survival": surv, "run_survival_sem": surv_sem, "n_sessions": int(d["ses_idx"].nunique())}
            elif name == "learning rate (CTT)":
                e = events[m]
                curves[tag] = {"change_mean": e.mean(0), "change_sem": e.std(0) / np.sqrt(len(e)), "n_changes": len(e)}
            else:
                mean, sem = bias_curve(d)
                curves[tag] = {"p_right": mean, "sem": sem}
            others = {o: float(basis.at[m, f"pred pct: {o}"]) for o in PROTOS}
            entry[tag] = {"subject_id": m, "animal_value": float(val), "animal_pct": float(r[m]), "pred_pct": others,
                          "session": ses, "n_task_trials": int(len(d)),
                          **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curves[tag].items()}}
        ax = axes[row, 3]
        if name == "switch rate":
            n = np.arange(1, 41)
            for tag in ("low", "high"):
                c = curves[tag]
                lo_b = np.maximum(c["run_survival"] - c["run_survival_sem"], 1e-4)
                ax.fill_between(n, lo_b, c["run_survival"] + c["run_survival_sem"], color=COL[tag], alpha=0.25, lw=0)
                ax.semilogy(n, c["run_survival"], "o-", ms=4, color=COL[tag], lw=2, label=f"mouse {entry[tag]['subject_id']}")
            ax.set_xlabel("run length n (same-side choices)", fontsize=21)
            ax.set_ylabel("P(run ≥ n)", fontsize=21)
            ax.set_title("how long they stay on a side", fontsize=25)
        elif name == "learning rate (CTT)":
            k = np.arange(-PRE, QUIET)
            for tag in ("low", "high"):
                c = curves[tag]
                ax.fill_between(k, c["change_mean"] - c["change_sem"], c["change_mean"] + c["change_sem"], color=COL[tag], alpha=0.2, lw=0)
                ax.plot(k, c["change_mean"], color=COL[tag], lw=2.2, label=f"mouse {entry[tag]['subject_id']}")
            ax.axvline(0, color="0.7", lw=0.8)
            ax.set_xlabel("trials since a reward-probability change", fontsize=21)
            ax.set_ylabel("P(choose the side it favoured)", fontsize=21)
            ax.set_ylim(0, 1)
            ax.set_title("how fast they follow a\nreward-probability change", fontsize=25)
        else:
            xc = BIAS_GRID
            for tag in ("low", "high"):
                c = curves[tag]
                ax.errorbar(xc, c["p_right"], c["sem"], fmt="o-", color=COL[tag], lw=2.2, ms=6, capsize=3,
                            label=f"mouse {entry[tag]['subject_id']}")
            ax.axhline(0.5, color="0.7", lw=0.8)
            ax.axvline(0, color="0.7", lw=0.8)
            ax.set_xlabel("P(reward | right) − P(reward | left)", fontsize=21)
            ax.set_ylabel("P(right)", fontsize=21)
            ax.set_ylim(0, 1)
            ax.set_xticks(BIAS_GRID)
            ax.set_title("which side they prefer at equal value", fontsize=25)
        ax.legend(frameon=False, fontsize=19)
        for a in axes[row]:
            a.tick_params(labelsize=18)
        out[name] = entry
        print(name, {tag: (e["subject_id"], round(e["animal_value"], 3), round(e["animal_pct"], 2),
                           {o: round(v, 2) for o, v in e["pred_pct"].items()}, e["session"]) for tag, e in entry.items() if tag in COL},
              flush=True)
    h, lab = axes[0, 1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="k", lw=1.5, ls="none", marker="|", ms=12), Line2D([], [], color="0.6", lw=1, ls="none", marker="|", ms=8)]
    lab += ["choice, rewarded (top: right, bottom: left)", "choice, unrewarded"]
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=20, bbox_to_anchor=(0.5, -0.075))
    fig.suptitle(f"Two mice along each embedding gradient ({TASK}; animal choices)", fontsize=30)
    fig.savefig(s12.figure_path("fig_F11_prototype_examples.png"), bbox_inches="tight")
    plt.close(fig)
    s12.write_json("f11_prototype_examples.json", {
        "task": TASK, "pool_size": int(len(pool)), "min_task_trials": MIN_TRIALS, "window": WINDOW,
        "selection": "low: pred pct ≤ 0.2 & animal pct ≤ 0.3; high: ≥ 0.8 & ≥ 0.7; others' pred pct in [0.2, 0.8]",
        "block_change": {"jump": JUMP, "pre": PRE, "quiet": QUIET, "min_events_for_pair": MIN_EVENTS,
                         "n_mice": int(len(late)), "spearman_late_rise_vs_predicted_learning_rate": float(rho_late),
                         "spearman_late_rise_vs_fitted_learning_rate": float(rho_late_fit)},
        "pairs": out,
    }, "analysis/f11_prototype_examples.py")


if __name__ == "__main__":
    main()
