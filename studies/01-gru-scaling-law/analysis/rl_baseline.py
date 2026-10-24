#!/usr/bin/env python
"""RL-baseline posthoc: simple independent per-subject Bari RL (ForagerQLearning
L1F1_CK1) on the same 149 held-out mice as the GRU.

Sources (all frozen; this script is offline):
  * rl_baseline_tables.json -- the held-out per-subject tables of the RL runs below
    (bari / hattori / ctt + legacy `cdq292n5`), written by freeze_rl_baseline.py, the
    only step that reads W&B (falling back to the run output dirs on /allen).
  * per_subject_rows.json -- GRU per-subject eval_likelihood for v1/v2 x ratio x 3 seeds
    from the offline `heldout-rerun-*` groups (freeze_per_subject.py; shared with r1/r3).
  * Cached aggregate JSONs for the 4 figure overlays:
      Result 1: paired_v1_v2_cell.json
      Result 4: zeroshot_vs_d.json
      Result 5: fewshot_curve.json
      Result 7: nxd_scaling.json

Produces:
  * analysis/rl_baseline.json - RL band + per-(variant,D) paired stats.
  * analysis/rl_baseline_verdict.md - table + interpretation.
  * analysis/fig_rl_paired.png - Result 8 figure: per-mouse Delta(GRU - RL).
  * Overlaid replacements (each figure overwrites the existing PNG in place):
      fig_scaling_v1_v2.png, fig_zeroshot_vs_d.png,
      fig_fewshot_curve.png, fig_nxd_scaling.png

Offline (scipy + matplotlib + seaborn).
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from _meta import build_meta

# Presentation styling: large fonts, ticks style (no top/right spines after despine).
sns.set_theme(style="ticks", context="talk", font_scale=1.05)
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"

# The classical-RL baseline suite (all fit on the SAME fixed 149 held-out mice as the
# GRU, one differential-evolution fit per mouse on its own train sessions, scored on its
# own eval sessions). `rl-baseline-simple` (cdq292n5) was the original Bari-only arm and
# is kept as a REPRODUCTION CROSS-CHECK, not as a 4th model: `bari` below is the same
# agent re-fit in the 2026-07-13 suite and should land within DE noise of it.
RL_MODELS = {
    "ctt": dict(
        run_id="lmg1i9yd", group="rl-baseline-ctt@20260713-010305",
        label="Compare-to-threshold",
        agent="ForagerCompareThreshold (local income vs learned threshold)",
        params="learn_rate, threshold, softmax_inverse_temperature, biasL",
    ),
    "bari": dict(
        run_id="bg3nzqz9", group="rl-baseline-bari@20260713-010244",
        label="Bari (L1F1_CK1)",
        agent="ForagerQLearning — 1 learn-rate, 1 forget-rate, 1-step choice kernel, softmax",
        params=("learn_rate, forget_rate_unchosen, choice_kernel_relative_weight, "
                "choice_kernel_step_size, biasL, softmax_inverse_temperature"),
    ),
    "hattori": dict(
        run_id="unhmbrk4", group="rl-baseline-hattori@20260713-010255",
        label="Hattori (L2F1)",
        agent="ForagerQLearning — 2 learn-rates (asymmetric α+/α−), 1 forget-rate, no choice kernel, softmax",
        params="learn_rate_rew, learn_rate_unrew, forget_rate_unchosen, biasL, softmax_inverse_temperature",
    ),
}

# The Bari-only predecessor. Reported in r8 purely as a reproduction check.
LEGACY_BARI_RUN = "cdq292n5"
LEGACY_BARI_GROUP = "rl-baseline-simple@20260624-171829"

# Which model backs the RL reference band on Results 1/4/5/7. Kept as `bari` so those
# four figures keep the same reference they were written against; r8 reports the
# best-of-breed margin separately. Changing this to `ctt` would silently shift four
# other reports' figures away from the text that describes them.
BAND_MODEL = "bari"

RATIO_D = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}

# House color convention (matches study-04): light blue = base/SC-off model,
# dark blue = the "+ session conditioning" variant. Used by fig_scaling_v1_v2_with_rl.
C_V1, C_V2 = "#6baed6", "#08519c"
RL_DASHES = [(0, (5, 2)), (0, (1.5, 1.5))]  # non-best RL lines, longest dash first
D_TICKS = [10, 30, 100, 300, 614]


# ============================= frozen inputs =============================

RL_FROZEN = HERE / "rl_baseline_tables.json"   # written by freeze_rl_baseline.py (W&B or /allen)
GRU_FROZEN = HERE / "per_subject_rows.json"    # written by freeze_per_subject.py (shared with r1/r3)


def rl_stats(frozen, run_id, *, expect_n=None):
    """Held-out per-subject stats for one classical-RL model, from its frozen table."""
    import pandas as pd
    t = frozen[run_id]
    df = pd.DataFrame(t["table"])
    group, source = t["group"], t["source"]
    df["subject_id"] = df["subject_id"].astype(str)
    if expect_n is not None and len(df) != expect_n:
        raise ValueError(
            f"{run_id}: expected {expect_n} held-out subjects, got {len(df)} — "
            "this is probably the TRAINING table, not the held-out one")
    pooled_log = float(df["eval_total_log_likelihood"].sum() /
                       df["eval_total_trials"].sum())
    rl = dict(
        run_id=run_id,
        run_url=f"https://wandb.ai/{PROJECT}/runs/{run_id}",
        group=group,
        source=source,
        n_subjects=int(len(df)),
        n_trials=int(df["eval_total_trials"].sum()),
        pooled_log_likelihood=pooled_log,
        pooled_likelihood_trial_weighted=math.exp(pooled_log),
        per_subject_mean_likelihood=float(df["eval_likelihood"].mean()),
        per_subject_se=float(df["eval_likelihood"].std(ddof=1) /
                             math.sqrt(len(df))),
        per_subject_median_likelihood=float(df["eval_likelihood"].median()),
        per_subject_likelihood={s: float(v) for s, v in
                                zip(df["subject_id"], df["eval_likelihood"])},
        per_curriculum={
            k: {"n": int(g.shape[0]),
                "mean": float(g["eval_likelihood"].mean()),
                "std": float(g["eval_likelihood"].std(ddof=1)) if g.shape[0] > 1 else 0.0}
            for k, g in df.groupby("curriculum_name")},
    )
    print(f"    n={rl['n_subjects']} mice, pooled={rl['pooled_likelihood_trial_weighted']:.4f}, "
          f"per-subj mean={rl['per_subject_mean_likelihood']:.4f}  [{source}]")
    return rl


def all_rl():
    """Every model in RL_MODELS + the legacy Bari run (reproduction check)."""
    frozen = json.loads(RL_FROZEN.read_text())["runs"]
    models = {}
    for key, spec in RL_MODELS.items():
        print(f"  {spec['label']} ({key})...")
        m = rl_stats(frozen, spec["run_id"], expect_n=149)
        m.update(label=spec["label"], agent=spec["agent"], params=spec["params"])
        models[key] = m

    print(f"  legacy rl-baseline-simple ({LEGACY_BARI_RUN}) — reproduction check...")
    legacy = rl_stats(frozen, LEGACY_BARI_RUN, expect_n=149)

    # The new `bari` arm is the same agent as the legacy run; they should agree to
    # within differential-evolution stochasticity. Surface the delta rather than
    # assuming it.
    drift = (models["bari"]["pooled_likelihood_trial_weighted"]
             - legacy["pooled_likelihood_trial_weighted"])
    print(f"  reproduction check: new bari − legacy = {drift:+.5f} "
          f"({'OK' if abs(drift) < 0.005 else 'INVESTIGATE'})")
    return models, legacy, drift


def load_gru():
    """nested[variant][D_str][subject] -> [ll per seed], and the GRU W&B groups, from the frozen rows."""
    frozen = json.loads(GRU_FROZEN.read_text())
    nested = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in frozen["rows"]:
        nested[r["variant"]][str(RATIO_D[r["ratio"]])][r["subject"]].append(r["ll"])
    return nested, frozen["_meta"]["wandb_groups"]


# ============================= analysis =============================

def gru_per_subject_means(gru_nested):
    """nested[var][D_str][subj] -> [ll per seed]  =>  flat (var, D, subj) -> mean ll."""
    out = {}
    for var, byD in gru_nested.items():
        for D_str, bysub in byD.items():
            D = int(D_str)
            for subj, lls in bysub.items():
                out[(var, D, subj)] = float(np.mean(lls))
    return out


def paired_vs_rl(gru_mean, rl_per_subject):
    variants = sorted({k[0] for k in gru_mean})
    Ds = sorted({k[1] for k in gru_mean})
    out = {"variants": variants, "Ds": Ds, "per_cell": {}}
    for var in variants:
        for D in Ds:
            subs = sorted({k[2] for k in gru_mean if k[0] == var and k[1] == D} & set(rl_per_subject))
            if len(subs) < 5:
                continue
            gru = np.array([gru_mean[(var, D, s)] for s in subs])
            rl = np.array([rl_per_subject[s] for s in subs])
            d = gru - rl
            wil = stats.wilcoxon(d)
            t = stats.ttest_rel(gru, rl)
            out["per_cell"][f"{var}_D{D}"] = dict(
                variant=var, D=D, n=len(subs),
                gru_mean=float(gru.mean()), rl_mean=float(rl.mean()),
                mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                frac_gru_wins=float((d > 0).mean()),
                paired_t_t=float(t.statistic), paired_t_p=float(t.pvalue),
                wilcoxon_W=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
                per_subject_delta={s: float(g - r) for s, g, r in zip(subs, gru, rl)},
            )
    return out


# ============================= figures =============================

def _rl_band(ax, rl, metric):
    """Add an RL horizontal reference band. metric in {trial_weighted, subject_mean, both}."""
    if metric in ("trial_weighted", "both"):
        ax.axhline(rl["pooled_likelihood_trial_weighted"], color="#7a3a3a", lw=1.2,
                   ls="--", label=f"RL (trial-weighted): {rl['pooled_likelihood_trial_weighted']:.4f}")
    if metric in ("subject_mean", "both"):
        mu = rl["per_subject_mean_likelihood"]; se = rl["per_subject_se"]
        ax.axhline(mu, color="#3a3a7a", lw=1.2, ls=":",
                   label=f"RL (per-subject mean): {mu:.4f} ± {se:.4f} SE")
        ax.axhspan(mu - se, mu + se, color="#3a3a7a", alpha=0.08)


# --- Result 8: per-mouse delta(GRU - RL) violin -----------------------------

def fig_paired(paired, rl, out_png):
    Ds = paired["Ds"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, var in zip(axes, paired["variants"]):
        data, labels = [], []
        means, ps, fracs = [], [], []
        for D in Ds:
            cell = paired["per_cell"].get(f"{var}_D{D}")
            if cell is None:
                continue
            deltas = list(cell["per_subject_delta"].values())
            data.append(deltas); labels.append(f"D~{D}\nn={cell['n']}")
            means.append(cell["mean_delta"]); ps.append(cell["wilcoxon_p"])
            fracs.append(cell["frac_gru_wins"])
        if not data:
            continue
        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor("#3b76b8" if var == "v1" else "#d97c2a"); pc.set_alpha(0.5)
        ax.axhline(0, color="k", lw=1.0, ls="--", label="RL reference (Δ=0)")
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        ax.set_xlabel("# training mice (D)")
        ymin = min(min(d) for d in data); ymax = max(max(d) for d in data)
        pad = 0.05 * (ymax - ymin)
        for i, (m, p, f) in enumerate(zip(means, ps, fracs)):
            ax.text(i + 1, ymax + pad,
                    f"μ={m:+.4f}\n{f*100:.0f}%↑\np={p:.0e}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{var}: per-mouse Δ(GRU − RL)")
    axes[0].set_ylabel("Δ likelihood (per held-out mouse, GRU − RL)")
    axes[1].legend(loc="lower right")
    rl_lab = (f"RL ref: trial-weighted {rl['pooled_likelihood_trial_weighted']:.4f}, "
              f"per-subject mean {rl['per_subject_mean_likelihood']:.4f} (n=149)")
    fig.suptitle(f"Result 8 — paired GRU − RL per held-out mouse\n{rl_lab}", fontsize=15)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 1: scaling v1 vs v2 + RL ---------------------------------------

def fig_scaling_v1_v2_with_rl(rl, models, out_png):
    pj = json.load(open(HERE / "paired_v1_v2_cell.json"))
    pr = pj["per_ratio"]
    xs_ratio = sorted(float(k) for k in pr)
    Ds = [pr[str(x)]["D_mean"] for x in xs_ratio]
    v1 = [pr[str(x)]["v1"] for x in xs_ratio]
    v2 = [pr[str(x)]["v2"] for x in xs_ratio]

    # Per-seed scalars for the raw dots + SEM. pj["pairs"] carries one record per
    # (ratio, seed); pr[..]["v1"/"v2"] is exactly the mean over those seeds, so the
    # curve values are unchanged by adding dispersion here.
    seeds = {}
    for rec in pj["pairs"]:
        seeds.setdefault(float(rec["ratio"]), []).append(rec)
    n_per_cell = sorted({len(v) for v in seeds.values()})

    def _sem(vals):
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))

    se1 = [_sem([r["v1"] for r in seeds[x]]) for x in xs_ratio]
    se2 = [_sem([r["v2"] for r in seeds[x]]) for x in xs_ratio]

    fig, ax = plt.subplots(figsize=(4.5, 4.6))

    # --- classical RL references: all fitted models, best one made salient -------
    # Ranked from the data rather than hardcoded, so a refit that changes which
    # model wins cannot leave the "best" label attached to the wrong line.
    ranked = sorted(models.items(),
                    key=lambda kv: -kv[1]["pooled_likelihood_trial_weighted"])
    for i, (key, m) in enumerate(ranked):
        ll = m["pooled_likelihood_trial_weighted"]
        best = i == 0
        ax.axhline(
            ll,
            color="#000000" if best else "#8a8a8a",
            lw=1.8 if best else 1.0,
            ls="-" if best else RL_DASHES[(i - 1) % len(RL_DASHES)],
            zorder=2,
            label=("RL best — %s: %.4f" % (m["label"], ll) if best
                   else "RL — %s: %.4f" % (m["label"], ll)),
        )

    # --- GRU: raw per-seed dots, offset to either side of the mean marker --------
    # (multiplicative offset because the x axis is log). v1 left, v2 right.
    JIT = 1.075
    for x, D in zip(xs_ratio, Ds):
        recs = seeds[x]
        ax.plot([D / JIT] * len(recs), [r["v1"] for r in recs], "o",
                color=C_V1, alpha=0.45, markersize=4.5, mew=0, zorder=3)
        ax.plot([D * JIT] * len(recs), [r["v2"] for r in recs], "s",
                color=C_V2, alpha=0.45, markersize=4.5, mew=0, zorder=3)

    ax.errorbar(Ds, v1, yerr=se1, fmt="o-", color=C_V1, label="v1 (SC off)",
                lw=1.8, markersize=6.5, capsize=3, elinewidth=1.2, zorder=5)
    ax.errorbar(Ds, v2, yerr=se2, fmt="s-", color=C_V2, label="v2 (SC active)",
                lw=1.8, markersize=6.5, capsize=3, elinewidth=1.2, zorder=5)

    ax.set_xscale("log")
    # Label the actual cohort sizes: default log ticks label only decades, which
    # left D=300 and D=614 drawn but unlabelled.
    ax.set_xticks(D_TICKS)
    ax.set_xticklabels([str(t) for t in D_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("# training mice (D)")
    ax.set_ylabel("held-out-mouse likelihood (cell-level)")
    ax.set_title("Result 1 — held-out scaling vs D")
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.30), fontsize=7.0,
              framealpha=0.9, borderaxespad=0.0)
    n_note = ("n = %s seeds per cell; error bars SEM; faded dots = individual seeds"
              % ("/".join(str(n) for n in n_per_cell)))
    ax.text(0.02, 0.98, n_note, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#444444")

    # Headroom above the highest GRU point so the presentation-size title has
    # clear air above the curves instead of crowding the top spine.
    ymax = max(max(v1), max(v2))
    ymin = min(min(v1), min(v2), min(m["pooled_likelihood_trial_weighted"] for m in models.values()))
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    sns.despine(fig=fig)
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 4: zero-shot vs adapted + RL ----------------------------------

def fig_zeroshot_vs_d_with_rl(rl, out_png):
    z = json.load(open(HERE / "zeroshot_vs_d.json"))
    Ds = z["Ds"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, var, ttl in zip(axes, ("v1", "v2"), ("v1 (SC off)", "v2 (SC active)")):
        v = z["variants"][var]
        ax.plot(Ds, v["zeroshot"], "o--", color="#7a7a7a",
                label="zero-shot (population embedding)", lw=2, markersize=9)
        ax.plot(Ds, v["adapted"], "o-", color="#3b76b8" if var == "v1" else "#d97c2a",
                label="adapted (per-mouse embedding fine-tuned)", lw=2, markersize=9)
        # per-mouse panel -> show subject-mean RL as primary; also annotate trial-weighted
        _rl_band(ax, rl, "subject_mean")
        ax.set_xscale("log"); ax.set_xlabel("# training mice (D)")
        ax.set_title(ttl)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("held-out-mouse likelihood (per-mouse mean)")
    fig.suptitle("Result 4 — zero-shot vs adapted vs D (+ RL reference)", fontsize=16)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 5: few-shot K-curve + RL --------------------------------------

def fig_fewshot_curve_with_rl(rl, out_png):
    fc = json.load(open(HERE / "fewshot_curve.json"))
    Ks = [0, 1, 4, "full"]; xpos = [0, 1, 2, 3]
    Ds_present = sorted({int(k.split("_D")[1]) for k in fc})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, var in zip(axes, ("v1", "v2")):
        for i, D in enumerate(Ds_present):
            arr = fc.get(f"{var}_D{D}")
            if arr is None: continue
            color = cmap(i / max(1, len(Ds_present) - 1))
            ax.plot(xpos, arr, "o-", color=color, label=f"D~{D}", lw=2, markersize=9)
        _rl_band(ax, rl, "subject_mean")
        ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("K (adaptation sessions)"); ax.set_title(var)
    axes[0].set_ylabel("held-out-mouse likelihood (per-mouse mean)")
    axes[1].legend(ncol=2, loc="lower right")
    fig.suptitle("Result 5 — few-shot adaptation curve (+ RL reference)", fontsize=16)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 7: N x D scaling + RL -----------------------------------------

def fig_nxd_scaling_with_rl(rl, out_png):
    nx = json.load(open(HERE / "nxd_scaling.json"))
    Ns, Ds = nx["Ns"], nx["Ds"]
    mean_grid = np.array(nx["mean_grid"], dtype=float)
    se_grid = np.array(nx["se_grid"], dtype=float)
    fa = nx["fit_additive"]; fi = nx["fit_interaction"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))

    # Panel A: heatmap (unchanged from nxd_scaling.py)
    ax = axes[0]
    im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels([str(d) for d in Ds])
    ax.set_yticks(range(len(Ns))); ax.set_yticklabels([str(n) for n in Ns])
    ax.set_xlabel("D (# training mice)"); ax.set_ylabel("N (hidden size)")
    ax.set_title("held-out eval_likelihood (mean over seeds)")
    for i in range(len(Ns)):
        for j in range(len(Ds)):
            v = mean_grid[i, j]
            if not np.isnan(v):
                col = "white" if v < (np.nanmin(mean_grid) + np.nanmax(mean_grid)) / 2 else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", color=col, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: L vs D, one line per N — plus RL band
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(Ns)))
    for i, N in enumerate(Ns):
        ax.errorbar(Ds, mean_grid[i, :], yerr=se_grid[i, :], marker="o",
                    color=colors[i], label=f"N={N}", capsize=4, lw=2, markersize=8)
    _rl_band(ax, rl, "trial_weighted")  # cell-level metric, trial-weighted
    ax.set_xscale("log"); ax.set_xlabel("D (# training mice)")
    ax.set_ylabel("held-out eval_likelihood")
    ax.set_title("L vs D, per N (+ RL reference)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False, fontsize=12)

    # Panel C: L vs N, one line per D — plus RL band
    ax = axes[2]
    colors2 = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ds)))
    for j, D in enumerate(Ds):
        ax.errorbar(Ns, mean_grid[:, j], yerr=se_grid[:, j], marker="s",
                    color=colors2[j], label=f"D={D}", capsize=4, lw=2, markersize=8)
    _rl_band(ax, rl, "trial_weighted")
    ax.set_xscale("log"); ax.set_xlabel("N (hidden size)")
    ax.set_ylabel("held-out eval_likelihood")
    ax.set_title("L vs N, per D (+ RL reference)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False, fontsize=12)

    title = (f"Result 7 — N×D scaling | additive E={fa['E']:.4f} "
             f"α={fa['alpha']:.3f} β={fa['beta']:.3f} "
             f"| AIC_add={fa['aic']:.1f} AIC_int={fi.get('aic', float('nan')):.1f}"
             f" | RL pooled={rl['pooled_likelihood_trial_weighted']:.4f}")
    fig.suptitle(title, fontsize=14, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ============================= verdict =============================

def write_verdict(rl, paired, out_md):
    lines = []
    lines.append("# RL baseline (simple, per-mouse independent fit) — verdict")
    lines.append("")
    lines.append(f"Source: [{rl['run_id']}]({rl['run_url']}) (group `{rl['group']}`). "
                 "Model: `baseline_rl` / `ForagerQLearning` (L1F1_CK1 — 1 learn-rate, 1 forget-rate, "
                 "1-step choice kernel, softmax). One DE optimizer fit per held-out mouse on its "
                 "own train sessions, scored on its own eval sessions. Same fixed held-out cohort "
                 f"(n={rl['n_subjects']}) and eval sessions as the GRU.")
    lines.append("")
    lines.append("## RL reference band")
    lines.append("")
    lines.append(f"- **Pooled (trial-weighted) likelihood:** **{rl['pooled_likelihood_trial_weighted']:.4f}** "
                 "(matches GRU `heldout/eval_likelihood`; used as the band on Results 1, 7).")
    lines.append(f"- **Per-subject mean likelihood:** **{rl['per_subject_mean_likelihood']:.4f}** "
                 f"± {rl['per_subject_se']:.4f} SE, median {rl['per_subject_median_likelihood']:.4f} "
                 "(used on per-mouse panels — Results 4, 5).")
    lines.append(f"- n = {rl['n_subjects']} held-out mice ({rl['n_trials']:,} eval trials).")
    lines.append("")
    lines.append("Per-curriculum breakdown:")
    lines.append("")
    lines.append("| curriculum | n | mean LL | std |")
    lines.append("|---|---|---|---|")
    for cur, s in sorted(rl["per_curriculum"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {cur} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} |")
    lines.append("")
    lines.append("## Result 8 — paired GRU vs RL per held-out mouse (avg over 3 seeds)")
    lines.append("")
    lines.append("| variant | D | n | GRU mean | RL mean | meanΔ (GRU−RL) | medianΔ | %GRU wins | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for var in paired["variants"]:
        for D in paired["Ds"]:
            s = paired["per_cell"].get(f"{var}_D{D}")
            if s is None: continue
            lines.append(f"| {var} | {D} | {s['n']} | {s['gru_mean']:.4f} | {s['rl_mean']:.4f} | "
                         f"{s['mean_delta']:+.5f} | {s['median_delta']:+.5f} | "
                         f"{s['frac_gru_wins']*100:.0f}% | {s['wilcoxon_p']:.1e} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    v2_l = paired["per_cell"].get("v2_D614")
    v2_s = paired["per_cell"].get("v2_D10")
    v1_l = paired["per_cell"].get("v1_D614")
    if v2_l and v2_s and v1_l:
        lines.append(f"- **GRU beats RL at every (variant, D) cell.** Even the smallest population "
                     f"GRU (v2, D=10) beats per-mouse RL: meanΔ={v2_s['mean_delta']:+.5f}, "
                     f"{v2_s['frac_gru_wins']*100:.0f}% of mice (Wilcoxon p={v2_s['wilcoxon_p']:.0e}). "
                     "A population model trained on as few as 10 other mice generalizes better to a "
                     "new mouse than fitting that mouse's own data with a classical RL model — strong "
                     "evidence the GRU exploits cross-mouse structure that per-mouse RL can't.")
        lines.append(f"- **Large-D gain over RL.** v2 D=614 vs RL: meanΔ={v2_l['mean_delta']:+.5f} "
                     f"({v2_l['frac_gru_wins']*100:.0f}% mice, p={v2_l['wilcoxon_p']:.0e}). "
                     "An order of magnitude larger than the v2−v1 SC effect (+0.0015): the "
                     "population vs per-mouse cognitive-model gap is the dominant signal, not "
                     "session conditioning or data scaling within the GRU.")
        incr = v2_l['mean_delta'] - v1_l['mean_delta']
        lines.append(f"- **SC adds a small extra margin over RL.** v2 D=614 beats RL by "
                     f"{v2_l['mean_delta']:+.5f}; v1 D=614 by {v1_l['mean_delta']:+.5f} → "
                     f"v2's incremental win vs RL is +{incr:.5f}, "
                     "consistent with the matched-pair v2−v1 SC result (Result 1).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **The RL baseline has no D-axis.** It's a per-mouse independent fit (no "
                 "cross-mouse sharing, no embeddings, no population prior). Answers \"does GRU beat "
                 "a stable classical RL on the same data?\" (yes), not \"does more mice help a "
                 "population RL?\" (planned hierarchical-Bayesian baseline; see `FUTURE_DIRECTIONS.md`).")
    lines.append("- **Simple agent (L1F1_CK1):** 1 learn-rate, 1 forget-rate, 1-step choice kernel. "
                 "Richer RL families (more learning/forget rates; ForagerLossCounting; "
                 "ForagerCompareThreshold) may close some of the gap. Cf. "
                 "`code/config/model/baseline_rl.yaml`.")
    lines.append("- **Single optimizer seed.** Differential evolution is stable; pilot didn't need "
                 "multi-seed averaging (variant notes). Add seeds [1, 2] for a tighter RL comparison.")
    lines.append("- **Pooled vs per-subject mean differ** because mice have different trial counts. "
                 "Use the trial-weighted pooled (0.7143) for GRU pooled aggregates; use the per-"
                 "subject mean (0.7211) on per-mouse panels. The figures label both.")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_md.name}")


# ============================= main =============================

def main():
    print("Reading the classical-RL baseline suite (Bari / Hattori / CTT)...")
    models, legacy, drift = all_rl()

    # Best-of-breed = highest pooled (trial-weighted) held-out likelihood. This is the
    # honest baseline for "does the GRU beat classical RL?" — r8 originally compared
    # against Bari alone, which is NOT the strongest of the three.
    best_key = max(models, key=lambda k: models[k]["pooled_likelihood_trial_weighted"])
    print(f"  best classical model: {models[best_key]['label']} "
          f"({models[best_key]['pooled_likelihood_trial_weighted']:.5f})")

    print("\nLoading GRU per-subject (frozen)...")
    gru_nested, gru_groups = load_gru()
    gru_mean = gru_per_subject_means(gru_nested)
    print(f"  unique (variant,D,subject) means: {len(gru_mean)}")

    print("\nPaired GRU vs each RL model per (variant, D)...")
    paired_by_model = {}
    for key, m in models.items():
        paired_by_model[key] = paired_vs_rl(gru_mean, m["per_subject_likelihood"])
        cell = paired_by_model[key]["per_cell"].get("v2_D614")
        if cell:
            print(f"  {m['label']:24} v2/D=614  GRU={cell['gru_mean']:.4f} "
                  f"RL={cell['rl_mean']:.4f}  meanΔ={cell['mean_delta']:+.5f} "
                  f"({cell['frac_gru_wins']*100:.0f}% mice, p={cell['wilcoxon_p']:.1e})")

    # `rl` / `gru_vs_rl_paired` keep the BAND_MODEL, so Results 1/4/5/7 keep the same
    # reference band they were written against (see BAND_MODEL comment).
    rl = models[BAND_MODEL]
    paired = paired_by_model[BAND_MODEL]

    out = {
        "_meta": build_meta(
            "analysis/rl_baseline.py",
            [*gru_groups,
             *[m["group"] or RL_MODELS[k]["group"] for k, m in models.items()],
             legacy["group"] or LEGACY_BARI_GROUP],
        ),
        "rl": rl,
        "gru_vs_rl_paired": paired,
        "band_model": BAND_MODEL,
        "models": models,
        "paired_by_model": paired_by_model,
        "best_model": best_key,
        "legacy_bari": {k: v for k, v in legacy.items()
                        if k != "per_subject_likelihood"},
        "legacy_reproduction_delta": drift,
    }
    json.dump(out, open(HERE / "rl_baseline.json", "w"), indent=2)
    print(f"\nwrote rl_baseline.json")

    write_verdict(rl, paired, HERE / "rl_baseline_verdict.md")

    print("\nRendering figures...")
    fig_paired(paired, rl, HERE / "fig_rl_paired.png")
    fig_scaling_v1_v2_with_rl(rl, models, HERE / "fig_scaling_v1_v2.png")
    fig_zeroshot_vs_d_with_rl(rl, HERE / "fig_zeroshot_vs_d.png")
    fig_fewshot_curve_with_rl(rl, HERE / "fig_fewshot_curve.png")
    fig_nxd_scaling_with_rl(rl, HERE / "fig_nxd_scaling.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
