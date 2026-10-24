#!/usr/bin/env python
"""Regenerate the ignore-trials scaling figure from the frozen grid + the 2-way baseline grid.

Offline: reads ``frozen_runs.json`` (one row per kept run, written by ``freeze_runs.py``,
which is the only step that needs W&B) and overlays the frozen 2-way grid for reference.
To take in newly landed cells, run ``make freeze`` first and commit the refreshed file.

    python scaling.py                            # writes fig_scaling.png, scaling.json/.csv

Outputs (next to this script):
    ignore_scaling.png   3-panel figure: LR-engaged vs D, ignore PR-AUC/recall vs D
    ignore_scaling_3way.csv   the live 3-way aggregate table it plotted

The 3-way grid lives in W&B project mice_ignore_scaling across these sweeps:
    o9oq3j4y  narrow  H16/64/128
    9iu2rcap  wide    H256
    95br9evz  rerun   H16/D10/seed2  (collision-replacement)
    4aq30mvb  rerun   H256/D614      (256G OOM-replacement)
The 2-way reference grid is 01-gru-scaling-law/analysis/nxd_scaling.json
(metric heldout/final/eval_likelihood, same fixed held-out set, same D×H layout).
"""
import sys, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                 # studies/02-gru-scaling-law-ignore/analysis
STUDY = HERE.parent                                     # studies/02-gru-scaling-law-ignore
TWO_WAY_JSON = STUDY.parent / "01-gru-scaling-law" / "analysis" / "nxd_scaling.json"
sys.path.insert(0, str(HERE))                           # for wandb_keys / update_reports
sys.path.insert(0, str(STUDY.parent / "util"))          # shared studies/util/_meta.py
from _meta import build_meta                            # noqa: E402
# HPC runs land in these named sweeps; Beaker runs (nxd-3way-<timestamp>-<hash>)
# carry no sweepName, so we no longer gate on sweep membership at all -- instead
# we require the run to have a resolved (D,H,seed) + a finished held-out metric.
SWEEPS = {"o9oq3j4y", "9iu2rcap", "95br9evz", "4aq30mvb"}
ENTITY, PROJECT = "AIND-disRNN", "mice_ignore_scaling"

# The W&B groups this study is built from. This allowlist is the ONLY source of
# runs: a run outside it is ignored no matter how well it matches (D,H,seed).
# Without it the project-wide query below silently absorbs ANY new run that
# happens to fit the grid -- a 200-step smoke run moved the published H16 mean
# from 0.71635 to 0.70348 (SEM 5.3e-5 -> 1.3e-2) during the 2026-07-11 skills
# validation. Pinning here matches studies 01/03/04 and makes the report
# reproducible from committed code + pinned groups. Add a group when a launch
# legitimately extends the grid.
WANDB_GROUPS = [
    "nxd-3way@20260704-205230",
    "nxd-3way@20260704-205239",
    "nxd-3way@20260707-075359",
    "nxd-3way@20260707-075401",
    "nxd-3way@20260707-075406",
    "nxd-3way@20260707-075412",
    "nxd-3way@20260707-110227",
    "nxd-3way@20260707-110245",
    "nxd-3way@20260708-231204",
    "nxd-3way@20260708-231938",
    "nxd-3way@20260709-231516",
    "rerun_H16D10s2@20260705-104925",
    "sweep-validate-capnsteps@20260710-183333",
]
SR_TO_D = {0.016: 10, 0.049: 30, 0.163: 100, 1.0: 614}
HS = [16, 64, 128, 256]


def _get_nested(cfg, *paths):
    """W&B config values are either flat dotted keys (HPC launcher_cmd runs) or
    nested dicts (Beaker Hydra-structured config). Try both."""
    for path in paths:
        cur = cfg
        ok = True
        for p in path.split("."):
            if isinstance(cur, dict) and "value" in cur and set(cur.keys()) <= {"value", "desc"}:
                cur = cur["value"]
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            if isinstance(cur, dict) and "value" in cur:
                cur = cur["value"]
            if cur is not None:
                return cur
    return None

# ---- shared style + stats helpers (studies/util/plot_style.py) -------------
# Presentation-grade rcParams, the per-hidden-size palette, and the t* helper
# all live in studies/util so every study's figures match. The FIGURE draws SEM
# (Han 2026-07-11, n noted in footer); t975()*sem populates the *_ci95 columns
# in the curated JSON/CSV but is not drawn.
from plot_style import apply_presentation_style, HCOLOR, t975 as _t975  # noqa: E402
apply_presentation_style()


FROZEN_RUNS = HERE / "frozen_runs.json"   # written by freeze_runs.py (needs W&B)
METRIC_KEYS = ["eval_likelihood_LR_engaged", "eval_likelihood_engage", "eval_likelihood",
               "engage_ignore_f1", "engage_ignore_recall", "engage_ignore_precision",
               "engage_ignore_pr_auc", "engage_ignore_base_rate"]


def load_3way():
    """The frozen 3-way grid: metric -> (D,H) -> [seed values], plus the groups used.

    One row per (D,H,seed), already deduplicated by freeze_runs.py; rows keep the
    order the pull returned, so seed lists come out in the same order every time.
    """
    rows = json.loads(FROZEN_RUNS.read_text())["runs"]
    out = {k: defaultdict(list) for k in METRIC_KEYS}
    for r in rows:
        for k in METRIC_KEYS:
            v = r[k]
            if isinstance(v, (int, float)):
                out[k][(r["D"], r["H"])].append(v)
    return out, sorted({r["wandb_group"] for r in rows if r["wandb_group"]})


def load_2way():
    d = json.load(open(TWO_WAY_JSON))
    Ns, Ds, grid, se = d["Ns"], d["Ds"], d["mean_grid"], d["se_grid"]
    mean = {(Ds[j], Ns[i]): grid[i][j] for i in range(len(Ns)) for j in range(len(Ds))}
    sem = {(Ds[j], Ns[i]): se[i][j] for i in range(len(Ns)) for j in range(len(Ds))}
    return mean, sem


def agg(metric_map):
    """(D,H)->[vals] to (D,H)->(mean, sem, ci95_halfwidth, n, raw_list).

    ci95_halfwidth = t_{0.975,n-1} * sem (t-distribution, per point). The drawn
    ci95_halfwidth is retained (t*sem) but the figure draws SEM (project request
    2026-07-11, n noted in footer); both sem and ci95 are written to the CSV.
    """
    o = {}
    for k, v in metric_map.items():
        a = np.array(v, float)
        n = len(a)
        sem = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci = _t975(n) * sem
        o[k] = (a.mean(), sem, ci, n, a.tolist())
    return o


def line_for_H(ax, stats, H, Ds_all, show_raw=True, **kw):
    """Draw mean line with SEM error bars (n noted in footer); overlay jittered raw
    per-seed dots offset to the LEFT of each mean marker (project convention:
    raw dots must not sit on top of the summary glyph)."""
    xs, ys, es = [], [], []
    for D in Ds_all:
        if (D, H) in stats:
            m, s, ci, n, raw = stats[(D, H)]
            xs.append(D); ys.append(m); es.append(s)  # SEM (n noted in footer)
            if show_raw and n > 0:
                # multiplicative offset in log-x: place dots ~7% left of the marker
                xr = [D * 0.93 for _ in raw]
                ax.scatter(xr, raw, s=16, color=HCOLOR[H], alpha=0.38,
                           edgecolors="none", zorder=2)
    if xs:
        ax.errorbar(xs, ys, yerr=es, color=HCOLOR[H], **kw)
    return xs, ys


def main():
    three, groups = load_3way()
    lr3 = agg(three["eval_likelihood_LR_engaged"])
    prauc3 = agg(three["engage_ignore_pr_auc"])
    rec3 = agg(three["engage_ignore_recall"])
    two_mean, two_sem = load_2way()
    Ds_all = [10, 30, 100, 614]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- Panel A: LR-engaged vs D, 3-way (solid) vs 2-way (faded dashed) ----
    axA = axes[0]
    for H in HS:
        # 2-way reference (faded, dashed, thin)
        xs2 = [D for D in Ds_all if (D, H) in two_mean]
        ys2 = [two_mean[(D, H)] for D in xs2]
        if xs2:
            axA.plot(xs2, ys2, color=HCOLOR[H], ls="--", lw=1.4, alpha=0.45, zorder=1)
        # 3-way (solid, marker, SEM error bar + offset raw dots)
        line_for_H(axA, lr3, H, Ds_all, ls="-", lw=2.4, marker="o", ms=8,
                   capsize=5, capthick=1.8, zorder=3, label=f"H={H}")
    axA.set_xscale("log")
    axA.set_xticks(Ds_all); axA.set_xticklabels([str(d) for d in Ds_all])
    axA.set_xlabel("# training mice (D)")
    axA.set_ylabel("held-out L/R likelihood (engaged trials)")
    axA.set_title("3-way model matches 2-way choice ceiling\nand scales with D (solid=3-way, faded=2-way)")
    axA.margins(0.06)
    axA.legend(title="hidden size", frameon=False, loc="lower right")

    # ---- Panel B: ignore PR-AUC vs D (per H) ----
    axB = axes[1]
    for H in HS:
        xs, ys = line_for_H(axB, prauc3, H, Ds_all, ls="-", lw=2.4, marker="s", ms=8,
                            capsize=5, capthick=1.8, label=f"H={H}")
    axB.set_xscale("log")
    axB.set_xticks(Ds_all); axB.set_xticklabels([str(d) for d in Ds_all])
    axB.set_xlabel("# training mice (D)")
    axB.set_ylabel("ignore-class PR-AUC (held-out)")
    axB.set_title("Ignore detection is well above chance\nand improves with D")
    axB.margins(0.35)
    # Zoomed to the data's own scale (PR-AUC ~0.61-0.64) so the trend across D
    # is visible; the no-skill base rate (~0.05-0.10) is far below this range
    # and is only noted in text, not spanned to (which would flatten the plot).
    axB.text(0.03, 0.03, "no-skill (base rate) \u2248 0.05\u20130.10, off-scale below",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=12, color="0.4")

    # ---- Panel C: ignore recall vs D (the base-rate-hedging check) ----
    axC = axes[2]
    for H in HS:
        line_for_H(axC, rec3, H, Ds_all, ls="-", lw=2.4, marker="^", ms=8,
                   capsize=5, capthick=1.8, label=f"H={H}")
    axC.axhline(0.5, color="0.6", ls=":", lw=1.4, zorder=0)
    axC.text(0.03, 0.5, "recall = 0.5 (half of ignores caught)",
             transform=axC.get_yaxis_transform(), ha="left", va="bottom",
             fontsize=12, color="0.4")
    axC.set_xscale("log")
    axC.set_xticks(Ds_all); axC.set_xticklabels([str(d) for d in Ds_all])
    axC.set_xlabel("# training mice (D)")
    axC.set_ylabel("ignore-class recall (held-out)")
    axC.set_title("Recall stays below 0.5: model misses\n>half the ignores, but improves with D")
    axC.set_ylim(top=0.505)
    axC.margins(x=0.06)

    # Error-bar convention note (SEM; n noted here per project request) + raw dots.
    fig.text(0.5, 0.005,
             "Error bars: SEM (n = 3 seeds per cell). "
             "Faded dots: individual seeds, offset left of the mean marker.",
             ha="center", va="bottom", fontsize=12, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png = HERE / "fig_scaling.png"
    fig.savefig(out_png, bbox_inches="tight")
    print("wrote", out_png)

    # ---- curated JSON (carries _meta; the source of truth for report numbers) ----
    # Each metric family carries its OWN seed count: n_LR (LR-engaged, complete
    # grid-wide) and n_ignore (pr_auc/recall, same runs). We store sem AND the
    # t-based 95% CI half-width per point (figure draws SEM; ci95 retained).
    nan = float("nan")
    grid = {}
    for H in HS:
        for D in Ds_all:
            if (D, H) not in lr3:
                continue
            m, s, ci, n_lr, raw_lr = lr3[(D, H)]
            pa, pas, paci, n_pa, _ = prauc3.get((D, H), (nan, nan, nan, 0, []))
            rc, rcs, rcci, n_rc, _ = rec3.get((D, H), (nan, nan, nan, 0, []))
            grid[f"{D}x{H}"] = {
                "D": D, "H": H,
                "LR_engaged": {"mean": m, "sem": s, "ci95": ci, "n": n_lr,
                               "seeds": [round(x, 6) for x in raw_lr]},
                "ignore_pr_auc": {"mean": pa, "sem": pas, "ci95": paci, "n": max(n_pa, n_rc)},
                "ignore_recall": {"mean": rc, "sem": rcs, "ci95": rcci, "n": max(n_pa, n_rc)},
            }
    # `groups` are those that contributed a kept run: a subset of the pinned
    # WANDB_GROUPS allowlist by construction (freeze_runs.py filters on it).
    out = {
        "_meta": build_meta("analysis/scaling.py", groups, study_root=STUDY),
        "metric_note": (
            "3-way (L/R/ignore) held-out finetune. LR_engaged = conditional L/R "
            "likelihood on engaged trials (comparable to 2-way, chance 1/2). "
            "ignore_* = ignore-class detection (rare positive, base rate ~0.05-0.10). "
            "Raw 3-way NL (chance 1/3) is NOT comparable to 2-way and is not tabulated here. "
            "Error bars in fig_scaling.png are SEM (n=3 seeds/cell)."
        ),
        "Hs": HS, "Ds": Ds_all,
        "grid": grid,
    }
    (HERE / "scaling.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", HERE / "scaling.json")

    # convenience flat CSV (same numbers, easy to eyeball / spreadsheet)
    rows = ["D,H,n_LR,LR_engaged,LR_engaged_sem,LR_engaged_ci95,"
            "n_ignore,pr_auc,pr_auc_sem,pr_auc_ci95,recall,recall_sem,recall_ci95"]
    for H in HS:
        for D in Ds_all:
            c = grid.get(f"{D}x{H}")
            if not c:
                continue
            lr, ig, rr = c["LR_engaged"], c["ignore_pr_auc"], c["ignore_recall"]
            rows.append(f"{D},{H},{lr['n']},{lr['mean']:.5f},{lr['sem']:.5f},{lr['ci95']:.5f},"
                        f"{ig['n']},{ig['mean']:.5f},{ig['sem']:.5f},{ig['ci95']:.5f},"
                        f"{rr['mean']:.5f},{rr['sem']:.5f},{rr['ci95']:.5f}")
    (HERE / "scaling.csv").write_text("\n".join(rows) + "\n")
    print("wrote", HERE / "scaling.csv")

    # regenerate the script-owned regions of the reports
    try:
        import update_reports
        update_reports.run(out)
        print("updated reports/")
    except Exception as exc:  # reporting is best-effort; JSON+fig are the artifacts
        print("WARN: report regen skipped:", exc)
    return fig


if __name__ == "__main__":
    main()
