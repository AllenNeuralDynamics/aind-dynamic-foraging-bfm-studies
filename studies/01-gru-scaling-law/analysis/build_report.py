#!/usr/bin/env python
"""Build the v1-vs-v2 data-scaling held-out report: cell-level paired test (loaded),
per-held-out-subject repeated-measures (from the offline heldout-rerun W&B tables, frozen
into per_subject_rows.json by freeze_per_subject.py). Writes report_data.json + figures
(feeds reports/r1, r2). Offline.
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _meta import build_meta

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
# offline per-subject re-run groups (v1 + v2, incl. retries), pinned (posthoc-reporting):
# freeze_per_subject.py reads only these.
WANDB_GROUPS = [
    "heldout-rerun-v1-retry@20260623-011405",
    "heldout-rerun-v1-retry@20260623-011406",
    "heldout-rerun-v1@20260623-004601",
    "heldout-rerun-v1@20260623-005608",
    "heldout-rerun-v2-retry@20260623-065818",
    "heldout-rerun-v2@20260623-053951",
]

FROZEN = HERE / "per_subject_rows.json"   # written by freeze_per_subject.py (needs W&B)


def load_per_subject():
    """Per-held-out-subject rows (variant, ratio, seed, subject, n_trials, ll) and the
    W&B groups they came from, from the frozen pull."""
    frozen = json.loads(FROZEN.read_text())
    return frozen["rows"], frozen["_meta"]["wandb_groups"]

def per_subject_analysis(rows):
    # average over seeds -> (variant, ratio, subject) mean ll
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["ratio"], r["subject"])].append(r["ll"])
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    ratios = sorted({r["ratio"] for r in rows})
    out = {"per_ratio": {}, "coverage": {}}
    for ratio in ratios:
        subs = sorted({k[2] for k in mean if k[1] == ratio and k[0] == "v1"}
                      & {k[2] for k in mean if k[1] == ratio and k[0] == "v2"})
        d = np.array([mean[("v2", ratio, s)] - mean[("v1", ratio, s)] for s in subs])
        if len(d) >= 5:
            w = stats.wilcoxon(d)
            out["per_ratio"][str(ratio)] = dict(n_subjects=len(d), mean_delta=float(d.mean()),
                                                median_delta=float(np.median(d)),
                                                frac_positive=float((d > 0).mean()),
                                                wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue))
        out["coverage"][str(ratio)] = len(subs)
    return out, mean, ratios

def make_figures(cell, persub, mean, ratios):
    # ratio -> approx D label
    Dlab = {0.016:10, 0.049:30, 0.163:100, 0.489:300, 1.0:614}
    # Fig A: scaling curves v1 vs v2 (cell-level aggregate from paired json)
    pr = cell["per_ratio"]
    xs = sorted(float(k) for k in pr)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([Dlab.get(round(x,3), x) for x in xs], [pr[str(x)]["v1"] for x in xs], "o-", label="v1 (SC off)")
    ax.plot([Dlab.get(round(x,3), x) for x in xs], [pr[str(x)]["v2"] for x in xs], "s-", label="v2 (SC active)")
    ax.set_xscale("log"); ax.set_xlabel("# training mice (D)"); ax.set_ylabel("held-out-mouse likelihood")
    ax.set_title("Held-out generalization vs D"); ax.legend()
    fig.tight_layout(); fig.savefig(HERE/"fig_scaling_v1_v2.png", dpi=150); plt.close(fig)
    # Fig B: per-D delta (v2-v1) with per-subject spread (violin) where available
    fig, ax = plt.subplots(figsize=(6, 4))
    data, labels = [], []
    for ratio in ratios:
        subs = sorted({k[2] for k in mean if k[1]==ratio and k[0]=="v1"} & {k[2] for k in mean if k[1]==ratio and k[0]=="v2"})
        if len(subs) >= 5:
            data.append([mean[("v2",ratio,s)]-mean[("v1",ratio,s)] for s in subs]); labels.append(f"D~{Dlab.get(ratio,ratio)}")
    if data:
        ax.violinplot(data, showmeans=True); ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("per-held-out-mouse Δ likelihood (v2−v1)"); ax.set_title("SC effect per held-out mouse, by D")
        fig.tight_layout(); fig.savefig(HERE/"fig_per_subject_delta.png", dpi=150); plt.close(fig)
    return ["fig_scaling_v1_v2.png", "fig_per_subject_delta.png"]

def main():
    cell = json.load(open(HERE/"paired_v1_v2_cell.json"))
    rows, groups = load_per_subject()
    print(f"collected {len(rows)} per-subject rows")
    persub, mean, ratios = per_subject_analysis(rows)
    figs = make_figures(cell, persub, mean, ratios)
    out = {
        "_meta": build_meta("analysis/build_report.py", groups),
        "cell_level": cell,
        "per_subject": persub,
    }
    json.dump(out, open(HERE/"report_data.json","w"), indent=2)
    # summary print
    print("\n=== per-subject repeated-measures (v2-v1, paired by held-out mouse, avg over seeds) ===")
    for r, s in persub["per_ratio"].items():
        print(f"  ratio={r}: n={s['n_subjects']} meanΔ={s['mean_delta']:+.5f} medΔ={s['median_delta']:+.5f} "
              f"frac+={s['frac_positive']:.2f} Wilcoxon p={s['wilcoxon_p']:.2e}")
    return cell, persub, figs

if __name__ == "__main__":
    main()
