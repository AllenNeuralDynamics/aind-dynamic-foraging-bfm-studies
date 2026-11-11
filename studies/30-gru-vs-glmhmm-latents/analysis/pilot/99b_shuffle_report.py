"""Summarise the shuffle nulls against the real K-scan decoding: figure + JSON + table.

Real values from out/kscan/parts/hurdle__K<K>__<fset>.json; nulls from out/shuffle/parts/.
Panels: pooled R^2 (all trials), pooled R^2 (within session), information recovered; x = K; latents and history;
real = large marker, each null shuffle = small marker (swap and circular shift side by side).
"""
import json
import sys
from pathlib import Path

import numpy as np

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

KS = [2, 3, 6, 10]
METRICS = [("pooled_r2", "pooled R², all trials"), ("pooled_r2_within", "pooled R², within session"),
           ("info_recovered", "information recovered")]
COL = {"latents": "#1b9e77", "history": "#7570b3"}
NULL_DX = {"swap": -0.12, "circ": 0.12}


def main():
    res = {}
    for fset in ("latents", "history"):
        for K in KS:
            real = json.loads((OUT / "kscan" / "parts" / f"hurdle__K{K}__{fset}.json").read_text())
            entry = {"real": {m: real[m] for m, _ in METRICS}}
            for null in ("swap", "circ"):
                vals = [json.loads(p.read_text()) for p in sorted((OUT / "shuffle" / "parts").glob(f"{fset}_{null}_K{K}_s*.json"))]
                entry[null] = {m: [v[m] for v in vals] for m, _ in METRICS}
                entry[null]["n"] = len(vals)
            res[f"{fset}_K{K}"] = entry
    (OUT / "shuffle" / "shuffle_results.json").write_text(json.dumps(res, indent=1))
    print(f"{'set':8s} {'K':>2s} {'real':>6s} | {'swap mean (max)':>16s} | {'circ mean (max)':>16s}   [pooled R² all; within]")
    for key, e in res.items():
        fset, K = key.split("_K")
        line = f"{fset:8s} {K:>2s} "
        for m in ("pooled_r2", "pooled_r2_within"):
            sw, ci = e["swap"][m], e["circ"][m]
            f = lambda v: f"{np.mean(v):6.3f} ({np.max(v):.3f})" if v else "   n/a"  # noqa: E731
            line += f"{e['real'][m]:6.3f} | {f(sw):>16s} | {f(ci):>16s} ;  "
        print(line, f"n={e['swap']['n']},{e['circ']['n']}")

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(150, 48))
    x = np.arange(len(KS))
    for ax, (m, title) in zip(axes, METRICS):
        for fset, off in (("latents", -0.2), ("history", 0.2)):
            ax.plot(x + off, [res[f"{fset}_K{K}"]["real"][m] for K in KS], "o", color=COL[fset], ms=4,
                    label=f"{fset}, real")
            for null, mk in (("swap", "x"), ("circ", "+")):
                for i, K in enumerate(KS):
                    v = res[f"{fset}_K{K}"][null][m]
                    ax.plot(np.full(len(v), i + off + NULL_DX[null] * 0.6), v, mk, color=COL[fset], ms=2.5, mew=0.5,
                            alpha=0.8, label=f"{fset}, {'session swap' if null == 'swap' else 'circular shift'}"
                            if i == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x, [str(K) for K in KS])
        ax.set_xlabel("K (number of states)")
        ax.set_title(title, fontsize=6)
        ax.set_ylim(-0.1, 1)
    axes[0].set_ylabel("score (out of fold)")
    axes[2].legend(frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.text(0.01, 0.01, "Hurdle GLM-HMM predictive posterior; same probe, folds and features as the K scan; decoder trained "
             "and scored on shuffled pairs. Session swap: target from another mouse's session; circular shift: target "
             "rotated 20–80 % within its session. 50 training mice. Scratch pilot.", fontsize=5)
    fig.subplots_adjust(left=0.08, right=0.8, bottom=0.27, top=0.88, wspace=0.35)
    fs.save_figure(fig, OUT / "shuffle" / "fig_shuffle_null.svg")


if __name__ == "__main__":
    main()
