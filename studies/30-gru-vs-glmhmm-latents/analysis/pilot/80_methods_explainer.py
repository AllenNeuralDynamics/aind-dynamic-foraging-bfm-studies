"""Stage 8: a methods figure that walks through the decoding, step by step, on real pilot data.

a  pipeline schematic
b  one example session (behaviour)                       f  cross-validation: 5 folds of 10 mice
c  step 1, target: GLM-HMM predictive posterior          g  step 4, score: latents, all trials
d  step 2, features: the GRU latents (256 units)         h  step 4, score: latents, within session
e  step 3, probe output (out of fold) vs target          i  step 4, score: history probe, all trials

Reads only pilot outputs. Scratch figure, paper style (studies/util/figure_style.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDIES = Path("/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
               "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
sys.path.insert(0, str(STUDIES))
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
SESSION = "729674_2024-10-03_135356"          # the pilot's example session (most posterior movement)
STATE = 1                                       # state used for the scoring scatters
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
NAMES = ["S0 ignore-prone", "S1 engaged, strong history", "S2 engaged, weak history"]


def r2(p, q):
    return 1 - ((p - q) ** 2).sum() / ((p - p.mean()) ** 2).sum()


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec="k", lw=0.5, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", shrinkA=0, shrinkB=0))


def schematic(ax):
    ax.axis("off")
    box(ax, 0.00, 0.25, 0.12, 0.50, "choices and\nrewards,\ntrials 1 … t−1", "#eeeeee")
    box(ax, 0.19, 0.58, 0.16, 0.40, "trained GRU\n(study 02, H = 256)", "#dde6f3")
    box(ax, 0.41, 0.58, 0.15, 0.40, "latents h_t\n(256 numbers)", "#dde6f3")
    box(ax, 0.62, 0.58, 0.19, 0.40, "linear probe\nπ̂_t = softmax(W·h_t + b)", "#fbe3d0")
    box(ax, 0.19, 0.02, 0.16, 0.40, "GLM-HMM\n(study 14, hurdle K = 3)", "#e3f0e3")
    box(ax, 0.41, 0.02, 0.40, 0.40, "target π_t = p(z_t | trials < t)\none-step predictive state posterior", "#e3f0e3")
    box(ax, 0.87, 0.25, 0.13, 0.50, "score\nR²(π_t, π̂_t)\non held-out mice", "#f5f5f5")
    for (x0, y0, x1, y1) in [(0.12, 0.60, 0.19, 0.78), (0.12, 0.40, 0.19, 0.22), (0.35, 0.78, 0.41, 0.78),
                             (0.56, 0.78, 0.62, 0.78), (0.35, 0.22, 0.41, 0.22), (0.81, 0.78, 0.87, 0.62),
                             (0.81, 0.22, 0.87, 0.38)]:
        arrow(ax, x0, y0, x1, y1)
    ax.text(0.715, 0.50, "fit W, b on 40 mice, predict the other 10", ha="center", va="center",
            transform=ax.transAxes, fontsize=5, style="italic")


def stacked(ax, t, P, title):
    ax.stackplot(t, P.T, colors=COLORS, alpha=0.85, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("p(state)")
    ax.set_title(title, loc="left")


def main():
    fs.apply_paper_style()
    lat_cols = [f"latent_{i}" for i in range(256)]
    base = ["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward", "stage",
            "reward_probabilityL", "reward_probabilityR", "choice_prob_0", "choice_prob_1",
            "pred_0", "pred_1", "pred_2"]
    j = pd.read_parquet(OUT / "joined_trials.parquet", columns=base + lat_cols)
    oof = pd.read_parquet(OUT / "probe_oof_predictions.parquet")
    for d in (j, oof):
        d["subject_id"] = d["subject_id"].astype(str)
        d["ses_idx"] = d["ses_idx"].astype(str)
    m = j.merge(oof, on=["subject_id", "ses_idx", "trial"], validate="one_to_one")
    assert len(m) == len(j)
    fold_of_mouse = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    sid = [s for s in m["ses_idx"].unique() if SESSION in s]
    assert len(sid) == 1, sid
    g = m[m["ses_idx"] == sid[0]].sort_values("trial")
    mouse = g["subject_id"].iloc[0]
    test_fold = fold_of_mouse[mouse]
    t = g["trial_in_session"].to_numpy()
    P = g[[f"pred_{k}" for k in range(3)]].to_numpy()
    Qlat = g[[f"probe_latents_{k}" for k in range(3)]].to_numpy()

    # Order units by which state they track best, measured on the four training folds only.
    train = m[m["subject_id"].map(fold_of_mouse) != test_fold].sample(100_000, random_state=0)
    Ltr = train[lat_cols].to_numpy(np.float64)
    Ltr = (Ltr - Ltr.mean(0)) / (Ltr.std(0) + 1e-9)
    C = np.stack([np.corrcoef(Ltr.T, train[f"pred_{k}"].to_numpy())[-1, :-1] for k in range(3)])
    best = np.abs(C).argmax(0)
    order = np.lexsort((-C[best, np.arange(256)], best))
    mu, sd = train[lat_cols].to_numpy().mean(0), train[lat_cols].to_numpy().std(0) + 1e-9
    Z = ((g[lat_cols].to_numpy() - mu) / sd)[:, order].T

    fig = plt.figure(figsize=fs.mm(183, 175))
    gs = fig.add_gridspec(5, 3, width_ratios=[1, 1, 0.62], height_ratios=[0.9, 1.15, 0.8, 1.0, 0.8],
                          hspace=0.75, wspace=0.45, left=0.07, right=0.98, top=0.97, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    schematic(ax_a)
    fs.panel_label(ax_a, "a", dx_pt=-14, dy_pt=0)

    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    ax_b = fig.add_subplot(gs[1, :2])
    choice = np.where(g["choice"].to_numpy() == 2, np.nan, g["choice"].to_numpy().astype(float))
    _, sess_axes = plot_foraging_session(choice, g["reward"].to_numpy(),
                                         [g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()],
                                         ax=ax_b, plot_list=["choice", "reward_prob"])
    ax_b = sess_axes[0]
    for a in sess_axes:
        if a.get_legend():
            a.get_legend().remove()
    ax_b.set_title(f"Example session (mouse {mouse}, {g['stage'].iloc[0].lower()}, test fold {test_fold + 1}): "
                   "choices, rewards, block reward probabilities; x = ignored", loc="left")
    for a in sess_axes:
        a.set_xlabel("")
    fs.panel_label(ax_b, "b")
    xlim = (t[0] - 0.5, t[-1] + 0.5)

    ax_c = fig.add_subplot(gs[2, :2])
    stacked(ax_c, t, P, "Step 1 · target: GLM-HMM predictive posterior π_t = p(z_t | trials < t)")
    fs.panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[3, :2])
    ax_d.imshow(np.clip(Z, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                extent=[t[0] - 0.5, t[-1] + 0.5, 256, 0], interpolation="nearest")
    edges = np.cumsum(np.bincount(best, minlength=3))
    for k, (lo, hi) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
        ax_d.add_patch(Rectangle((t[0] - 12, lo), 8, hi - lo, color=COLORS[k], clip_on=False))
        if k < 2:
            ax_d.axhline(hi, color="k", lw=0.4)
    ax_d.set_ylabel("GRU unit")
    ax_d.set_title("Step 2 · features: GRU latents h_t, 256 units (z-scored), grouped by the state each tracks best", loc="left")
    fs.panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[4, :2])
    for k in range(3):
        ax_e.plot(t, P[:, k], color=COLORS[k], lw=1.0)
        ax_e.plot(t, Qlat[:, k], color="k", lw=0.5, ls=(0, (2, 1)))
        ax_e.plot(t, Qlat[:, k], color=COLORS[k], lw=0.5, ls=(1, (2, 1)))
    ax_e.set_ylim(-0.02, 1.02)
    ax_e.set_ylabel("p(state)")
    ax_e.set_xlabel("Trial in session")
    ax_e.set_title("Step 3 · probe output: solid = target π_t, dashed = decoded π̂_t from h_t "
                   "(probe never saw this mouse)", loc="left")
    fs.panel_label(ax_e, "e")
    for a in (ax_c, ax_d, ax_e):
        a.set_xlim(xlim)

    # f: cross-validation layout
    ax_f = fig.add_subplot(gs[1, 2])
    mice = sorted(fold_of_mouse, key=lambda s: (fold_of_mouse[s], s))
    for f in range(5):
        for i in range(10):
            is_test = f == test_fold
            ax_f.add_patch(Rectangle((i, 4 - f), 0.85, 0.8, fc="#DD8452" if is_test else "#cccccc", ec="none"))
        ax_f.text(-0.4, 4 - f + 0.4, f"fold {f + 1}", ha="right", va="center", fontsize=5)
    pos = [s for s in mice if fold_of_mouse[s] == test_fold].index(mouse)
    ax_f.add_patch(Rectangle((pos, 4 - test_fold), 0.85, 0.8, fc="none", ec="k", lw=0.8))
    ax_f.set_xlim(-3.2, 10)
    ax_f.set_ylim(-3.6, 5.2)
    ax_f.axis("off")
    ax_f.set_title("Cross-validation by mouse: 5 folds of 10.\nFit on 4 folds (grey), score the 5th\n(orange), repeat 5×. Box: example mouse", loc="left", fontsize=5)
    fs.panel_label(ax_f, "f", dx_pt=-6)
    for k in range(3):
        ax_f.add_patch(Rectangle((-3.0, -1.4 - 0.9 * k), 0.8, 0.6, fc=COLORS[k], ec="none", clip_on=False))
        ax_f.text(-1.9, -1.1 - 0.9 * k, NAMES[k], va="center", fontsize=5)
    ax_f.text(-3.0, -0.5, "GLM-HMM states (colours in c–e):", va="center", fontsize=5)

    # g-i: scoring scatters on pooled out-of-fold trials, state STATE
    p = m[f"pred_{STATE}"].to_numpy()
    q_lat = m[f"probe_latents_{STATE}"].to_numpy()
    q_his = m[f"probe_history_{STATE}"].to_numpy()
    ses = m["ses_idx"].to_numpy()
    dm = lambda v: v - pd.Series(v).groupby(ses).transform("mean").to_numpy()  # noqa: E731
    panels = [(gs[2, 2], p, q_lat, "g", "Step 4 · score, latents probe\nall out-of-fold trials", (0, 1)),
              (gs[3, 2], dm(p), dm(q_lat), "h", "latents probe, within session\n(session means removed)", (-0.8, 0.8)),
              (gs[4, 2], p, q_his, "i", "history probe (control)\nall out-of-fold trials", (0, 1))]
    for spec, x, y, letter, title, lim in panels:
        ax = fig.add_subplot(spec)
        ax.hexbin(x, y, gridsize=40, extent=(*lim, *lim), bins="log", cmap="Greys", mincnt=1, lw=0)
        ax.plot(lim, lim, color=COLORS[STATE], lw=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"target π_t ({NAMES[STATE].split(' ')[0]})")
        ax.set_ylabel("decoded π̂_t")
        ax.set_title(title, loc="left", fontsize=5)
        ax.text(0.04, 0.96, f"R² = {fs.num(r2(x, y), '.2f')}", transform=ax.transAxes, va="top")
        fs.panel_label(ax, letter, dx_pt=-20)
    fig.text(0.07, 0.012, f"R² = 1 − Σ(π − π̂)² / Σ(π − mean π)², over all {len(m):,} out-of-fold trials of "
             "50 training mice (eval sessions). Shown for S1; the same is computed for every state. "
             "Shading: log trial count. Scratch pilot figure.", fontsize=5)
    out = OUT / "fig_methods_explainer.svg"
    fs.save_figure(fig, out)
    print(out)


if __name__ == "__main__":
    main()
