"""Rastermap-sorted GRU latents for the four held-out example sessions, with the GLM-HMM states underneath.

Unit order from 96_rastermap_fit.py (seed 0), fitted on the 40 training-fold mice only and never on the
GLM-HMM. Per session: behaviour; raster of raw standardised latents in the raw-fit order; raster of
20-trial-smoothed latents in the smooth-fit order; then the hurdle GLM-HMM predictive posterior at K = 2, 4,
6, 10 as stacked strips. Left of each raster, a bar shows which K = 4 state each unit correlates with most
(training-fold mice), shaded by |r| -- a check on whether Rastermap's bands line up with the states.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
spec_path = PILOT / "scripts/92_unit_heatmaps.py"
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("uh", spec_path)
uh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uh)
sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

OUT, KOUT, TEST_FOLD = uh.OUT, uh.KOUT, uh.TEST_FOLD
KS, SMOOTH, K_BAR = [2, 4, 6, 10], 20, 4


def main():
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    fs.apply_paper_style()
    cmap = plt.get_cmap("tab10")
    rm = np.load(OUT / "rastermap_order.npz")
    if len(sys.argv) > 1 and sys.argv[1] == "sel":   # sessions picked for clear transitions (98_pick_sessions.py)
        sel = json.loads((OUT / "selected_sessions.json").read_text())
        sessions, tags, prefix = sel["sessions"], sel["tags"], "fig_rastermap_sel"
    else:
        meta = json.loads((OUT / "fig_unit_heatmaps.json").read_text())
        sessions = [meta[f"fig_unit_heatmap_{i}.svg"]["session"] for i in range(4)]
        tags = [meta[f"fig_unit_heatmap_{i}.svg"]["tag"] for i in range(4)]
        prefix = "fig_rastermap"
    j, cols = uh.load_latents()
    sc = np.load(KOUT / "weights" / f"hurdle_K2_fold{TEST_FOLD}.npz")
    Z = (j[cols].to_numpy() - sc["mean"]) / sc["scale"]
    ses_all = j["ses_idx"].to_numpy()
    choice = j["choice"].to_numpy()
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    train = np.flatnonzero(j["subject_id"].map(fom).to_numpy() != TEST_FOLD)
    samp = np.sort(np.random.default_rng(0).choice(train, 100_000, replace=False))
    post, dec = {}, {}
    for K in KS:
        P = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet", columns=[f"pred_{k}" for k in range(K)]).to_numpy()
        o = np.argsort(-((P * (choice == 2)[:, None]).sum(0) / P.sum(0)))
        post[K], dec[K] = P[:, o], np.load(KOUT / "oof" / f"hurdle_K{K}_latents.npy")[:, o]
    Ps, Zs = post[K_BAR][samp], Z[samp]
    r = ((Zs - Zs.mean(0)).T @ (Ps - Ps.mean(0))) / (len(samp) * Zs.std(0)[:, None] * Ps.std(0)[None, :])
    bar_state, bar_r = r.argmax(1), r.max(1)

    def bar_rgb(order):
        rgb = np.array([to_rgb(cmap(s)) for s in bar_state[order]])
        a = np.clip(bar_r[order] / 0.6, 0, 1)[:, None]
        return (a * rgb + (1 - a) * 1.0)[:, None, :]

    for si, ses in enumerate(sessions):
        rows = np.flatnonzero(ses_all == ses)
        g = j.iloc[rows]
        t = g["trial_in_session"].to_numpy()
        Zr = Z[rows]
        Zsm = pd.DataFrame(Zr).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
        Wmm, Hmm = 183, 236
        fig = plt.figure(figsize=fs.mm(Wmm, Hmm))
        ax_at = lambda x, y, w, h: fig.add_axes([x / Wmm, y / Hmm, w / Wmm, h / Hmm])  # noqa: E731
        gsb = fig.add_gridspec(1, 1, left=24 / Wmm, right=178 / Wmm, bottom=(Hmm - 30) / Hmm, top=(Hmm - 8) / Hmm)
        ch = np.where(g["choice"].to_numpy() == 2, np.nan, g["choice"].to_numpy().astype(float))
        _, sess_axes = plot_foraging_session(ch, g["reward"].to_numpy(),
                                             [g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()],
                                             ax=fig.add_subplot(gsb[0]), plot_list=["choice", "reward_prob"])
        for a in sess_axes:
            if a.get_legend():
                a.get_legend().remove()
            a.set_xlabel("")
        sess_axes[0].set_title(f"Mouse {g['subject_id'].iloc[0]}, {str(g['stage'].iloc[0]).lower()}, {len(g)} trials · "
                               f"{tags[si]} · held-out mouse", loc="left")
        x0, w = 30, 148
        for ri, (M, key, title) in enumerate([(Zr, "raw_s0_isort", "Rastermap order (raw latents)"),
                                              (Zsm, "smooth_s0_isort", "Rastermap order (20-trial-smoothed latents)")]):
            order = rm[key]
            y = Hmm - 92 - ri * 62
            ax = ax_at(x0, y, w, 56)
            ax.imshow(np.clip(M[:, order].T, -2, 2), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                      extent=[t[0] - 0.5, t[-1] + 0.5, len(cols), 0], interpolation="nearest")
            ax.set_xticklabels([])
            ax.set_yticks([0, 128, 256])
            ax.set_title(title + " · fitted on the other 40 mice, without the GLM-HMM", loc="left")
            axb = ax_at(x0 - 5, y, 3, 56)
            axb.imshow(bar_rgb(order), aspect="auto", extent=[0, 1, len(cols), 0])
            axb.set_xticks([])
            axb.set_yticks([])
            if ri == 0:
                axb.set_ylabel("unit", labelpad=14)
        for kk, K in enumerate(KS):
            for dd, (M, lab) in enumerate([(post[K], f"K={K} target"), (dec[K], "decoded")]):
                ax = ax_at(x0, Hmm - 166 - kk * 17 - dd * 7.5, w, 6.5)
                ax.stackplot(t, M[rows].T, colors=[cmap(k) for k in range(K)], lw=0)
                ax.set(xlim=(t[0], t[-1]), ylim=(0, 1), yticks=[])
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=2, fontsize=5)
                if kk < len(KS) - 1 or dd == 0:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("trial in session")
        fig.text(0.01, 0.004, "Rasters: standardised latents (training-fold scaler), clipped at ±2. Left bar: the K = 4 state "
                 "each unit correlates with most on the other 40 mice (colour), shaded by r (white = r ≤ 0, full = r ≥ 0.6).\n"
                 "Strips: GLM-HMM predictive posterior (target) and its out-of-fold decoding from the GRU latents; states sorted by "
                 "p(ignore), colours as in fig_kscan_decoding. "
                 "Rastermap 1.0, 40 clusters, 64 PCs, seed 0. Scratch pilot.", fontsize=5)
        out = OUT / f"{prefix}_{si}.svg"
        fs.save_figure(fig, out)
        plt.close(fig)
        print(out, flush=True)


if __name__ == "__main__":
    main()
