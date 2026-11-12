"""Four views of how the GRU population carries the GLM-HMM state, for the four example sessions.

Per session one figure. Top: session plot, then the GRU's top principal components (view C,
K-independent). Then one column per K = 2, 4, 6, 10:
  row 1  target pi_t (stacked)
  row 2  view A: state logit directions, l_k(t) = w_k . z_t + b_k, centred across states (what softmax sees)
  row 3  view B: all 256 units, grouped by the state they correlate with most (signed Pearson r with pi_k on
         training-fold mice), sorted by r within the group
  row 4  view D: top 15 units per state by centred probe weight (units in that state's weight group)
z_t = latents standardised with the training-fold scaler (the probe's input). Probe = fold-3 refit
(trained on the other 40 mice). States sorted by p(ignore), high to low, colours as in fig_kscan_decoding.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
sys.path.insert(0, str(PILOT / "scripts"))
sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("uh", PILOT / "scripts/92_unit_heatmaps.py")
uh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uh)

import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT, KOUT, WDIR = uh.OUT, uh.KOUT, uh.WDIR
KS, TEST_FOLD = uh.KS, uh.TEST_FOLD
N_TOP, N_PC, SMOOTH = 15, 6, 20


def group_bar(ax, t, counts, colors, n_rows):
    edges = np.cumsum(counts)
    span = t[-1] - t[0]
    for k, (lo, hi) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
        if hi > lo:
            ax.add_patch(Rectangle((t[0] - 0.06 * span, lo), 0.04 * span, hi - lo, color=colors[k],
                                   clip_on=False, lw=0))
        if 0 < hi < n_rows:
            ax.axhline(hi, color="k", lw=0.3)


def heat(ax, Z, t):
    ax.imshow(np.clip(Z, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              extent=[t[0] - 0.5, t[-1] + 0.5, Z.shape[0], 0], interpolation="nearest")


def main():
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    from sklearn.decomposition import PCA
    fs.apply_paper_style()
    cmap = plt.get_cmap("tab10")
    meta = json.loads((OUT / "fig_unit_heatmaps.json").read_text())
    sessions = [meta[f"fig_unit_heatmap_{i}.svg"]["session"] for i in range(4)]
    tags = [meta[f"fig_unit_heatmap_{i}.svg"]["tag"] for i in range(4)]
    j, cols = uh.load_latents()
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    train = np.flatnonzero(j["subject_id"].map(fom).to_numpy() != TEST_FOLD)
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(train, 100_000, replace=False))
    sc = np.load(WDIR / f"hurdle_K2_fold{TEST_FOLD}.npz")
    zs = lambda X: (X - sc["mean"]) / sc["scale"]  # noqa: E731  (same training-fold scaler for every K)
    Zs = zs(j[cols].to_numpy()[samp])
    pca = PCA(N_PC).fit(Zs)
    ses_all = j["ses_idx"].to_numpy()
    # slow fraction of each PC: variance kept by a centred 20-trial moving average within sessions.
    # NOTE (PR #137 review): the average below runs over the random sample, so a 20-row window spans many more
    # than 20 consecutive trials. The values reported in r1 are recomputed by analysis/freeze_r1.py on complete
    # training-fold sessions; this script is kept as run.
    pcs_s = pd.DataFrame(pca.transform(Zs))
    sm = pcs_s.groupby(ses_all[samp]).transform(lambda v: v.rolling(SMOOTH, center=True, min_periods=1).mean())
    slow_frac = (sm.var() / pcs_s.var()).to_numpy()

    choice_all = j["choice"].to_numpy()
    per_k = {}
    for K in KS:
        P = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet", columns=[f"pred_{k}" for k in range(K)]).to_numpy()
        order = np.argsort(-((P * (choice_all == 2)[:, None]).sum(0) / P.sum(0)))
        P = P[:, order]
        z = np.load(WDIR / f"hurdle_K{K}_fold{TEST_FOLD}.npz")
        W, b = z["W"][:-1][:, order], z["W"][-1][order]
        Wc = W - W.mean(1, keepdims=True)
        wgroup = Wc.argmax(1)
        top = [np.flatnonzero(wgroup == k)[np.argsort(-Wc[wgroup == k, k])][:N_TOP] for k in range(K)]
        Ps = P[samp]
        r = ((Zs - Zs.mean(0)).T @ (Ps - Ps.mean(0))) / (len(samp) * Zs.std(0)[:, None] * Ps.std(0)[None, :])
        rgroup = r.argmax(1)
        rorder = np.lexsort((-r[np.arange(len(r)), rgroup], rgroup))
        per_k[K] = dict(P=P, W=W, b=b, top=top, rorder=rorder, rcounts=np.bincount(rgroup, minlength=K),
                        r_best=np.round([r[rgroup == k, k].max() if (rgroup == k).any() else np.nan for k in range(K)], 2))

    for si, ses in enumerate(sessions):
        rows = np.flatnonzero(ses_all == ses)
        g = j.iloc[rows]
        t = g["trial_in_session"].to_numpy()
        Z = zs(g[cols].to_numpy())
        Wmm, Hmm = 183, 235
        fig = plt.figure(figsize=fs.mm(Wmm, Hmm))
        ax_at = lambda x, y, w, h: fig.add_axes([x / Wmm, y / Hmm, w / Wmm, h / Hmm])  # noqa: E731
        gsb = fig.add_gridspec(1, 1, left=18 / Wmm, right=173 / Wmm, bottom=(Hmm - 30) / Hmm, top=(Hmm - 8) / Hmm)
        ch = np.where(g["choice"].to_numpy() == 2, np.nan, g["choice"].to_numpy().astype(float))
        _, sess_axes = plot_foraging_session(ch, g["reward"].to_numpy(),
                                             [g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()],
                                             ax=fig.add_subplot(gsb[0]), plot_list=["choice", "reward_prob"])
        for a in sess_axes:
            if a.get_legend():
                a.get_legend().remove()
            a.set_xlabel("")
        sess_axes[0].set_title(f"Mouse {g['subject_id'].iloc[0]}, {str(g['stage'].iloc[0]).lower()}, {len(g)} trials · "
                               f"{tags[si]} · probe never saw this mouse", loc="left")
        # view C: top PCs, raw (thin) and 20-trial smoothed (thick)
        axc = ax_at(18, Hmm - 58, 122, 22)
        pcs = pca.transform(Z)
        pcs = pcs / pcs_s.std().to_numpy()
        for p in range(N_PC):
            off = -3.2 * p
            sm_p = pd.Series(pcs[:, p]).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
            axc.plot(t, pcs[:, p] + off, color="0.75", lw=0.3)
            axc.plot(t, sm_p + off, color="k", lw=0.8)
            axc.text(t[-1] + 0.01 * (t[-1] - t[0]), off,
                     f"PC{p + 1} ({fs.num(100 * pca.explained_variance_ratio_[p], '.0f')}% var, "
                     f"slow {fs.num(slow_frac[p], '.2f')})", fontsize=5, va="center")
        axc.set_xlim(t[0], t[-1])
        axc.set_yticks([])
        axc.set_title("C · top principal components of the latents (grey raw, black 20-trial average); "
                      "'slow' = share of variance kept by the average", loc="left")
        for c, K in enumerate(KS):
            d = per_k[K]
            colors = [cmap(k) for k in range(K)]
            x0, w = 18 + c * 40, 35
            ax = ax_at(x0, Hmm - 80, w, 14)
            ax.stackplot(t, d["P"][rows].T, colors=colors, lw=0)
            ax.set(xlim=(t[0], t[-1]), ylim=(0, 1), yticks=[0, 1], xticklabels=[])
            ax.set_title(f"K = {K}", fontsize=7)
            if c == 0:
                ax.set_ylabel("target π_t")
            ax = ax_at(x0, Hmm - 103, w, 18)
            L = Z @ d["W"] + d["b"]
            L = L - L.mean(1, keepdims=True)
            for k in range(K):
                ax.plot(t, L[:, k], color=colors[k], lw=0.5)
            ax.set_xlim(t[0], t[-1])
            ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("A · state logit\nw_k·z_t + b_k")
            ax = ax_at(x0, Hmm - 162, w, 52)
            heat(ax, Z[:, d["rorder"]].T, t)
            group_bar(ax, t, d["rcounts"], colors, len(cols))
            ax.set_xticklabels([])
            ax.set_yticks([0, 128, 256] if c == 0 else [])
            if c == 0:
                ax.set_ylabel("B · 256 units grouped by the state\nthey correlate with most", labelpad=2)
            ax = ax_at(x0, 14, w, 52)
            idx = np.concatenate(d["top"])
            heat(ax, Z[:, idx].T, t)
            group_bar(ax, t, [len(x) for x in d["top"]], colors, len(idx))
            ax.set_yticks([])
            ax.set_xlabel("trial in session")
            if c == 0:
                ax.set_ylabel(f"D · top {N_TOP} units per state\nby probe weight", labelpad=8)
        fig.text(0.01, 0.008, "Heatmaps: standardised latents (training-fold scaler), clipped at ±2.5. B: groups from "
                 "correlation with π_k on the other 40 mice. D: units whose centred probe weight is largest for that "
                 "state, top 15 each.\nA: centred logits (softmax of these = decoded posterior). Scratch pilot.",
                 fontsize=5)
        out = OUT / f"fig_unit_views_{si}.svg"
        fs.save_figure(fig, out)
        plt.close(fig)
        print(out, flush=True)
    (OUT / "fig_unit_views.json").write_text(json.dumps({
        "sessions": sessions, "pc_explained_var": pca.explained_variance_ratio_.round(4).tolist(),
        "pc_slow_fraction": slow_frac.round(3).tolist(),
        "best_unit_r_per_state": {K: per_k[K]["r_best"].tolist() for K in KS},
        "units_per_state_by_r": {K: per_k[K]["rcounts"].tolist() for K in KS}}, indent=1))


if __name__ == "__main__":
    main()
