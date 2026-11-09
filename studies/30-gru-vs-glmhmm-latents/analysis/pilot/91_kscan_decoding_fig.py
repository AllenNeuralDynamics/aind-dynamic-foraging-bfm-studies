"""Decoding at several K: example traces and confusion matrices, hurdle K = 2, 4, 6, 10.

Row per K. Columns: example session target (stacked predictive posterior), decoded from GRU latents,
decoded from history (out of fold), confusion matrix for latents and for history (argmax, rows
sum to 1, trials with max pi >= CONF). States are sorted by their observed p(ignore), high to low
(weights = predictive posterior), so state 0 is the most ignore-prone at every K.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
KOUT = OUT / "kscan"
KS = [2, 4, 6, 10]
CONF = 0.6
SESSION = "729674_2024-10-03_135356"
CMAP = plt.get_cmap("tab10")


def main():
    fs.apply_paper_style()
    j = pd.read_parquet(OUT / "joined_trials.parquet",
                        columns=["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward"])
    j["ses_idx"] = j["ses_idx"].astype(str)
    ses = j["ses_idx"].to_numpy()
    choice = j["choice"].to_numpy()
    prev_choice = np.r_[np.nan, choice[:-1]].astype(float)
    prev_rew = np.r_[np.nan, j["reward"].to_numpy()[:-1]].astype(float)
    first = np.r_[True, ses[1:] != ses[:-1]]
    resp = (choice != 2) & (prev_choice != 2) & ~first
    stay_after_rew = resp & (prev_rew == 1)
    sid = [s for s in np.unique(ses) if SESSION in s][0]
    rows = np.flatnonzero(ses == sid)
    t = j["trial_in_session"].to_numpy()[rows]

    W, H = 183, 162
    fig = plt.figure(figsize=fs.mm(W, H))
    ax_at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731
    summary = {}
    for r, K in enumerate(KS):
        P = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet",
                            columns=[f"pred_{k}" for k in range(K)]).to_numpy()
        Q = {f: np.load(KOUT / "oof" / f"hurdle_K{K}_{f}.npy") for f in ("latents", "history")}
        p_ign = (P * (choice == 2)[:, None]).sum(0) / P.sum(0)
        win_stay = (P[stay_after_rew] * (choice[stay_after_rew] == prev_choice[stay_after_rew])[:, None]).sum(0) \
            / P[stay_after_rew].sum(0)
        order = np.argsort(-p_ign)
        P = P[:, order]
        Q = {f: q[:, order] for f, q in Q.items()}
        colors = [CMAP(k) for k in range(K)]
        y0 = H - 34 - r * 34
        info = {"p_ignore": p_ign[order].round(3).tolist(), "win_stay": win_stay[order].round(3).tolist(),
                "occupancy": P.mean(0).round(3).tolist()}
        for c, (M, title) in enumerate([(P, "target π_t (GLM-HMM)"), (Q["latents"], "decoded from GRU latents"),
                                        (Q["history"], "decoded from history")]):
            ax = ax_at(16 + c * 37, y0, 33, 22)
            ax.stackplot(t, M[rows].T, colors=colors, lw=0)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            if r == 0:
                ax.set_title(title, fontsize=6)
            if r == len(KS) - 1:
                ax.set_xlabel("trial in session")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"K = {K}\np(state)")
        conf = P.max(1) >= CONF
        for c, f in enumerate(("latents", "history")):
            tt, dd = P[conf].argmax(1), Q[f][conf].argmax(1)
            cm = np.zeros((K, K))
            np.add.at(cm, (tt, dd), 1)
            cmr = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            bacc = float(np.mean([cmr[k, k] for k in range(K) if cm[k].sum() > 0]))
            ax = ax_at(130 + c * 27, y0, 22, 22)
            ax.imshow(cmr, cmap="Greys", vmin=0, vmax=1)
            if K <= 4:
                for a in range(K):
                    for b in range(K):
                        ax.text(b, a, fs.num(cmr[a, b], ".2f"), ha="center", va="center", fontsize=5,
                                color="white" if cmr[a, b] > 0.55 else "black")
            ax.set_xticks(range(K), [str(k) for k in range(K)], fontsize=5)
            ax.set_yticks(range(K), [str(k) for k in range(K)], fontsize=5)
            for lab, col in zip(ax.get_yticklabels(), colors):
                lab.set_color(col)
            for lab, col in zip(ax.get_xticklabels(), colors):
                lab.set_color(col)
            ax.set_title(f"{'latents' if f == 'latents' else 'history'}: bal. acc. {fs.num(bacc, '.2f')}",
                         fontsize=5)
            if r == len(KS) - 1:
                ax.set_xlabel("decoded")
            if c == 0:
                ax.set_ylabel("GLM-HMM")
            info[f"bal_acc_{f}"] = bacc
        info["frac_confident"] = float(conf.mean())
        summary[K] = info
    fig.text(0.01, 0.012, f"Hurdle GLM-HMM, global parameters; mouse 729674 example session (picked for posterior "
             f"movement).\nStates sorted by observed p(ignore), high → low; same colours in traces and matrix ticks. "
             f"Matrices: out-of-fold argmax, trials with max π ≥ {CONF}, rows sum to 1. 50 training mice. Scratch pilot.",
             fontsize=5)
    fs.save_figure(fig, OUT / "fig_kscan_decoding.svg")
    (KOUT / "kscan_decoding_states.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
