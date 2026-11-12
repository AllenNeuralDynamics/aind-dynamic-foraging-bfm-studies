"""Targeted dimensionality reduction (Mante et al. 2013) of the GRU latents: GLM-HMM state vs choice/reward axes.

Each standardised latent unit z_i(t) (row t = the state that predicts trial t) is regressed on task variables
that are all known before trial t:
  state      pi_ign    p(ignore-prone state)       hurdle K = 3 predictive posterior (sorted by p(ignore))
             pi_weak   p(engaged, weak win-stay)    (pi_strong = 1 - pi_ign - pi_weak is implied)
  choice     side_ema  signed choice history, EMA tau = 10 (+1 R, -1 L, ignores 0): a value / preference proxy
             prev_side choice t-1 (+1 R, -1 L, 0 ignore)
             prev_ign  ignore at t-1
  reward     prev_rew  reward at t-1
             prev_win  prev_side x prev_rew (signed win)
             rew_ema   reward rate, EMA tau = 10
  time       tis       trial in session / 500
Axes are fitted on the 40 training-fold mice (folds != 3) and evaluated on the 10 test-fold mice. Regression
axes are denoised in the top N_PC principal components (as in Mante et al.). Outputs:
  unique variance per variable (drop-one R^2 loss, held-out), held-out correlation of each axis projection with
  its variable, cosine similarity between axes, and projections over time for the example sessions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT, KOUT = PILOT / "out", PILOT / "out" / "kscan"
sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
TEST_FOLD, N_PC, TAU = 2, 12, 10
NAMES = ["pi_ign", "pi_weak", "side_ema", "prev_side", "prev_ign", "prev_rew", "prev_win", "rew_ema", "tis"]
LABELS = {"pi_ign": "p(ignore-prone)", "pi_weak": "p(engaged, weak WS)", "side_ema": "choice history",
          "prev_side": "choice t−1", "prev_ign": "ignore t−1", "prev_rew": "reward t−1",
          "prev_win": "signed win t−1", "rew_ema": "reward rate", "tis": "trial in session"}
GROUP_COLOR = {"pi_ign": "#4C72B0", "pi_weak": "#4C72B0", "side_ema": "#C44E52", "prev_side": "#C44E52",
               "prev_ign": "#C44E52", "prev_rew": "#55A868", "prev_win": "#55A868", "rew_ema": "#55A868",
               "tis": "#8C8C8C"}


def ema(v, ses):
    out = np.zeros(len(v))
    first = np.r_[True, ses[1:] != ses[:-1]]
    a, acc = 1.0 / TAU, 0.0
    for t in range(len(v)):          # value at t uses trials < t only
        if first[t]:
            acc = 0.0
        out[t] = acc
        acc = (1 - a) * acc + a * v[t]
    return out


def regressors(j, P):
    ses = j["ses_idx"].to_numpy()
    c, r = j["choice"].to_numpy(), j["reward"].to_numpy().astype(float)
    first = np.r_[True, ses[1:] != ses[:-1]]
    side = np.where(c == 1, 1.0, np.where(c == 0, -1.0, 0.0))
    rew = np.where(c == 2, 0.0, r)
    lag = lambda v: np.where(first, 0.0, np.r_[0.0, v[:-1]])  # noqa: E731
    X = np.column_stack([P[:, 0], P[:, 1], ema(side, ses), lag(side), lag((c == 2).astype(float)), lag(rew),
                         lag(side * rew), ema(rew, ses), j["trial_in_session"].to_numpy() / 500.0])
    return X


def main():
    import pyarrow.parquet as pq
    cols = [c for c in pq.ParquetFile(OUT / "joined_trials.parquet").schema.names if c.startswith("latent_")]
    j = pd.read_parquet(OUT / "joined_trials.parquet",
                        columns=["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward"] + cols)
    j["subject_id"], j["ses_idx"] = j["subject_id"].astype(str), j["ses_idx"].astype(str)
    P3 = pd.read_parquet(KOUT / "glmhmm_hurdle_K3.parquet", columns=["pred_0", "pred_1", "pred_2"]).to_numpy()
    c = j["choice"].to_numpy()
    order = np.argsort(-((P3 * (c == 2)[:, None]).sum(0) / P3.sum(0)))         # ignore-prone first
    P3 = P3[:, order]
    # among the two engaged states, which has the weaker win-stay?
    ses = j["ses_idx"].to_numpy()
    first = np.r_[True, ses[1:] != ses[:-1]]
    pc, pr = np.r_[-1, c[:-1]], np.r_[0, j["reward"].to_numpy()[:-1]]
    m = (~first) & (c != 2) & (pc != 2) & (pr == 1)
    ws = (P3[m] * (c[m] == pc[m])[:, None]).sum(0) / P3[m].sum(0)
    if ws[1] > ws[2]:                     # put the weak-win-stay engaged state in column 1
        P3 = P3[:, [0, 2, 1]]
        ws = ws[[0, 2, 1]]
    X = regressors(j, P3)
    sc = np.load(KOUT / "weights" / f"hurdle_K2_fold{TEST_FOLD}.npz")
    Z = (j[cols].to_numpy() - sc["mean"]) / sc["scale"]
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    test = j["subject_id"].map(fom).to_numpy() == TEST_FOLD
    tr, te = ~test, test
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xs = (X - mu) / sd
    Zc = Z - Z[tr].mean(0)

    def fit(cols_idx):
        A = np.column_stack([Xs[tr][:, cols_idx], np.ones(tr.sum())])
        B, *_ = np.linalg.lstsq(A, Zc[tr], rcond=None)
        return B[:-1], B[-1]

    def r2_heldout(cols_idx):
        B, b0 = fit(cols_idx)
        pred = Xs[te][:, cols_idx] @ B + b0
        return 1 - ((Zc[te] - pred) ** 2).sum() / ((Zc[te] - Zc[te].mean(0)) ** 2).sum()

    allc = list(range(len(NAMES)))
    r2_full = float(r2_heldout(allc))
    unique = {NAMES[k]: float(r2_full - r2_heldout([i for i in allc if i != k])) for k in allc}
    single = {NAMES[k]: float(r2_heldout([k])) for k in allc}
    # axes: full-model coefficients, denoised in the top N_PC PCs of training activity
    B, _ = fit(allc)                                         # p x units
    U, S, Vt = np.linalg.svd(Zc[tr][::4], full_matrices=False)
    D = Vt[:N_PC].T @ Vt[:N_PC]                              # projector onto the top PCs
    axes = (B @ D).T                                          # units x p
    axes_n = axes / np.linalg.norm(axes, axis=0, keepdims=True)
    cos = axes_n.T @ axes_n
    proj = Zc @ axes_n                                        # trials x p
    corr_te = {NAMES[k]: float(np.corrcoef(proj[te, k], X[te, k])[0, 1]) for k in allc}
    var_pc = (S[:N_PC] ** 2).sum() / (S ** 2).sum()
    res = {"r2_full_heldout": r2_full, "unique_r2": unique, "single_r2": single, "corr_projection_heldout": corr_te,
           "cosine": cos.round(3).tolist(), "names": NAMES, "n_pc_denoise": N_PC, "var_in_top_pcs": float(var_pc),
           "win_stay_by_state": ws.round(3).tolist(), "n_train": int(tr.sum()), "n_test": int(te.sum())}
    (OUT / "tdr_results.json").write_text(json.dumps(res, indent=1))
    np.savez(OUT / "tdr_axes.npz", axes_n=axes_n, z_mean_train=Z[tr].mean(0), x_mean=mu, x_sd=sd,
             proj_mean_train=proj[tr].mean(0), proj_sd_train=proj[tr].std(0), names=np.array(NAMES))
    print(json.dumps({k: v for k, v in res.items() if k != "cosine"}, indent=1))

    # ---- figure
    import figure_style as fs
    import matplotlib.pyplot as plt
    fs.apply_paper_style()
    labs = [LABELS[n] for n in NAMES]
    Wmm, Hmm = 183, 150
    fig = plt.figure(figsize=fs.mm(Wmm, Hmm))
    ax_at = lambda x, y, w, h: fig.add_axes([x / Wmm, y / Hmm, w / Wmm, h / Hmm])  # noqa: E731
    ax = ax_at(28, 84, 58, 58)
    ax.imshow(np.abs(cos), cmap="Greys", vmin=0, vmax=1)
    for a in range(len(NAMES)):
        for b in range(len(NAMES)):
            ax.text(b, a, fs.num(cos[a, b], ".1f"), ha="center", va="center", fontsize=5,
                    color="white" if abs(cos[a, b]) > 0.55 else "black")
    ax.set_xticks(range(len(NAMES)), labs, rotation=60, ha="right", fontsize=5)
    ax.set_yticks(range(len(NAMES)), labs, fontsize=5)
    for t1, t2 in zip(ax.get_xticklabels(), ax.get_yticklabels()):
        t1.set_color(GROUP_COLOR[NAMES[labs.index(t1.get_text())]])
        t2.set_color(GROUP_COLOR[NAMES[labs.index(t2.get_text())]])
    ax.set_title("a · cosine between regression axes\n(denoised in top 12 PCs)", fontsize=6)
    ax = ax_at(108, 100, 30, 36)
    y = np.arange(len(NAMES))
    ax.barh(y, [unique[n] for n in NAMES], color=[GROUP_COLOR[n] for n in NAMES])
    ax.set_yticks(y, labs, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("unique R² (held-out mice)")
    ax.set_title(f"b · population variance each variable adds\nbeyond the other 8 (full model R² = {fs.num(r2_full, '.2f')})",
                 fontsize=6)
    ax = ax_at(150, 100, 30, 36)
    ax.barh(y, [corr_te[n] for n in NAMES], color=[GROUP_COLOR[n] for n in NAMES])
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("r (projection, variable)")
    ax.set_title("c · each axis read out\non held-out mice", fontsize=6)
    meta = json.loads((OUT / "fig_unit_heatmaps.json").read_text())
    shows = [meta["fig_unit_heatmap_0.svg"]["session"], meta["fig_unit_heatmap_3.svg"]["session"]]
    pick = ["pi_ign", "pi_weak", "side_ema", "prev_win"]
    for si, sname in enumerate(shows):
        rows = np.flatnonzero(ses == sname)
        t = j["trial_in_session"].to_numpy()[rows]
        for pi, n in enumerate(pick):
            k = NAMES.index(n)
            ax = ax_at(14 + si * 88, 60 - pi * 14, 80, 11)
            pz = (proj[rows, k] - proj[tr, k].mean()) / proj[tr, k].std()
            vz = (X[rows, k] - X[tr, k].mean()) / X[tr, k].std()
            ax.plot(t, vz, color="0.6", lw=0.6)
            ax.plot(t, pz, color=GROUP_COLOR[n], lw=0.6)
            ax.set_xlim(t[0], t[-1])
            ax.set_yticks([])
            ax.text(0.005, 0.95, LABELS[n], transform=ax.transAxes, fontsize=5, va="top")
            if pi < len(pick) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("trial in session")
            if pi == 0:
                ax.set_title(("d · " if si == 0 else "") + f"mouse {sname.split('_')[0]}: grey = variable, colour = "
                             "projection on its axis (z-scored)", loc="left", fontsize=5)
    fig.text(0.01, 0.01, "Hurdle K = 3 GLM-HMM predictive posterior; GRU H256/D614 latents (row t predicts trial t; all "
             "variables use trials < t). Axes fitted on 40 training-fold mice, evaluated on the 10 test-fold mice. "
             "Blue = state, red = choice, green = reward, grey = time. Scratch pilot.", fontsize=5)
    fs.save_figure(fig, OUT / "fig_tdr.svg")


if __name__ == "__main__":
    main()
