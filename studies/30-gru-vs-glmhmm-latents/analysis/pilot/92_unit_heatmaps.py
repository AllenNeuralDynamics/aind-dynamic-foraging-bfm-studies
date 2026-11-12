"""GRU unit activity, ordered by each unit's contribution to each state, for several sessions and K.

Mode `fit <K>`: refit the latents probe of fold TEST_FOLD (trained on the other four folds, exactly the
fit behind the out-of-fold predictions for these mice), save its weights and scaler, and check that it
reproduces the saved out-of-fold predictions on the test fold.

Mode `plot`: one figure per example session. Columns K = 2, 4, 6, 10. Rows: GLM-HMM target, decoded
posterior (out of fold), and a heatmap of the 256 standardised latents (the probe's input), units grouped
by the state whose logit they raise most (centred weight W_ik - mean_k W_ik, largest k) and sorted by that
weight within the group (strongest at the top of each group). States sorted by p(ignore), high to low,
as in fig_kscan_decoding.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT, KOUT = PILOT / "out", PILOT / "out" / "kscan"
WDIR = KOUT / "weights"
KS = [2, 4, 6, 10]
TEST_FOLD = 2           # holds mouse 729674, the earlier example
SHOWCASE = "729674_2024-10-03_135356"
N_RANDOM, SEED = 3, 7


def load_latents():
    import pyarrow.parquet as pq
    cols = [c for c in pq.ParquetFile(OUT / "joined_trials.parquet").schema.names if c.startswith("latent_")]
    j = pd.read_parquet(OUT / "joined_trials.parquet",
                        columns=["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward",
                                 "stage", "reward_probabilityL", "reward_probabilityR"] + cols)
    j["subject_id"] = j["subject_id"].astype(str)
    j["ses_idx"] = j["ses_idx"].astype(str)
    return j, cols


def fit(K):
    import importlib.util
    from sklearn.preprocessing import StandardScaler
    spec = importlib.util.spec_from_file_location("kp", PILOT / "scripts/kscan_probes.py")
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    j, cols = load_latents()
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    fold = j["subject_id"].map(fom).to_numpy()
    tr, te = np.flatnonzero(fold != TEST_FOLD), np.flatnonzero(fold == TEST_FOLD)
    X = j[cols].to_numpy(np.float64)
    P = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet", columns=[f"pred_{k}" for k in range(K)]).to_numpy()
    sc = StandardScaler().fit(X[tr])
    # Same optimisation as kscan_probes.fit_soft_logistic, keeping W.
    from scipy.optimize import minimize
    Xtr = sc.transform(X[tr])
    n, d = Xtr.shape
    Xb = np.hstack([Xtr, np.ones((n, 1))])
    Ptr = P[tr]

    def f(w):
        W = w.reshape(d + 1, K)
        Z = Xb @ W
        Z = Z - Z.max(1, keepdims=True)
        logq = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        loss = -(Ptr * logq).sum() / n + 0.5 * kp.L2 * (W[:-1] ** 2).sum()
        G = Xb.T @ (np.exp(logq) - Ptr) / n
        G[:-1] += kp.L2 * W[:-1]
        return loss, G.ravel()

    res = minimize(f, np.zeros((d + 1) * K), jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-7})
    W = res.x.reshape(d + 1, K)
    Q = kp.softmax(np.hstack([sc.transform(X[te]), np.ones((len(te), 1))]) @ W)
    oof = np.load(KOUT / "oof" / f"hurdle_K{K}_latents.npy")[te]
    err, mean_err = float(np.abs(Q - oof).max()), float(np.abs(Q - oof).mean())
    print(K, "max / mean |Q - saved oof| on test fold:", err, mean_err, flush=True)
    assert err < 1e-2  # same fit; L-BFGS on a different thread count differs by ~1e-3
    WDIR.mkdir(exist_ok=True)
    np.savez(WDIR / f"hurdle_K{K}_fold{TEST_FOLD}.npz", W=W, mean=sc.mean_, scale=sc.scale_, cols=np.array(cols))


def plot():
    sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                       "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
    import figure_style as fs
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    fs.apply_paper_style()
    cmap = plt.get_cmap("tab10")
    j, cols = load_latents()
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    in_fold = j["subject_id"].map(fom).to_numpy() == TEST_FOLD
    n = j[in_fold].groupby("ses_idx").agg(n=("trial", "size"), mouse=("subject_id", "first"))
    showcase = [s for s in n.index if SHOWCASE in s][0]
    pool = n[(n["n"].between(300, 700)) & (n["mouse"] != n.loc[showcase, "mouse"])]
    rng = np.random.default_rng(SEED)
    mice = rng.choice(pool["mouse"].unique(), N_RANDOM, replace=False)
    sessions = [showcase] + [rng.choice(pool.index[pool["mouse"] == m]) for m in mice]

    choice_all = j["choice"].to_numpy()
    states = json.loads((KOUT / "kscan_decoding_states.json").read_text())
    per_k = {}
    for K in KS:
        P = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet", columns=[f"pred_{k}" for k in range(K)]).to_numpy()
        order = np.argsort(-((P * (choice_all == 2)[:, None]).sum(0) / P.sum(0)))
        z = np.load(WDIR / f"hurdle_K{K}_fold{TEST_FOLD}.npz")
        W = z["W"][:-1][:, order]                      # units x states, states in p(ignore) order
        Wc = W - W.mean(1, keepdims=True)
        group = Wc.argmax(1)
        unit_order = np.lexsort((-Wc[np.arange(len(Wc)), group], group))
        per_k[K] = dict(P=P[:, order], Q=np.load(KOUT / "oof" / f"hurdle_K{K}_latents.npy")[:, order],
                        mean=z["mean"], scale=z["scale"], unit_order=unit_order,
                        counts=np.bincount(group, minlength=K))

    meta = {}
    for si, ses in enumerate(sessions):
        rows = np.flatnonzero(j["ses_idx"].to_numpy() == ses)
        g = j.iloc[rows]
        t = g["trial_in_session"].to_numpy()
        Wmm, Hmm = 183, 172
        fig = plt.figure(figsize=fs.mm(Wmm, Hmm))
        ax_at = lambda x, y, w, h: fig.add_axes([x / Wmm, y / Hmm, w / Wmm, h / Hmm])  # noqa: E731
        gsb = fig.add_gridspec(1, 1, left=18 / Wmm, right=173 / Wmm, bottom=(Hmm - 32) / Hmm, top=(Hmm - 9) / Hmm)
        axb = fig.add_subplot(gsb[0])
        ch = np.where(g["choice"].to_numpy() == 2, np.nan, g["choice"].to_numpy().astype(float))
        _, sess_axes = plot_foraging_session(ch, g["reward"].to_numpy(),
                                             [g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()],
                                             ax=axb, plot_list=["choice", "reward_prob"])
        for a in sess_axes:
            if a.get_legend():
                a.get_legend().remove()
            a.set_xlabel("")
        tag = "showcase (most posterior movement)" if si == 0 else f"random pick {si} (seed {SEED})"
        sess_axes[0].set_title(f"Mouse {g['subject_id'].iloc[0]}, {str(g['stage'].iloc[0]).lower()}, "
                               f"{len(g)} trials · {tag} · test fold {TEST_FOLD + 1} (probe never saw this mouse)",
                               loc="left")
        for c, K in enumerate(KS):
            d = per_k[K]
            colors = [cmap(k) for k in range(K)]
            x0, w = 18 + c * 40, 35
            for r, (M, lab) in enumerate([(d["P"], "target π_t"), (d["Q"], "decoded from latents")]):
                ax = ax_at(x0, Hmm - 56 - r * 22, w, 17)
                ax.stackplot(t, M[rows].T, colors=colors, lw=0)
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 1])
                ax.set_xticklabels([])
                if r == 0:
                    ax.set_title(f"K = {K}", fontsize=7)
                if c == 0:
                    ax.set_ylabel(lab)
            ax = ax_at(x0, 14, w, Hmm - 96)
            Z = ((g[cols].to_numpy() - d["mean"]) / d["scale"])[:, d["unit_order"]].T
            ax.imshow(np.clip(Z, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                      extent=[t[0] - 0.5, t[-1] + 0.5, len(cols), 0], interpolation="nearest")
            edges = np.cumsum(d["counts"])
            for k, (lo, hi) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
                if hi > lo:
                    ax.add_patch(Rectangle((t[0] - 0.06 * (t[-1] - t[0]), lo), 0.04 * (t[-1] - t[0]), hi - lo,
                                           color=colors[k], clip_on=False, lw=0))
                if 0 < hi < len(cols):
                    ax.axhline(hi, color="k", lw=0.3)
            ax.set_yticks([0, 128, 256] if c == 0 else [])
            ax.set_xlabel("trial in session")
            if c == 0:
                ax.set_ylabel("GRU unit (grouped by state, strongest weight first)")
        fig.text(0.01, 0.01, "Heatmap: standardised GRU latents (the probe's input; z-scored with the training-fold "
                 "scaler), clipped at ±2.5. Bar: the state whose logit each unit raises most (centred probe weight). "
                 "States sorted by p(ignore), high → low; colours as in fig_kscan_decoding. Hurdle GLM-HMM, global "
                 "parameters. Scratch pilot.", fontsize=5, wrap=True)
        out = OUT / f"fig_unit_heatmap_{si}.svg"
        fs.save_figure(fig, out)
        plt.close(fig)
        meta[out.name] = {"session": ses, "mouse": str(g["subject_id"].iloc[0]), "n_trials": int(len(g)), "tag": tag}
        print(out, meta[out.name], flush=True)
    meta["units_per_state"] = {K: per_k[K]["counts"].tolist() for K in KS}
    (OUT / "fig_unit_heatmaps.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import traceback
    try:
        fit(int(sys.argv[2])) if sys.argv[1] == "fit" else plot()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
