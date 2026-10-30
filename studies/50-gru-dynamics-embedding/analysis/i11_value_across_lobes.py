"""I11: how v is carried across trials and across the two lobes, in the (choice axis, v) plane.

    python analysis/i11_value_across_lobes.py

Coordinates: I2's orthonormal TDR axes, v = the value axis (I4 ``value_coordinate``) and c = the
choice axis (orthogonalised after v; it separates the two lobes). Row t's state has seen trial
t−1 and predicts trial t; each state is coloured by that t−1 input (B3h's colours).
Top row:
  * an example session (s12.example_sessions, Uncoupled Baiting; trials WINDOW) in the (c, v)
    plane, consecutive states joined; switch transitions drawn bold;
  * the same window over trials: v, c and the GRU's logit, with the animal's choices;
  * switch-triggered averages: aligned on the first state after a side switch that ends a run of
    ≥ MIN_RUN same-side choices, signed so the old side is positive, each variable divided by its
    SD over all trials (N_SWITCH switches sampled; mean ± SEM).
Bottom row, one panel per input u of trial t: the mean displacement (c, v)(t) → (c, v)(t+1) over
N_TRANSITIONS within-session transitions, binned in the plane, full-length arrows; I3's stable
attractor of u (all-sessions mean context) as a star.
Also ``fig_I11b_cv_plane.png``: B3h's 150,000-trial subsample in the (c, v) plane, coloured by
the t−1 input (left) and by the GRU's P(right) (right; colouring by v would only repeat the
y-axis). And ``fig_I11c_carry_in_pcs.png``: carry-over in PC space. Only post-switch states (the
first state after a choice switch, row k+1 for a switch on trial k), in PC1–PC2 and PC1–PC4,
coloured by (1) their own v, (2) the v of the state before the switch (row k), and (3) a
control: (2) shuffled among switches of the same direction. If v is carried, (2) reproduces
(1)'s gradient; the control keeps the side-level colouring but not the individual pairing.
Writes ``i11_value_across_lobes.json``, ``fig_I11_value_across_lobes.png``,
``fig_I11b_cv_plane.png`` and ``fig_I11c_carry_in_pcs.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from b3h_pc_overview import COLORS  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

SEED = 11
WINDOW = (60, 140)
MIN_RUN, LAGS, N_SWITCH = 5, 8, 50_000
N_TRANSITIONS = 1_000_000
INPUTS = {"R rewarded": (1, 1), "R unrewarded": (1, 0), "L rewarded": (0, 1), "L unrewarded": (0, 0)}
EC, EV = np.linspace(-2.6, 2.6, 21), np.linspace(-4, 4.4, 22)


def main() -> None:
    s12.apply_presentation_style()
    rng = np.random.default_rng(SEED)
    t = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "prev_action", "prev_outcome", "logit_diff", "task"])
    ax_file = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in ax_file["terms"]]
    choice_axis = ax_file["basis"][:, terms.index("choice")]
    v = value_coordinate()
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(np.float32)
    c = (states.astype(np.float64) - s12.load_pca()["mean"]) @ choice_axis
    del states
    logit = t["logit_diff"].to_numpy(float)
    ses = t["ses_idx"].to_numpy()
    action = t["action"].to_numpy()
    reward = (t["reward"].to_numpy() > 0).astype(int)
    pa, po = t["prev_action"].to_numpy(), t["prev_outcome"].to_numpy()
    label = np.full(len(t), "", dtype=object)
    for name, (a, r) in INPUTS.items():
        label[(pa == a) & (po == r)] = name
    same_next = np.r_[ses[1:] == ses[:-1], False]  # row k+1 is the same session
    att = json.loads((s12.ANALYSIS / "i3_tdr_dynamics.json").read_text())["attractors_all_sessions_context"]
    att_cv = {n: (a["tdr"][0][terms.index("choice")], a["tdr"][0][terms.index("value")]) for n, a in att.items()}

    fig = plt.figure(figsize=(34, 17), layout="constrained")
    top = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    g1 = top[0].subgridspec(1, 3, width_ratios=[1, 1.7, 1])
    g2 = top[1].subgridspec(1, 4)
    bg = rng.choice(len(t), 60_000, replace=False)

    # (a) example session in the (c, v) plane.
    ses_id = s12.example_sessions(t)["Uncoupled Baiting"]
    rows = np.flatnonzero(ses == ses_id)[WINDOW[0]:WINDOW[1]]
    ax = fig.add_subplot(g1[0])
    ax.scatter(c[bg], v[bg], s=1, color="0.88", linewidths=0, rasterized=True)
    for k0, k1 in zip(rows[:-1], rows[1:]):
        switch = action[k0] != pa[k0] if not np.isnan(pa[k0]) else False
        ax.plot([c[k0], c[k1]], [v[k0], v[k1]], color="k", lw=2.6 if switch else 0.8, alpha=0.9 if switch else 0.5, zorder=3)
    ax.scatter(c[rows], v[rows], s=70, c=[COLORS.get(x, "0.5") for x in label[rows]], edgecolors="k", linewidths=0.5, zorder=4)
    ax.set_xlabel("choice axis c", fontsize=21)
    ax.set_ylabel("value axis v", fontsize=21)
    ax.set_title(f"one session, trials {WINDOW[0]}–{WINDOW[1]} (bold: switches)", fontsize=22)

    # (b) the same window over trials.
    ax = fig.add_subplot(g1[1])
    x = np.arange(len(rows)) + WINDOW[0]
    ax.axhline(0, color="0.8", lw=1)
    ax.plot(x, v[rows], "-o", color="k", lw=2.4, ms=5, label="v (value axis)")
    ax.plot(x, c[rows], "-", color="#7570b3", lw=2.2, label="c (choice axis)")
    ax.plot(x, logit[rows] / logit.std() * v.std(), "-", color="C1", lw=1.6, alpha=0.8, label="GRU logit (scaled to v's SD)")
    for k, row in enumerate(rows):
        ax.vlines(x[k], 4.3, 4.7 if action[row] == 1 else 3.9, color="k" if reward[row] else "0.6", lw=1.5)
    ax.text(x[0], 4.85, "choice ticks: up = right, down = left; black rewarded", fontsize=15)
    ax.set_xlabel("trial", fontsize=21)
    ax.set_ylabel("coordinate", fontsize=21)
    ax.set_title("the same trials over time", fontsize=22)
    ax.legend(frameon=False, fontsize=17, loc="lower right")

    # (c) switch-triggered averages.
    run = np.zeros(len(t), int)
    for k in range(1, len(t)):
        run[k] = run[k - 1] + 1 if (ses[k] == ses[k - 1] and action[k] == action[k - 1]) else 1
    first = np.flatnonzero((np.r_[False, ses[1:] == ses[:-1]]) & (np.r_[0, action[:-1]] != action) & (np.r_[0, run[:-1]] >= MIN_RUN))
    # first[k] = trial whose choice switched; the first state after the switch is row first+1.
    aligned = first + 1
    ok = (aligned + LAGS < len(t)) & (aligned - LAGS >= 0)
    aligned = aligned[ok]
    ok = np.array([ses[a - LAGS] == ses[a] == ses[a + LAGS] for a in aligned])
    aligned = aligned[ok]
    aligned = rng.choice(aligned, min(N_SWITCH, len(aligned)), replace=False)
    sign = np.where(action[aligned - 2] == 1, 1.0, -1.0)  # old side (the choice before the switch) positive
    lags = np.arange(-LAGS, LAGS + 1)
    ax = fig.add_subplot(g1[2])
    sta = {}
    for name, arr, col in (("v", v, "k"), ("c", c, "#7570b3"), ("logit", logit, "C1")):
        m = np.array([sign * arr[aligned + L] for L in lags]) / arr.std()
        mu, se = m.mean(1), m.std(1) / np.sqrt(m.shape[1])
        sta[name] = {"mean": mu.tolist(), "sem": se.tolist()}
        ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        ax.plot(lags, mu, "-o", color=col, lw=2.4, ms=5, label={"v": "v", "c": "c", "logit": "GRU logit"}[name])
    ax.axvline(-0.5, color="0.6", ls="--", lw=1.5)
    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlabel("states from the first one after a switch", fontsize=21)
    ax.set_ylabel("toward the old side (SDs)", fontsize=21)
    ax.set_title(f"switch-triggered average ({len(aligned):,} switches)", fontsize=22)
    ax.legend(frameon=False, fontsize=18)

    # (d) flow in the (c, v) plane per input.
    pool = np.flatnonzero(same_next & (label != ""))
    pool = rng.choice(pool, min(N_TRANSITIONS, len(pool)), replace=False)
    flows = {}
    for k, (name, (a, r)) in enumerate(INPUTS.items()):
        ax = fig.add_subplot(g2[k])
        ax.scatter(c[bg], v[bg], s=1, color="0.88", linewidths=0, rasterized=True)
        sel = pool[(action[pool] == a) & (reward[pool] == r)]
        dc, dv = c[sel + 1] - c[sel], v[sel + 1] - v[sel]
        ic, iv = np.digitize(c[sel], EC) - 1, np.digitize(v[sel], EV) - 1
        gx, gy, gu, gw = [], [], [], []
        for i in range(len(EC) - 1):
            for j in range(len(EV) - 1):
                m = (ic == i) & (iv == j)
                if m.sum() >= 30:
                    gx.append((EC[i] + EC[i + 1]) / 2)
                    gy.append((EV[j] + EV[j + 1]) / 2)
                    gu.append(dc[m].mean())
                    gw.append(dv[m].mean())
        ax.quiver(gx, gy, gu, gw, angles="xy", scale_units="xy", scale=1.0, width=0.004, color=COLORS[name])
        ax.scatter(*att_cv[name], s=480, marker="*", color=COLORS[name], edgecolors="k", linewidths=1.4, zorder=5)
        ax.set_xlim(EC[0], EC[-1])
        ax.set_ylim(EV[0], EV[-1])
        ax.set_xlabel("choice axis c", fontsize=21)
        if k == 0:
            ax.set_ylabel("value axis v", fontsize=21)
        ax.set_title(f"input: {name}", fontsize=22, color=COLORS[name])
        flows[name] = {"n": int(len(sel)), "mean_dc": float(dc.mean()), "mean_dv": float(dv.mean())}
    for a in fig.axes:
        a.tick_params(labelsize=17)
    handles = [Line2D([], [], marker="o", ls="none", ms=13, color=col, markeredgecolor="k", label=n) for n, col in COLORS.items()]
    fig.axes[0].legend(handles=handles, title="t−1 input", frameon=False, fontsize=15, title_fontsize=16, loc="upper left")
    fig.savefig(s12.figure_path("fig_I11_value_across_lobes.png"), bbox_inches="tight")
    plt.close(fig)
    # (e) The (c, v) plane alone, coloured by the t−1 input and by P(right), B3h's style.
    sub = s12.subsample_rows(len(t), 150_000)
    order = np.random.default_rng(0).permutation(len(sub))
    sub = sub[order]
    p_right = 1 / (1 + np.exp(-logit))
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), sharex=True, sharey=True, layout="constrained")
    has = label[sub] != ""
    axes[0].scatter(c[sub][has], v[sub][has], s=2, c=[COLORS[x] for x in label[sub][has]], linewidths=0, alpha=0.6,
                    rasterized=True)
    sc = axes[1].scatter(c[sub], v[sub], s=2, c=p_right[sub], cmap="coolwarm", vmin=0, vmax=1, linewidths=0, rasterized=True)
    for a in axes:
        a.set_xlabel("choice axis c", fontsize=22)
        a.tick_params(labelsize=18)
    axes[0].set_ylabel("value axis v", fontsize=22)
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="none", ms=13, color=col, label=n) for n, col in COLORS.items()],
                   title="t−1 input", frameon=False, fontsize=18, title_fontsize=19, loc="upper left")
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.05, pad=0.02)
    cb.set_label("P(right)", fontsize=21)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(s12.figure_path("fig_I11b_cv_plane.png"), bbox_inches="tight")
    plt.close(fig)

    # (f) Carry-over in PC space: post-switch states coloured by the pre-switch v.
    from scipy.stats import spearmanr  # noqa: PLC0415

    scores = s12.load_pca()["scores"][:, :4]
    k = np.flatnonzero((np.r_[False, action[1:] != action[:-1]]) & np.r_[False, ses[1:] == ses[:-1]] & same_next)
    post, pre = k + 1, k
    direction = action[k]  # the new side
    shuffled = v[pre].copy()
    for d in (0, 1):
        m = np.flatnonzero(direction == d)
        shuffled[m] = v[pre][rng.permutation(m)]
    show = rng.choice(len(post), min(60_000, len(post)), replace=False)
    lim_c = float(np.abs(np.r_[v[post], v[pre]]).max())
    colourings = [("own v (after the switch)", v[post]), ("v before the switch", v[pre]),
                  ("v before the switch, shuffled within direction", shuffled)]
    carry = {name: float(spearmanr(col, v[post])[0]) for name, col in colourings}
    fig, axes = plt.subplots(2, 3, figsize=(25, 14), sharex="row", sharey="row", layout="constrained")
    for r, (i, j) in enumerate(((0, 1), (0, 3))):
        bgs = sub[:60_000]
        for cidx, (name, col) in enumerate(colourings):
            ax = axes[r, cidx]
            ax.scatter(scores[bgs, i], scores[bgs, j], s=1, color="0.9", linewidths=0, rasterized=True)
            sc = ax.scatter(scores[post[show], i], scores[post[show], j], s=3, c=col[show], cmap="coolwarm",
                            vmin=-lim_c, vmax=lim_c, linewidths=0, rasterized=True)
            ax.set_xlabel(f"PC{i + 1}", fontsize=22)
            if cidx == 0:
                ax.set_ylabel(f"PC{j + 1}", fontsize=22)
            if r == 0:
                ax.set_title(f"{name}\nρ with own v = {carry[name]:.2f}", fontsize=21)
            ax.tick_params(labelsize=18)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("colour: v (> 0 favours right)", fontsize=20)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(s12.figure_path("fig_I11c_carry_in_pcs.png"), bbox_inches="tight")
    plt.close(fig)
    print("carry ρ", {n: round(x, 3) for n, x in carry.items()}, "| switches", len(post), flush=True)

    s12.write_json("i11_value_across_lobes.json", {
        "example_session": ses_id, "window": WINDOW, "min_run": MIN_RUN, "n_switches": int(len(aligned)),
        "switch_triggered": sta, "lags": lags.tolist(), "flows": flows, "attractors_cv": att_cv,
        "sd": {"v": float(v.std()), "c": float(c.std()), "logit": float(logit.std())},
        "carry_in_pcs": {"n_switches": int(len(post)), "spearman_with_own_v": carry},
    }, "analysis/i11_value_across_lobes.py")
    print({k: [round(x, 2) for x in d["mean"]] for k, d in sta.items()}, flush=True)


if __name__ == "__main__":
    main()
