"""Producer for r1: the training-mice pilot, from frozen_r1.json + frozen_r1_sessions.npz (offline).

Writes the r1 figures (fig_r1_*.svg), r1_pilot_decoding.json and the result blocks of
reports/r1-pilot-decoding.md:
  result-1  setup              result-5  shuffle nulls
  result-2  K = 3 vs baselines result-6  where the state sits in the population
  result-3  beyond engagement  result-7  example sessions with transitions
  result-4  K scan
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
import upstream_sources  # noqa: E402  (analysis/upstream_sources.py)
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

FROZEN, SESS = HERE / "frozen_r1.json", HERE / "frozen_r1_sessions.npz"
REPORT = HERE / "reports" / "r1-pilot-decoding.md"
# Study 14's GLM-HMM groups. The GRU run (study 02, group nxd-3way@20260707-075401, which has no launch record
# in study 02) is read through its W&B artifact and recorded as a `wandb_run` source and in `upstream`.
WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
SETS = ["1_gru_latents", "2_gru_logits", "3_history", "4_latents+history", "5_untrained_latents", "6_trial_index"]
SET_LABELS = ["GRU latents", "GRU logits", "History", "Latents + history", "Untrained GRU", "Trial index"]
K3_NAMES = ["S0 ignore-prone", "S1 engaged, strong win-stay", "S2 engaged, weak win-stay"]
K3_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
FSET_COL = {"latents": "#1b9e77", "logits": "#d95f02", "history": "#7570b3", "untrained": "#8c8c8c"}
TAB = plt.get_cmap("tab10")
KS_SHOW, KS_NULL = [2, 4, 6, 10], [2, 3, 6, 10]
SMOOTH = 20
OUTPUTS: list[Path] = []


def save(fig, name):
    path = HERE / name
    fs.save_figure(fig, path)
    plt.close(fig)
    OUTPUTS.append(path)
    return name


def mm_axes(fig, W, H):
    return lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])


def stacked(ax, t, M, K):
    # fills are rasterized (text and axes stay vector) to keep the SVGs small
    for coll in ax.stackplot(t, M.T, colors=[TAB(k) for k in range(K)], lw=0):
        coll.set_rasterized(True)
    ax.set(xlim=(t[0] - 0.5, t[-1] + 0.5), ylim=(0, 1), yticks=[])


def session_plot(fig, rect_mm, W, H, z, pre, title):
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    x, y, w, h = rect_mm
    gs = fig.add_gridspec(1, 1, left=x / W, right=(x + w) / W, bottom=y / H, top=(y + h) / H)
    ch = z[pre + "choice"].astype(float)
    ch[ch == 2] = np.nan
    _, axes = plot_foraging_session(ch, z[pre + "reward"].astype(float), [z[pre + "pL"], z[pre + "pR"]],
                                    ax=fig.add_subplot(gs[0]), plot_list=["choice", "reward_prob"])
    n = len(ch)
    for a in axes:
        if a.get_legend():
            a.get_legend().remove()
        a.set_xlabel("")
        a.set_xlim(0.5, n + 0.5)     # plot_foraging_session draws trial i at x = i (1-based); state panels match
    axes[0].set_title(title, loc="left")
    return axes


# ---------------------------------------------------------------- figures
def fig_method(d, z):
    W, H = 183, 200
    fig = plt.figure(figsize=fs.mm(W, H))
    at = mm_axes(fig, W, H)
    ax = at(10, H - 26, 170, 22)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="k",
                                    lw=0.5, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", shrinkA=0, shrinkB=0))

    box(0.00, 0.25, 0.12, 0.50, "choices and\nrewards,\ntrials 1 … t−1", "#eeeeee")
    box(0.19, 0.58, 0.16, 0.40, "trained GRU\n(study 02, H = 256)", "#dde6f3")
    box(0.41, 0.58, 0.15, 0.40, "latents h_t\n(256 numbers)", "#dde6f3")
    box(0.62, 0.58, 0.19, 0.40, "linear probe\nπ̂_t = softmax(W·h_t + b)", "#fbe3d0")
    box(0.19, 0.02, 0.16, 0.40, "GLM-HMM\n(study 14, hurdle K = 3)", "#e3f0e3")
    box(0.41, 0.02, 0.40, 0.40, "target π_t = p(z_t | trials < t)\none-step predictive state posterior", "#e3f0e3")
    box(0.87, 0.25, 0.13, 0.50, "score\nR²(π_t, π̂_t)\non held-out mice", "#f5f5f5")
    for a in [(0.12, 0.60, 0.19, 0.78), (0.12, 0.40, 0.19, 0.22), (0.35, 0.78, 0.41, 0.78), (0.56, 0.78, 0.62, 0.78),
              (0.35, 0.22, 0.41, 0.22), (0.81, 0.78, 0.87, 0.62), (0.81, 0.22, 0.87, 0.38)]:
        arrow(*a)
    ax.text(0.715, 0.50, "fit W, b on 40 mice, predict the other 10", ha="center", va="center",
            transform=ax.transAxes, fontsize=5, style="italic")
    fs.panel_label(ax, "a", dx_pt=-14, dy_pt=0)

    pre = "s0_"
    t = z[pre + "trial_in_session"] + 1   # 1-based, as plot_foraging_session draws trials
    info = d["sessions"]["items"][0]
    axes = session_plot(fig, (18, H - 57, 105, 24), W, H, z, pre,
                        f"example session (mouse {info['mouse']}, {info['stage'].lower()}, test fold 3)")
    fs.panel_label(axes[0], "b")
    P, Q = z[pre + "target_K3"], z[pre + "decoded_latents_K3"]
    ax = at(18, H - 80, 105, 15)
    for coll in ax.stackplot(t, P.T, colors=K3_COLORS, lw=0):
        coll.set_rasterized(True)
    ax.set(xlim=(t[0] - 0.5, t[-1] + 0.5), ylim=(0, 1), ylabel="p(state)", xticklabels=[])
    ax.set_title("step 1, target: GLM-HMM predictive posterior π_t", loc="left")
    fs.panel_label(ax, "c")
    ax = at(18, H - 110, 105, 22)
    r3 = z["unit_r_K3"]
    best = np.abs(r3).argmax(1)
    order = np.lexsort((-r3[np.arange(len(r3)), best], best))
    Zs = z[pre + "latents_z"].astype(float)[:, order].T
    ax.imshow(np.clip(Zs, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              extent=[t[0] - 0.5, t[-1] + 0.5, 256, 0], interpolation="nearest")
    edges = np.cumsum(np.bincount(best, minlength=3))
    for k, (lo, hi) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
        ax.add_patch(Rectangle((t[0] - 0.06 * (t[-1] - t[0]), lo), 0.04 * (t[-1] - t[0]), hi - lo, color=K3_COLORS[k],
                               clip_on=False, lw=0))
    ax.set(ylabel="GRU unit", xticklabels=[])
    ax.set_title("step 2, features: 256 GRU latents (z-scored), grouped by the state each tracks best", loc="left")
    fs.panel_label(ax, "d")
    ax = at(18, H - 132, 105, 15)
    for k in range(3):
        ax.plot(t, P[:, k], color=K3_COLORS[k], lw=1.0)
        ax.plot(t, Q[:, k], color="k", lw=0.5, ls=(0, (2, 1)))
    ax.set(xlim=(t[0] - 0.5, t[-1] + 0.5), ylim=(-0.02, 1.02), ylabel="p(state)", xlabel="trial in session")
    ax.set_title("step 3, probe output: solid = target, dashed = decoded (probe never saw this mouse)", loc="left")
    fs.panel_label(ax, "e")
    ax = at(140, H - 60, 38, 26)
    fom = d["k3"]["fold_of_mouse"]
    for f in range(5):
        for i in range(10):
            ax.add_patch(Rectangle((i, 4 - f), 0.85, 0.8, fc="#DD8452" if f == 2 else "#cccccc", ec="none"))
        ax.text(-0.4, 4 - f + 0.4, f"fold {f + 1}", ha="right", va="center", fontsize=5)
    mice = sorted([m for m, f in fom.items() if f == 2])
    pos = mice.index(info["mouse"])
    ax.add_patch(Rectangle((pos, 2), 0.85, 0.8, fc="none", ec="k", lw=0.8))
    ax.set(xlim=(-3.2, 10), ylim=(-0.3, 5.2))
    ax.axis("off")
    ax.set_title("cross-validation by mouse:\n5 folds of 10; box = example mouse", loc="left", fontsize=5)
    fs.panel_label(ax, "f", dx_pt=-6)
    for i, (key, title, letter) in enumerate([("latents_all", "step 4, score: latents", "g"),
                                              ("latents_within", "latents, within session", "h"),
                                              ("history_all", "history probe (control)", "i")]):
        h = d["hist2d_k3_state1"][key]
        lo, hi = h["lim"]
        ax = at(146, H - 104 - i * 42, 28, 28)
        C = np.array(h["counts"], float).T
        ax.imshow(np.log1p(C), origin="lower", extent=[lo, hi, lo, hi], cmap="Greys", aspect="equal")
        ax.plot([lo, hi], [lo, hi], color=K3_COLORS[1], lw=0.6)
        ax.set(xlabel="target π (S1)", ylabel="decoded π̂")
        ax.set_title(title, loc="left", fontsize=5)
        ax.text(0.04, 0.96, f"R² = {fs.num(h['r2'], '.2f')}", transform=ax.transAxes, va="top")
        fs.panel_label(ax, letter, dx_pt=-26)
    fig.text(0.01, 0.006, "R² = 1 − Σ(π − π̂)² / Σ(π − mean π)² over all out-of-fold trials (shading: log count). "
             "States: blue S0 ignore-prone, orange S1 engaged with strong win-stay, green S2 engaged with weak win-stay.",
             fontsize=5)
    return save(fig, "fig_r1_method.svg")


def fig_k3(d):
    res = d["k3"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(175, 52), sharey=True)
    x = np.arange(len(SETS))
    for ax, key, title in [(axes[0], "r2", "all trials"), (axes[1], "r2_within", "within session (session means removed)")]:
        for k in range(3):
            ax.bar(x + (k - 1) * 0.26, [res[f"pred|{s}"]["logistic"][key][k] for s in SETS], 0.26, color=K3_COLORS[k],
                   label=K3_NAMES[k])
            ax.scatter(x + (k - 1) * 0.26, [res[f"pred|{s}"]["ridge_alr"][key][k] for s in SETS], s=5,
                       facecolors="none", edgecolors="k", linewidths=0.4, zorder=3, label="ridge check" if k == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x, SET_LABELS, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("R² of the predictive posterior")
    axes[0].set_ylim(-0.1, 1)
    axes[1].legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.subplots_adjust(left=0.07, right=0.8, bottom=0.3, top=0.9, wspace=0.1)
    return save(fig, "fig_r1_k3.svg")


def fig_engaged(d):
    e = d["k3_engaged"]
    sets = ["GRU latents", "GRU logits", "History"]
    W, H = 165, 58
    fig = plt.figure(figsize=fs.mm(W, H))
    at = mm_axes(fig, W, H)
    for i, name in enumerate(sets):
        cmr = np.array(e["sets"][name]["confusion_confident_row"])
        ax = at(16 + i * 34, 16, 27, 27)
        ax.imshow(cmr, cmap="Greys", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, fs.num(cmr[a, b], ".2f"), ha="center", va="center",
                        color="white" if cmr[a, b] > 0.55 else "black")
        ax.set_xticks(range(3), ["S0", "S1", "S2"])
        ax.set_yticks(range(3), ["S0", "S1", "S2"])
        ax.set_xlabel("decoded (argmax)")
        if i == 0:
            ax.set_ylabel("GLM-HMM (argmax)")
        ax.set_title(f"{name}\nbalanced acc. {fs.num(e['sets'][name]['balanced_acc_confident'], '.2f')}", fontsize=6)
        fs.panel_label(ax, "abc"[i], dx_pt=-30 if i == 0 else -14)
    ax = at(126, 16, 36, 27)
    x = np.arange(3)
    for k, (key, lab, col) in enumerate([("engaged_r2_S1_vs_S2", "all engaged trials", "#555555"),
                                         ("engaged_r2_S1_vs_S2_within", "within session", "#bbbbbb")]):
        ax.bar(x + (k - 0.5) * 0.36, [e["sets"][n][key] for n in sets], 0.36, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, sets, rotation=25, ha="right")
    ax.set(ylim=(-0.1, 1), ylabel="R², p(S1 | S1 or S2)")
    ax.set_title("engaged trials (p(S0) < 0.2):\nS1 vs S2 only", fontsize=6)
    ax.legend(frameon=False, fontsize=5, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    fs.panel_label(ax, "d", dx_pt=-26)
    fig.text(0.01, 0.015, f"a–c: out of fold, trials with max π ≥ 0.8 ({e['n_confident']:,} of {e['n_trials']:,}); "
             f"rows sum to 1.\nd: {e['n_engaged']:,} engaged trials; the same 3-class probes, renormalised over S1 and S2.",
             fontsize=5)
    return save(fig, "fig_r1_engaged.svg")


def fig_kscan(d):
    parts, glm = d["kscan"]["parts"], d["kscan"]["glmhmm"]
    Ks = list(range(2, 11))
    fig, axes = plt.subplots(1, 4, figsize=fs.mm(183, 54))
    for ax, (m, title) in zip(axes[:3], [("pooled_r2", "pooled R², all trials"),
                                         ("pooled_r2_within", "pooled R², within session"),
                                         ("info_recovered", "information recovered")]):
        for fset in ("latents", "untrained", "logits", "history"):
            for e, ls in (("hurdle", "-"), ("reference", "--")):
                ax.plot(Ks, [parts[f"{e}__K{K}__{fset}"][m] for K in Ks], ls, marker="o", ms=2, color=FSET_COL[fset],
                        lw=0.8, label=f"{fset}" if e == "hurdle" else None)
        ax.set(ylim=(0, 1), xticks=Ks, xlabel="K")
        ax.set_title(title, fontsize=6)
    axes[0].set_ylabel("score (out of fold)")
    axes[0].legend(frameon=False, fontsize=5, loc="lower left")
    ax = axes[3]
    for e, ls in (("hurdle", "-"), ("reference", "--")):
        ax.plot(Ks, [glm[e][str(K)]["glmhmm_likelihood_pilot"] for K in Ks], ls, marker="o", ms=2, color="k", lw=0.8,
                label=f"{e}, these trials (global)")
        ax.plot(Ks, [glm[e][str(K)]["logged_train_mice_eval_likelihood"] for K in Ks], ls, marker="o", ms=2,
                color="0.6", lw=0.8, label=f"{e}, logged (per-mouse refit)")
    ax.set(xticks=Ks, xlabel="K")
    ax.set_title("GLM-HMM predictive likelihood", fontsize=6)
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    fig.text(0.01, 0.01, "Solid: hurdle emission; dashed: reference emission. Same 50 mice, folds and probe at every K.",
             fontsize=5)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.33, top=0.88, wspace=0.35)
    return save(fig, "fig_r1_kscan.svg")


def fig_kdecoding(d, z):
    W, H = 183, 162
    fig = plt.figure(figsize=fs.mm(W, H))
    at = mm_axes(fig, W, H)
    pre = "s0_"
    t = z[pre + "trial_in_session"] + 1   # 1-based, as plot_foraging_session draws trials
    for r, K in enumerate(KS_SHOW):
        y0 = H - 34 - r * 34
        for c, (key, title) in enumerate([("target", "target π_t (GLM-HMM)"), ("decoded_latents", "decoded from GRU latents"),
                                          ("decoded_history", "decoded from history")]):
            ax = at(16 + c * 37, y0, 33, 22)
            stacked(ax, t, z[f"{pre}{key}_K{K}"], K)
            ax.set_yticks([0, 1])
            if r == 0:
                ax.set_title(title, fontsize=6)
            if r < len(KS_SHOW) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("trial in session")
            if c == 0:
                ax.set_ylabel(f"K = {K}\np(state)")
        for c, f in enumerate(("latents", "history")):
            cm = d["confusions"][str(K)][f]
            row = np.array(cm["row"])
            ax = at(130 + c * 27, y0, 22, 22)
            ax.imshow(row, cmap="Greys", vmin=0, vmax=1)
            if K <= 4:
                for a in range(K):
                    for b in range(K):
                        ax.text(b, a, fs.num(row[a, b], ".2f"), ha="center", va="center", fontsize=5,
                                color="white" if row[a, b] > 0.55 else "black")
            ax.set_xticks(range(K), [str(k) for k in range(K)], fontsize=5)
            ax.set_yticks(range(K), [str(k) for k in range(K)], fontsize=5)
            for lab, col in zip(list(ax.get_xticklabels()) + list(ax.get_yticklabels()),
                                [TAB(k) for k in range(K)] * 2):
                lab.set_color(col)
            ax.set_title(f"{f}: bal. acc. {fs.num(cm['balanced_acc'], '.2f')}", fontsize=5)
            if r == len(KS_SHOW) - 1:
                ax.set_xlabel("decoded")
            if c == 0:
                ax.set_ylabel("GLM-HMM")
    fig.text(0.01, 0.012, "Hurdle GLM-HMM; the example session of fig_r1_method (picked for posterior movement). States "
             "sorted by observed p(ignore), high → low; same colours in traces and matrix ticks.\nMatrices: out of fold, "
             "argmax, trials with max π ≥ 0.6, rows sum to 1.", fontsize=5)
    return save(fig, "fig_r1_kdecoding.svg")


def fig_shuffle(d):
    parts, sh = d["kscan"]["parts"], d["shuffle"]
    metrics = [("pooled_r2", "pooled R², all trials"), ("pooled_r2_within", "pooled R², within session"),
               ("info_recovered", "information recovered")]
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(150, 60))
    x = np.arange(len(KS_NULL))
    for ax, (m, title) in zip(axes, metrics):
        for fset, off in (("latents", -0.2), ("history", 0.2)):
            ax.plot(x + off, [parts[f"hurdle__K{K}__{fset}"][m] for K in KS_NULL], "o", color=FSET_COL[fset], ms=4,
                    label=f"{fset}, real")
            for null, mk, dx in (("swap", "x", -0.07), ("circ", "+", 0.07)):
                for i, K in enumerate(KS_NULL):
                    v = [p[m] for key, p in sh.items() if key.startswith(f"{fset}_{null}_K{K}_")]
                    ax.plot(np.full(len(v), i + off + dx), v, mk, color=FSET_COL[fset], ms=2.5, mew=0.5,
                            label=(f"{fset}, {'session swap' if null == 'swap' else 'circular shift'}" if i == 0 else None))
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xticks=x, xticklabels=[str(K) for K in KS_NULL], xlabel="K", ylim=(-0.25, 1))
        ax.set_title(title, fontsize=6)
    axes[0].set_ylabel("score (out of fold)")
    axes[2].legend(frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.text(0.01, 0.01, "Decoder trained and scored on shuffled pairs. Session swap: target from another mouse's session "
             "in the same fold; circular shift: target rotated 20–80 % within its own session.", fontsize=5)
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.34, top=0.88, wspace=0.35)
    return save(fig, "fig_r1_shuffle.svg")


def fig_population(d, z):
    tdr = d["tdr"]
    names = tdr["names"]
    labels = {"pi_ign": "p(ignore-prone)", "pi_weak": "p(engaged, weak WS)", "side_ema": "choice history",
              "prev_side": "choice t−1", "prev_ign": "ignore t−1", "prev_rew": "reward t−1",
              "prev_win": "signed win t−1", "rew_ema": "reward rate", "tis": "trial in session"}
    gcol = {"pi_ign": "#4C72B0", "pi_weak": "#4C72B0", "side_ema": "#C44E52", "prev_side": "#C44E52",
            "prev_ign": "#C44E52", "prev_rew": "#55A868", "prev_win": "#55A868", "rew_ema": "#55A868", "tis": "#8C8C8C"}
    W, H = 183, 122
    fig = plt.figure(figsize=fs.mm(W, H))
    at = mm_axes(fig, W, H)
    pre = "s0_"
    t = z[pre + "trial_in_session"] + 1   # 1-based, as plot_foraging_session draws trials
    for c, K in enumerate(KS_SHOW):
        ax = at(16 + c * 42, H - 34, 36, 22)
        L = z[f"{pre}state_logit_K{K}"]
        for k in range(K):
            ax.plot(t, L[:, k], color=TAB(k), lw=0.5)
        ax.set(xlim=(t[0] - 0.5, t[-1] + 0.5), xlabel="trial in session")
        ax.set_title(f"K = {K}", fontsize=6)
        if c == 0:
            ax.set_ylabel("state logit\nw_k·z_t + b_k (centred)")
            fs.panel_label(ax, "a")
    cos = np.array(tdr["cosine"])
    labs = [labels[n] for n in names]
    ax = at(30, 22, 52, 52)
    ax.imshow(np.abs(cos), cmap="Greys", vmin=0, vmax=1)
    for a in range(len(names)):
        for b in range(len(names)):
            ax.text(b, a, fs.num(cos[a, b], ".1f"), ha="center", va="center", fontsize=5,
                    color="white" if abs(cos[a, b]) > 0.55 else "black")
    ax.set_xticks(range(len(names)), labs, rotation=60, ha="right", fontsize=5)
    ax.set_yticks(range(len(names)), labs, fontsize=5)
    for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tl.set_color(gcol[names[labs.index(tl.get_text())]])
    ax.set_title("cosine between TDR axes", fontsize=6)
    fs.panel_label(ax, "b", dx_pt=-60)
    y = np.arange(len(names))
    ax = at(106, 34, 30, 40)
    ax.barh(y, [tdr["unique_r2"][n] for n in names], color=[gcol[n] for n in names])
    ax.set_yticks(y, labs, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("drop in R² without it (held-out mice)")
    ax.set_title(f"variance each variable adds beyond\nthe other 8 (full model R² {fs.num(tdr['r2_full_heldout'], '.2f')})",
                 fontsize=6)
    fs.panel_label(ax, "c", dx_pt=-60)
    ax = at(148, 34, 30, 40)
    ax.barh(y, [tdr["corr_projection_heldout"][n] for n in names], color=[gcol[n] for n in names])
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="r (projection, variable)")
    ax.set_title("each axis read out\non held-out mice", fontsize=6)
    fs.panel_label(ax, "d", dx_pt=-6)
    fig.text(0.01, 0.01, "a: the example session of fig_r1_method; the probe's logits (softmax of these = decoded "
             "posterior), states sorted by p(ignore). b–d: targeted dimensionality reduction (Mante et al. 2013): each "
             "unit regressed on 9 variables\nknown before trial t, axes denoised in the top 12 PCs, fitted on 40 mice and "
             "evaluated on the 10 test-fold mice. Blue = state (K = 3), red = choice, green = reward, grey = time.",
             fontsize=5)
    return save(fig, "fig_r1_population.svg")


def fig_session(d, z, i):
    info = d["sessions"]["items"][i]
    pre = info["key"]
    t = z[pre + "trial_in_session"] + 1   # 1-based, as plot_foraging_session draws trials
    Zr = z[pre + "latents_z"].astype(float)
    Zsm = pd.DataFrame(Zr).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    r4 = z["unit_r_K4"]
    bstate, br = r4.argmax(1), r4.max(1)
    W, H = 183, 252
    fig = plt.figure(figsize=fs.mm(W, H))
    at = mm_axes(fig, W, H)
    session_plot(fig, (30, H - 30, 148, 22), W, H, z, pre,
                 f"Mouse {info['mouse']}, {info['stage'].lower()}, {info['n_trials']} trials · {info['tag']} · test fold 3")
    x0, w = 30, 148
    for ri, (M, key, title) in enumerate([(Zr, "rastermap_raw_isort", "Rastermap order, raw latents"),
                                          (Zsm, "rastermap_smooth_isort", "Rastermap order, 20-trial-smoothed latents")]):
        order = z[key]
        y = H - 100 - ri * 64
        ax = at(x0, y, w, 56)
        ax.imshow(np.clip(M[:, order].T, -2, 2), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                  extent=[t[0] - 0.5, t[-1] + 0.5, 256, 0], interpolation="nearest")
        ax.set(xticklabels=[], yticks=[0, 128, 256])
        ax.set_title(title + " (fitted on the other 40 mice, without the GLM-HMM)", loc="left")
        axb = at(x0 - 11, y, 3, 56)
        rgb = np.array([to_rgb(TAB(s)) for s in bstate[order]])
        a = np.clip(br[order] / 0.6, 0, 1)[:, None]
        axb.imshow(np.clip(a * rgb + (1 - a), 0, 1)[:, None, :], aspect="auto", extent=[0, 1, 256, 0])
        axb.set(xticks=[], yticks=[])
    for kk, K in enumerate(KS_SHOW):
        for dd, (key, lab) in enumerate([("target", f"K={K} target"), ("decoded_latents", "decoded")]):
            ax = at(x0, H - 176 - kk * 17 - dd * 7.5, w, 6.5)
            stacked(ax, t, z[f"{pre}{key}_K{K}"], K)
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=2, fontsize=5)
            if kk < len(KS_SHOW) - 1 or dd == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("trial in session")
    fig.text(0.01, 0.004, "Rasters: standardised latents, clipped at ±2. Left bar: the K = 4 state each unit correlates with "
             "most on the other 40 mice, shaded by r (white ≤ 0, full ≥ 0.6). Strips: GLM-HMM predictive posterior\n"
             "(target) and its out-of-fold decoding from the latents; states sorted by p(ignore). Rastermap 1.0, 40 "
             "clusters, 64 PCs, seed 0.", fontsize=5)
    return save(fig, f"fig_r1_session_{i}.svg")


# ---------------------------------------------------------------- report blocks
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def blocks(d, figs):
    p, v = d["pilot"], d["pilot"]["verification"]["join"]
    res = d["k3"]["results"]
    step1 = p["step1_filtered_convention"]
    s1 = step1.get("eval", step1)
    b = {}
    b["result-1"] = "\n".join([
        table(["", "value"], [
            ["GRU", f"study 02 3-way, H256 / D614 / seed 0, run `{p['gru']['run_id']}` (W&B `{p['gru']['wandb']}`), "
                    f"artifact `{p['gru']['artifact']}` digest `{p['gru']['artifact_digest']}`, step 90,000"],
            ["GLM-HMM", "study 14 hurdle and reference K = 2–10, L = 10, global parameters for every mouse "
                        "(groups `hurdle@20261027-212905`, `reference@20261027-213006`)"],
            ["mice / data", f"{v['n_mice']} GRU-training mice, odd (eval) sessions only: {v['n_sessions']} sessions, "
                            f"{v['n_trials']:,} trials (snapshot 20260603, ignores included)"],
            ["cross-validation", "5 folds of 10 mice (GroupKFold, seed 0); every score is out of fold"],
            ["target", "one-step predictive posterior p(z_t | trials < t)"],
            ["decoder", f"softmax linear readout, soft labels, L2 {d['k3']['l2_logistic']:g} on z-scored features"],
            ["likelihood on these trials", f"GRU {v['gru_normalized_likelihood']:.4f}, GLM-HMM (K = 3, global) "
                                           f"{v['glmhmm_normalized_likelihood']:.4f}; GRU better on "
                                           f"{100 * v['frac_trials_gru_beats_glmhmm']:.0f} % of trials"],
            ["saved `filtered_k` (study 14)", f"= p(z_t | trials ≤ t), max diff {s1['maxabs_saved_vs_a_filtered_incl_t']:.1e}; "
                                              f"differs from the predictive posterior by up to "
                                              f"{s1['maxabs_saved_vs_b_predictive']:.2f}"],
        ]),
        "", f"![method](../{figs['method']})"])
    rows = []
    for s, lab in zip(SETS, SET_LABELS):
        lg = res[f"pred|{s}"]["logistic"]
        rows.append([lab, " / ".join(f2(x) for x in lg["r2"]), " / ".join(f2(x) for x in lg["r2_within"]),
                     f"{lg['kl']:.3f}"])
    vs = d["k3"]["variance_split"]["pred"]["within_frac"]
    b["result-2"] = "\n".join([
        "K = 3 hurdle, per-state R² (S0 ignore-prone / S1 engaged strong win-stay / S2 engaged weak win-stay), out of "
        f"fold. Within-session share of the target's variance: {' / '.join(f'{100 * x:.0f} %' for x in vs)}.", "",
        table(["features", "R² all trials", "R² within session", "KL (nats)"], rows), "",
        f"![K = 3](../{figs['k3']})"])
    e = d["k3_engaged"]["sets"]
    b["result-3"] = "\n".join([
        table(["features", "balanced acc. (max π ≥ 0.8)", "S2 → S2", "engaged-only R², S1 vs S2", "within session"],
              [[n, f2(e[n]["balanced_acc_confident"]), f2(e[n]["confusion_confident_row"][2][2]),
                f2(e[n]["engaged_r2_S1_vs_S2"]), f2(e[n]["engaged_r2_S1_vs_S2_within"])]
               for n in ("GRU latents", "GRU logits", "History")]), "",
        f"![engaged](../{figs['engaged']})"])
    parts, glm = d["kscan"]["parts"], d["kscan"]["glmhmm"]
    rows = []
    for K in range(2, 11):
        g = glm["hurdle"][str(K)]
        rows.append([K, g["run_id"], f"{g['glmhmm_likelihood_pilot']:.4f}"]
                    + [f"{f2(parts[f'hurdle__K{K}__{f}']['pooled_r2'])} / {f2(parts[f'hurdle__K{K}__{f}']['pooled_r2_within'])}"
                       for f in ("latents", "untrained", "logits", "history")])
    def emission_gap(f):   # largest |hurdle − reference| over K = 2–10, pooled R² all trials and within session
        return max(abs(parts[f"hurdle__K{K}__{f}"][m] - parts[f"reference__K{K}__{f}"][m])
                   for K in range(2, 11) for m in ("pooled_r2", "pooled_r2_within"))

    gaps = ", ".join(f"{lab} {emission_gap(f):.3f}" for f, lab in (("latents", "latents"), ("untrained", "untrained GRU"),
                                                                   ("logits", "GRU logits"), ("history", "history")))
    b["result-4"] = "\n".join([
        "Hurdle emission; pooled R² all trials / within session. Largest hurdle − reference difference over K = 2–10 "
        f"(either score): {gaps} (figure, dashed). The untrained-GRU probe hit the 1,000-iteration cap in every fold "
        "(unconverged fits).", "",
        table(["K", "run", "GLM-HMM lik.", "latents", "untrained GRU", "GRU logits", "history"], rows), "",
        f"![K scan](../{figs['kscan']})", "", f"![decoding at K = 2, 4, 6, 10](../{figs['kdecoding']})"])
    sh = d["shuffle"]
    rows = []
    for fset in ("latents", "history"):
        for K in KS_NULL:
            real = parts[f"hurdle__K{K}__{fset}"]
            cell = [fset, K, f"{f2(real['pooled_r2'])} / {f2(real['pooled_r2_within'])}"]
            for null in ("swap", "circ"):
                vals = [p for k, p in sh.items() if k.startswith(f"{fset}_{null}_K{K}_")]
                cell.append(f"{f2(np.mean([x['pooled_r2'] for x in vals]))} / "
                            f"{f2(np.mean([x['pooled_r2_within'] for x in vals]))} (max within "
                            f"{f2(np.max([x['pooled_r2_within'] for x in vals]))}, n = {len(vals)})")
            rows.append(cell)
    b["result-5"] = "\n".join([
        "Pooled R² all trials / within session; nulls as mean (max within-session value, number of shuffles).", "",
        table(["features", "K", "real", "session swap", "circular shift"], rows), "",
        f"![shuffle](../{figs['shuffle']})"])
    tdr, pop = d["tdr"], d["population"]
    pc = pop["pc"]
    b["result-6"] = "\n".join([
        table(["measure", "value"], [
            ["TDR full model, held-out R² of the population", f2(tdr["r2_full_heldout"])],
            ["unique R² (drop without it): p(ignore-prone) + p(engaged, weak WS)",
             f"{f2(tdr['unique_r2']['pi_ign'])} + {f2(tdr['unique_r2']['pi_weak'])}"],
            ["unique R² (drop without it): ignore t−1 / choice history / reward t−1",
             f"{f2(tdr['unique_r2']['prev_ign'])} / {f2(tdr['unique_r2']['side_ema'])} / {f2(tdr['unique_r2']['prev_rew'])}"],
            ["cosine, choice t−1 vs ignore t−1 axes", f2(tdr["cosine"][3][4])],
            ["top 6 PCs: variance / slow share (20-trial average, complete sessions)",
             ", ".join(f"PC{i + 1} {100 * a:.0f} % / {s:.2f}" for i, (a, s) in
                       enumerate(zip(pc["explained_var"], pc["slow_fraction"])))],
            ["Rastermap seed stability (Spearman of embeddings, raw / smooth)",
             f"{f2(pop['rastermap_seed_stability_spearman']['raw'])} / {f2(pop['rastermap_seed_stability_spearman']['smooth'])}"],
        ]), "", f"![population](../{figs['population']})"])
    rows = [[it["mouse"], it["stage"].lower(), it["n_trials"], it["tag"]] for it in d["sessions"]["items"]]
    c = d["sessions"]["selection_criterion"]
    b["result-7"] = "\n".join([
        f"All five sessions are in test fold 3. Sessions 1–4 were ranked by clear K = {c['K']} transitions (max π ≥ "
        f"{c['conf']}, runs ≥ {c['min_run']} trials, one session per mouse).", "",
        table(["mouse", "stage", "trials", "why shown"], rows), ""]
        + [f"![session {i}](../{figs[f'session_{i}']})" for i in range(1, 5)])
    return b


def write_blocks(b):
    text = REPORT.read_text()
    for name, body in b.items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(\n<!-- END {name} -->)", re.S)
        assert pat.search(text), f"{name} markers missing from {REPORT.name}"
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    REPORT.write_text(text)


def main() -> None:
    fs.apply_paper_style()
    d = json.loads(FROZEN.read_text())
    z = dict(np.load(SESS))
    figs = {"method": fig_method(d, z), "k3": fig_k3(d), "engaged": fig_engaged(d), "kscan": fig_kscan(d),
            "kdecoding": fig_kdecoding(d, z), "shuffle": fig_shuffle(d), "population": fig_population(d, z)}
    for i in range(1, 5):
        figs[f"session_{i}"] = fig_session(d, z, i)
    b = blocks(d, figs)
    write_blocks(b)
    meta = upstream_sources.apply(build_meta(
        "analysis/r1_pilot_decoding.py", WANDB_GROUPS, study_root=HERE.parent,
        code=[__file__, HERE / "upstream_sources.py"], inputs=[FROZEN, SESS] + upstream_sources.RECORDS))
    k3 = d["k3"]["results"]
    summary = {
        "k3_r2_latents": k3["pred|1_gru_latents"]["logistic"]["r2"],
        "k3_r2_within_latents": k3["pred|1_gru_latents"]["logistic"]["r2_within"],
        "k3_r2_history": k3["pred|3_history"]["logistic"]["r2"],
        "kscan_pooled_r2_latents": {K: d["kscan"]["parts"][f"hurdle__K{K}__latents"]["pooled_r2"] for K in range(2, 11)},
        "kscan_pooled_r2_within_latents": {K: d["kscan"]["parts"][f"hurdle__K{K}__latents"]["pooled_r2_within"]
                                           for K in range(2, 11)},
        "shuffle_max_within": {f"{f}_{n}_K{K}": max(p["pooled_r2_within"] for k, p in d["shuffle"].items()
                                                    if k.startswith(f"{f}_{n}_K{K}_"))
                               for f in ("latents", "history") for n in ("swap", "circ") for K in KS_NULL},
    }
    write_artifact(HERE / "r1_pilot_decoding.json", {"_meta": meta, **summary},
                   outputs=OUTPUTS + [f"{REPORT}#{k}" for k in b])


if __name__ == "__main__":
    main()
