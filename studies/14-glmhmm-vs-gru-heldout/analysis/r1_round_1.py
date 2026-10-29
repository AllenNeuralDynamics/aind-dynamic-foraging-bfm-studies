"""Producer for r1: study 14 round 1, from ``frozen_round1.json`` (offline).

Writes ``r1_round_1.json``, the figures ``fig_r1_*.svg`` and the result blocks of
``reports/r1-round-1.md``:
- result-1: the selection and held-out scores against K for both emissions, the lag scan,
  and study 02's 3-way GRU at D = 614 (read from its committed ``frozen_runs.json``);
- result-2: the states of each emission's selected K (argmax of the selection score);
- result-3: restart and subject-alignment diagnostics.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS = [
    "lag-scan@20261027-200129",
    "lag-scan@20261027-223410",
    "hurdle@20261027-212905",
    "reference@20261027-213006",
]
FROZEN = HERE / "frozen_round1.json"
# Study 02 r1's committed freeze: the 3-way GRU, D = 614 cells, 3 seeds per H.
GRU_SOURCE = HERE.parents[1] / "02-gru-scaling-law-ignore" / "analysis" / "frozen_runs.json"
GRU_GROUP = "nxd-3way@20260707-075401"
GRU_KEYS = {"matched": "eval_likelihood", "LR_engaged": "eval_likelihood_LR_engaged",
            "engage": "eval_likelihood_engage", "pr_auc": "engage_ignore_pr_auc"}
REPORT = HERE / "reports" / "r1-round-1.md"
EMISSIONS = ("hurdle", "reference")
COLORS = {"hurdle": fs.OKABE_ITO["blue"], "reference": fs.OKABE_ITO["vermillion"]}
STATE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
                "#000000", "#999999", "#882255"]
TOL = 1e-4   # nats per trial: a restart this close to the best counts as reaching it
SCORES = {   # label -> summary key
    "select": "glmhmm/select/eval_likelihood",
    "matched": "heldout/final/eval_likelihood",
    "zero_shot": "heldout/zero_shot/eval_likelihood",
    "LR_engaged": "heldout/final/eval_likelihood_LR_engaged",
    "engage": "heldout/final/eval_likelihood_engage",
    "pr_auc": "heldout/final/engage_ignore_pr_auc",
}


def gru_d614():
    """Study 02's 3-way GRU at D = 614: mean over seeds, per hidden size."""
    runs = [r for r in json.loads(GRU_SOURCE.read_text())["runs"]
            if r["D"] == 614 and r["wandb_group"] == GRU_GROUP]
    out = {}
    for H in sorted({r["H"] for r in runs}):
        cell = [r for r in runs if r["H"] == H]
        out[H] = {name: float(np.mean([r[key] for r in cell])) for name, key in GRU_KEYS.items()}
        out[H]["n_seeds"] = len(cell)
    return out


def global_same_mice(run):
    """The global fit alone on the training mice's eval sessions: how Faeze chose K."""
    return run["global_fit_scores"]["training_mice_eval"]["eval_likelihood"]


def grid(runs, emission):
    """The K-grid runs of one emission, by K."""
    return {r["num_states"]: r for r in runs
            if r["group"].startswith(emission + "@") and r["emission"] == emission}


def restart_stats(run):
    """How many restarts reached the best log joint (within TOL nats per trial)."""
    s = run["summary"]
    joints = np.asarray(s.get("glmhmm/global/restart_log_joints", []), float)
    n = s["glmhmm/global/num_train_trials"]
    converged = s.get("glmhmm/global/restart_converged", [s["glmhmm/global/converged"]])
    if len(joints) <= 1:
        return {"n_restarts": int(len(joints)), "n_at_best": int(len(joints)), "median_gap": 0.0,
                "max_gap": 0.0, "n_converged": int(sum(bool(c) for c in converged))}
    gap = (joints.max() - joints) / n
    return {"n_restarts": int(len(joints)), "n_at_best": int((gap <= TOL).sum()),
            "median_gap": float(np.median(gap)), "max_gap": float(gap.max()),
            "n_converged": int(sum(bool(c) for c in converged))}


def state_order(run):
    """States ordered by overall occupancy, largest first (display labels 1..K)."""
    return [int(k) for k in np.argsort(run["state_summary"]["occupancy"]["all"])[::-1]]


def state_table(run):
    """One row per state, in display order."""
    ss, A = run["state_summary"], np.asarray(run["global"]["transition_matrix"])
    rows = []
    for label, k in enumerate(state_order(run), start=1):
        st = ss["states"][k]
        rows.append({
            "state": label, "fitted_index": k,
            "occupancy": ss["occupancy"]["all"][k],
            "occupancy_mature": ss["occupancy"]["mature"][k],
            "occupancy_non_mature": ss["occupancy"]["non_mature"][k],
            "occupancy_stage_unknown": ss["occupancy"]["stage_unknown"][k],
            "dwell_expected": float(1 / max(1 - A[k, k], 1e-12)),
            "dwell_viterbi_mean": ss["dwell_viterbi"][k]["mean"],
            "predicted": dict(zip(("left", "right", "ignore"), st["predicted_class_probs"])),
            "observed": st["observed_viterbi"], "sessions_using": st["sessions_using"],
        })
    return rows


def lag_scan(runs):
    """Selection score by L, both emissions, from the two lag-scan launches."""
    out = {e: {} for e in EMISSIONS}
    for r in runs:
        if r["group"].startswith("lag-scan@"):
            out[r["emission"]][r["n_lags"]] = r["summary"][SCORES["select"]]
    return {e: dict(sorted(v.items())) for e, v in out.items()}


# ---------------------------------------------------------------------- figures

def fig_scores(grids, lags, gru, path):
    """Lag scan, selection score and held-out scores against K, with the GRU."""
    fig, axes = plt.subplots(2, 3, figsize=fs.mm(183, 100), layout="constrained")
    ax = axes[0, 0]
    for e in EMISSIONS:
        L = sorted(lags[e])
        ax.plot(L, [lags[e][x] for x in L], "o-", color=COLORS[e], label=e, ms=3)
    ax.set(xlabel="history length L (K = 1)", ylabel="selection score", xticks=[3, 5, 7, 10, 15])
    ax.legend(frameon=False)
    fs.panel_label(ax, "a")
    panels = [("select", "training mice, unseen sessions", None, "b"),
              ("matched", "held-out 3-way likelihood", "matched", "c"),
              ("LR_engaged", "held-out L/R on engaged trials", "LR_engaged", "d"),
              ("engage", "held-out engage vs ignore", "engage", "e"),
              ("pr_auc", "held-out ignore PR-AUC", "pr_auc", "f")]
    for (name, label, gru_key, letter), ax in zip(panels, axes.flat[1:]):
        for e in EMISSIONS:
            Ks = sorted(grids[e])
            ax.plot(Ks, [grids[e][k]["summary"][SCORES[name]] for k in Ks], "o-",
                    color=COLORS[e], ms=3, label=e)
            if name == "matched":
                ax.plot(Ks, [grids[e][k]["summary"][SCORES["zero_shot"]] for k in Ks], "o--",
                        color=COLORS[e], ms=2, alpha=0.6, label=f"{e}, global fit only")
            if name == "select":
                ax.plot(Ks, [global_same_mice(grids[e][k]) for k in Ks], "s--", color=COLORS[e],
                        ms=2, alpha=0.6, label=f"{e}, global fit only (Faeze's choice)")
        if gru_key:
            for H, ls in ((128, ":"), (256, "-.")):
                if H in gru:
                    ax.axhline(gru[H][gru_key], color="0.3", ls=ls, lw=0.8, label=f"GRU H{H}")
        ax.set(xlabel="states K", ylabel=label, xticks=range(1, 11))
        if name in ("select", "matched"):
            ax.legend(frameon=False, fontsize=5)
        fs.panel_label(ax, letter)
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_weights(run, path):
    """Global GLM weights of each state (display order), by history lag."""
    L, K, e = run["n_lags"], run["num_states"], run["emission"]
    em = run["global"]["emissions"]
    lags = np.arange(1, L + 1)
    fig, axes = plt.subplots(K, 3, figsize=fs.mm(150, 8 + 17 * K), layout="constrained",
                             sharex="col", squeeze=False, gridspec_kw={"width_ratios": [1, 1, 0.6]})
    occ = run["state_summary"]["occupancy"]["all"]
    for row, k in enumerate(state_order(run)):
        if e == "hurdle":
            side, eng = np.asarray(em["side_weights"][k]), np.asarray(em["engage_weights"][k])
            fam = {"RewC": side[:L], "UnrC": side[L:2 * L], "Ignored": eng[:L], "Rewarded": eng[L:2 * L]}
            biases = {"side": side[-1], "engage": eng[-1]}
            titles = ("side kernel, p(right | respond)", "engagement kernel, p(respond)", "biases")
        else:
            # (2, 4L+1): left, right logits against ignore = 0. The contrast R − L sets the side;
            # the common part (L + R) / 2 sets responding against ignoring.
            W = np.asarray(em["weights"][k])
            c, m = W[1] - W[0], (W[0] + W[1]) / 2
            fam = {"RewC": c[:L], "UnrC": c[L:2 * L], "Ignored": m[2 * L:3 * L], "Rewarded": m[3 * L:4 * L]}
            cross = {"RewC": m[:L], "UnrC": m[L:2 * L], "Ignored": c[2 * L:3 * L], "Rewarded": c[3 * L:4 * L]}
            biases = {"R − L": c[-1], "(L+R)/2": m[-1]}
            titles = ("side contrast R − L (dashed: common part)", "common part (L+R)/2 (dashed: contrast)",
                      "biases")
        a0, a1, a2 = axes[row]
        a0.plot(lags, fam["RewC"], "o-", ms=2, color=fs.OKABE_ITO["green"], label="RewC")
        a0.plot(lags, fam["UnrC"], "o-", ms=2, color=fs.OKABE_ITO["vermillion"], label="UnrC")
        a1.plot(lags, fam["Ignored"], "o-", ms=2, color="0.4", label="Ignored")
        a1.plot(lags, fam["Rewarded"], "o-", ms=2, color=fs.OKABE_ITO["blue"], label="Rewarded")
        if e != "hurdle":   # the reference emission's other component on the same regressors
            a0.plot(lags, cross["RewC"], "--", lw=0.7, color=fs.OKABE_ITO["green"])
            a0.plot(lags, cross["UnrC"], "--", lw=0.7, color=fs.OKABE_ITO["vermillion"])
            a1.plot(lags, cross["Ignored"], "--", lw=0.7, color="0.4")
            a1.plot(lags, cross["Rewarded"], "--", lw=0.7, color=fs.OKABE_ITO["blue"])
        for a in (a0, a1):
            a.axhline(0, color="0.7", lw=0.5)
        a2.bar(range(len(biases)), list(biases.values()), color=STATE_COLORS[row % 10], width=0.6)
        a2.set_xticks(range(len(biases)), list(biases.keys()))
        a2.axhline(0, color="0.7", lw=0.5)
        a0.set_ylabel(f"state {row + 1}\n{100 * occ[k]:.0f}% of trials", color=STATE_COLORS[row % 10])
        if row == 0:
            for a, t in zip((a0, a1, a2), titles):
                a.set_title(t)
            a0.legend(frameon=False, fontsize=5)
            a1.legend(frameon=False, fontsize=5)
    axes[-1, 0].set_xlabel("lag (trials back)")
    axes[-1, 1].set_xlabel("lag (trials back)")
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_states(run, path):
    """Transition matrix, occupancy by stage and across the session, behaviour per state."""
    K = run["num_states"]
    order = state_order(run)
    ss = run["state_summary"]
    A = np.asarray(run["global"]["transition_matrix"])[np.ix_(order, order)]
    fig, axes = plt.subplots(1, 4, figsize=fs.mm(183, 55), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 0.9, 1.1, 1.3]})
    ax = axes[0]
    ax.imshow(np.log10(np.maximum(A, 1e-4)), cmap="Greys", vmin=-4, vmax=0)
    for i in range(K):
        for j in range(K):
            if K <= 6 or i == j:
                ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=5,
                        color="white" if A[i, j] > 0.3 else "black")
    ax.set_xticks(range(K), range(1, K + 1))
    ax.set_yticks(range(K), range(1, K + 1))
    ax.set(xlabel="to state", ylabel="from state", title="transitions (log shading)")
    fs.panel_label(ax, "a")
    ax = axes[1]
    groups = [("all", "all"), ("mature", "mature"), ("non_mature", "non-mature"), ("stage_unknown", "n/a")]
    bottom = np.zeros(len(groups))
    for row, k in enumerate(order):
        vals = np.array([ss["occupancy"][g][k] for g, _ in groups])
        ax.bar(range(len(groups)), vals, bottom=bottom, color=STATE_COLORS[row % 10], width=0.7)
        bottom += vals
    ax.set_xticks(range(len(groups)), [lab for _, lab in groups], rotation=30)
    ax.set(ylabel="occupancy", ylim=(0, 1), title="by training stage")
    fs.panel_label(ax, "b")
    ax = axes[2]
    dec = np.asarray(ss["occupancy_by_session_decile"])[:, order]
    ax.stackplot((np.arange(10) + 0.5) / 10, dec.T, colors=[STATE_COLORS[i % 10] for i in range(K)])
    ax.set(xlabel="position in session", ylabel="occupancy", xlim=(0.05, 0.95), ylim=(0, 1),
           title="within a session")
    fs.panel_label(ax, "c")
    ax = axes[3]
    measures = [("p_ignore", "p(ignore)"), ("p_right_given_respond", "p(R | resp)"),
                ("p_stay_after_reward", "stay | reward"), ("p_stay_after_no_reward", "stay | no rew."),
                ("p_choose_better", "p(better side)")]
    for row, k in enumerate(order):
        obs = ss["states"][k]["observed_viterbi"]
        y = [obs[m] if obs[m] is not None else np.nan for m, _ in measures]
        ax.plot(range(len(measures)), y, "o-", ms=2.5, lw=0.6, color=STATE_COLORS[row % 10],
                label=str(row + 1))
    ax.set_xticks(range(len(measures)), [lab for _, lab in measures], rotation=30)
    ax.set(ylim=(0, 1), ylabel="rate on the state's trials", title="behaviour (Viterbi trials)")
    ax.legend(frameon=False, fontsize=5, ncol=2, title="state", title_fontsize=5)
    fs.panel_label(ax, "d")
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_examples(examples, run, path):
    """Example held-out sessions, drawn with the lab's session plot, and the state posterior."""
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session

    order = state_order(run)
    K = run["num_states"]
    n = len(examples)
    fig = plt.figure(figsize=fs.mm(183, 52 * n), layout="constrained")
    gs = fig.add_gridspec(2 * n, 1, height_ratios=[2.2, 0.7] * n)
    pairs = []
    for i, (ses, ex) in enumerate(sorted(examples.items())):
        choice = np.asarray(ex["choice"], float)
        choice[choice == 2] = np.nan                       # the util's code for an ignored trial
        pred = ex["runs"][run["wandb_run_id"]]
        holder = fig.add_subplot(gs[2 * i])        # the util replaces it with its own two axes
        _, (ax_choice, ax_schedule) = plot_foraging_session(
            choice, np.asarray(ex["reward"], bool), np.vstack([ex["pL"], ex["pR"]]),
            fitted_data=np.asarray(pred["p_right_given_respond"]), ax=holder)
        if holder.figure is not None:
            holder.remove()
        if i > 0:   # the util's legends once, on the first session
            for ax in (ax_choice, ax_schedule):
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        ax_choice.set_title(f"mouse {ex['subject_id']}, {ex['stage']}, {len(choice)} trials, "
                            f"{100 * np.mean(np.isnan(choice)):.0f}% ignored; overlay: the global "
                            "fit's p(right | respond)", loc="left", pad=14)
        fs.panel_label(ax_choice, "abcdefgh"[i], dy_pt=14)
        bot = fig.add_subplot(gs[2 * i + 1])
        t = np.arange(1, len(choice) + 1)                  # the util numbers trials from 1
        post = np.asarray(pred["smoothed"])[:, order]
        bot.stackplot(t, post.T, colors=[STATE_COLORS[j % 10] for j in range(K)], lw=0,
                      labels=[f"state {j + 1}" for j in range(K)])
        bot.set(ylim=(0, 1), ylabel="p(state)")
        if i == 0:
            bot.legend(frameon=False, fontsize=5, ncol=K, loc="lower right")
        pairs.append((ax_choice, bot))
    # The util lays out its own axes, so after layout pin each posterior strip to its session's
    # x position and range: trial t sits at the same x in both.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax_choice, bot in pairs:
        top_box, box = ax_choice.get_position(), bot.get_position()
        bot.set_position([top_box.x0, box.y0, top_box.width, box.height])
        bot.set_xlim(ax_choice.get_xlim())
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_timescales(grids, path):
    """How long states last, how often sessions switch, and whether states belong to a few mice."""
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(165, 52), layout="constrained")
    for e in EMISSIONS:
        Ks = [K for K in sorted(grids[e]) if K > 1 and "state_summary" in grids[e][K]]
        if not Ks:
            continue
        sw = [grids[e][K]["state_summary"]["switching_viterbi"]["share_with_a_switch"] for K in Ks]
        dwell = []
        conc = []
        for K in Ks:
            run = grids[e][K]
            A = np.asarray(run["global"]["transition_matrix"])
            occ = np.asarray(run["state_summary"]["occupancy"]["all"])
            dwell.append(float((occ / (1 - np.diag(A))).sum()))
            conc.append(float(max(run["state_summary"]["top10pct_subject_share"])))
        axes[0].plot(Ks, sw, "o-", color=COLORS[e], ms=3, label=e)
        axes[1].plot(Ks, dwell, "o-", color=COLORS[e], ms=3, label=e)
        axes[2].plot(Ks, conc, "o-", color=COLORS[e], ms=3, label=e)
    axes[0].set(xlabel="states K", ylabel="held-out sessions with a switch", ylim=(0, 1))
    axes[1].set(xlabel="states K", ylabel="expected dwell (trials)")
    axes[2].axhline(0.1, color="0.6", ls=":", lw=0.7)
    axes[2].set(xlabel="states K", ylabel="top 10% of mice's share\nof per-mouse mean occupancy", ylim=(0, 1),
                title="most mouse-specific state")
    axes[0].legend(frameon=False)
    for ax, letter in zip(axes, "abc"):
        ax.set_xticks(range(2, 11))
        fs.panel_label(ax, letter)
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_restarts(grids, path):
    """Restart log joints relative to the best, per K."""
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(square_panels=2), 50), layout="constrained")
    for off, e in ((-0.15, "hurdle"), (0.15, "reference")):
        for K, run in sorted(grids[e].items()):
            s = run["summary"]
            j = np.asarray(s.get("glmhmm/global/restart_log_joints", []), float)
            if len(j) > 1:
                gap = (j.max() - j) / s["glmhmm/global/num_train_trials"]
                ax.scatter(np.full(len(gap), K + off), np.maximum(gap, 1e-7), s=3, color=COLORS[e],
                           alpha=0.6, label=e if K == min(k for k in grids[e] if k > 1) else None)
    ax.axhline(TOL, color="0.5", ls=":", lw=0.7)
    ax.set_yscale("log")
    fs.plain_log_axis(ax, "y")
    ax.set(xlabel="states K", ylabel="gap to the best restart (nats/trial)", xticks=range(2, 11),
           title="20 restarts per K (floored at 1e-7)")
    ax.legend(frameon=False)
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_state_map(grids, selected, path):
    """Every state of every K, placed by its behaviour: what the states are, and how they split."""
    fig, axes = plt.subplots(2, 2, figsize=fs.mm(150, 115), layout="constrained")
    cmap = plt.get_cmap("viridis")
    panels = (("p_ignore", "stay_after_no_reward", "p(ignore)", "stay after no reward (perseverance)"),
              ("p_stay_after_reward", "p_choose_better", "stay after reward (win-stay)",
               "p(choose the better side)"))
    for row, e in enumerate(EMISSIONS):
        for col, (xm, ym, xl, yl) in enumerate(panels):
            ax = axes[row, col]
            for K, run in sorted(grids[e].items()):
                if "state_summary" not in run:
                    continue
                ss = run["state_summary"]
                for k, st in enumerate(ss["states"]):
                    obs = st["observed_viterbi"]
                    x = obs["p_ignore"] if xm == "p_ignore" else obs[xm]
                    y = obs["p_stay_after_no_reward"] if ym == "stay_after_no_reward" else obs[ym]
                    if x is None or y is None:
                        continue
                    occ = ss["occupancy"]["all"][k]
                    is_sel = K == selected.get(e)
                    ax.scatter(x, y, s=4 + 120 * occ, color=cmap((K - 1) / 9), alpha=0.85,
                               edgecolor="black" if is_sel else "none", linewidth=0.8, zorder=3 if is_sel else 2)
                    if is_sel:
                        label = state_order(run).index(k) + 1
                        ax.annotate(str(label), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6)
            ax.set(xlabel=xl, ylabel=yl)
            ax.set_title(f"{e}: states of every K (ringed: selected K = {selected.get(e)})", loc="left")
            fs.panel_label(ax, "abcd"[2 * row + col])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 10))
    fig.colorbar(sm, ax=axes, shrink=0.5, label="K (dot size = occupancy)")
    fs.save_figure(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------- report

def replace_block(text, n, body):
    """Replace the content between the result-n markers."""
    pattern = re.compile(rf"(<!-- BEGIN result-{n} -->\n).*?(\n<!-- END result-{n} -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"result-{n} markers missing")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def f4(x):
    """Four decimals, or a dash."""
    return "—" if x is None else f"{x:.4f}"


def block_scores(grids, lags, gru, selected):
    """result-1: scores against K, the lag scan and the GRU."""
    lines = ["Each score is the geometric-mean probability given to the mouse's actual choice on "
             "sessions never fitted (chance 1/3). *Refit*: training mice, after the per-mouse refit "
             "(study 14's selection score; bold = each emission's best). *Global fit, same mice*: "
             "training mice, global fit only (Faeze's way of choosing K). *Held-out*: 149 new mice, "
             "with refit (matched, the GRU's protocol) and global fit only (zero-shot). GRU = study "
             "02 r1's 3-way GRU at D = 614, mean of 3 seeds. One seed per GLM-HMM cell, so no error "
             "bars.",
             "", "| emission | K | refit (selection) | global fit, same mice | held-out, refit | held-out, global fit | held-out refit: L/R engaged | engage | ignore PR-AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in EMISSIONS:
        for K, run in sorted(grids[e].items()):
            s = run["summary"]
            sel = f4(s[SCORES["select"]])
            if K == selected.get(e):
                sel = f"**{sel}**"
            lines.append(f"| {e} | {K} | {sel} | {f4(global_same_mice(run))} | {f4(s[SCORES['matched']])} | {f4(s[SCORES['zero_shot']])} | "
                         f"{f4(s[SCORES['LR_engaged']])} | {f4(s[SCORES['engage']])} | {f4(s[SCORES['pr_auc']])} |")
    for H, g in gru.items():
        lines.append(f"| GRU H{H} (study 02) | — | — | — | {f4(g['matched'])} | — | {f4(g['LR_engaged'])} | "
                     f"{f4(g['engage'])} | {f4(g['pr_auc'])} |")
    Ls = sorted(set(lags["hurdle"]) | set(lags["reference"]))
    lines += ["", "**Lag scan** (K = 1, selection score; L = 15 from the follow-up launch):", "",
              "| emission | " + " | ".join(f"L = {L}" for L in Ls) + " |",
              "|---|" + "---|" * len(Ls)]
    for e in EMISSIONS:
        lines.append(f"| {e} | " + " | ".join(f4(lags[e].get(L)) for L in Ls) + " |")
    lines += ["", "![scores](../fig_r1_scores.svg)"]
    return "\n".join(lines)


def block_states(grids, selected, tables):
    """result-2: the states of each emission's selected K."""
    lines = []
    for e in EMISSIONS:
        K = selected.get(e)
        if K is None:
            continue
        rid = grids[e][K]["wandb_run_id"]
        lines += [f"**{e}, K = {K}** (run [{rid}](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/{rid})). "
                  "States ordered by occupancy: the mean smoothed posterior, under the global fit, over "
                  "the held-out mice's trials (both splits). Dwell: expected 1/(1 − A_kk) from the global transitions, "
                  "and the mean Viterbi run. *Predicted*: the global model's p(left, right, ignore), "
                  "averaged over the state's posterior weight. *Observed*: rates on the trials "
                  "Viterbi assigns to the state.", "",
                  "| state | occupancy | mature / non-mature / n.a. | dwell exp. / obs. | predicted L / R / I | "
                  "obs. p(ignore) | obs. p(R \\| resp) | stay after rew. / no rew. | p(better side) | sessions using ≥10% |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in tables[e]:
            p, o = r["predicted"], r["observed"]

            def g(x, spec=".2f"):
                return "—" if x is None else format(x, spec)
            lines.append(
                f"| {r['state']} | {r['occupancy']:.3f} | {r['occupancy_mature']:.2f} / "
                f"{r['occupancy_non_mature']:.2f} / {r['occupancy_stage_unknown']:.2f} | "
                f"{r['dwell_expected']:.0f} / {g(r['dwell_viterbi_mean'], '.0f')} | "
                f"{p['left']:.2f} / {p['right']:.2f} / {p['ignore']:.2f} | {g(o['p_ignore'])} | "
                f"{g(o['p_right_given_respond'])} | {g(o['p_stay_after_reward'])} / "
                f"{g(o['p_stay_after_no_reward'])} | {g(o['p_choose_better'])} | {r['sessions_using']:.2f} |")
        lines += ["", f"![{e} weights](../fig_r1_weights_{e}.svg)", "",
                  f"![{e} states](../fig_r1_states_{e}.svg)", "",
                  f"![{e} example sessions](../fig_r1_examples_{e}.svg)", ""]
    lines += ["**Across K.** Every state of every K = 2–10 fit, placed by its behaviour under the "
              "global fit, shows what the states are and how they split as K grows:", "",
              "![state map](../fig_r1_state_map.svg)", "",
              "How long states last, how often a held-out session switches state, and how much of "
              "the most mouse-specific state's use the top 10% of mice hold (each mouse's mean "
              "occupancy; dotted line: 10%, equal use):", "",
              "![timescales](../fig_r1_timescales.svg)", ""]
    return "\n".join(lines)


def block_diagnostics(grids):
    """result-3: restarts, convergence and subject alignment."""
    lines = [f"*Restarts at best*: final log joint within {TOL:g} nats/trial of the best restart. "
             "*Aligned subjects*: held-out mice whose refit states still match the global states, by "
             "weight distance, in the same order. As K grows the refits reorder or repurpose states, "
             "which is why the state summaries use the global fit.", "",
             "| emission | K | restarts at best | median gap (nats/trial) | restarts converged | EM iters (best) | aligned subjects | run time (h) |",
             "|---|---|---|---|---|---|---|---|"]
    for e in EMISSIONS:
        for K, run in sorted(grids[e].items()):
            rs, s = restart_stats(run), run["summary"]
            al = run["state_summary"]["alignment"]
            lines.append(f"| {e} | {K} | {rs['n_at_best']}/{rs['n_restarts']} | {rs['median_gap']:.1e} | "
                         f"{rs['n_converged']}/{rs['n_restarts']} | {s['glmhmm/global/em_iters']} | "
                         f"{al['identity_matched']}/{al['n_subjects']} | {s['glmhmm/fit_seconds'] / 3600:.1f} |")
    lines += ["", "![restarts](../fig_r1_restarts.svg)"]
    return "\n".join(lines)


def _paired(base, other):
    """Per-subject matched likelihood: on how many mice ``other`` beats ``base``."""
    a = dict(zip(base["per_subject"]["subject_id"], base["per_subject"]["eval_likelihood"]))
    b = dict(zip(other["per_subject"]["subject_id"], other["per_subject"]["eval_likelihood"]))
    common = sorted(set(a) & set(b))
    diff = np.array([b[s] - a[s] for s in common])
    return {"n_subjects": len(common), "n_better": int((diff > 0).sum()),
            "median_diff": float(np.median(diff)), "mean_diff": float(diff.mean())}


def main() -> None:
    frozen = json.loads(FROZEN.read_text())
    runs = frozen["runs"]
    grids = {e: grid(runs, e) for e in EMISSIONS}
    lags = lag_scan(runs)
    gru = gru_d614()
    selected = {e: max(g, key=lambda k: g[k]["summary"][SCORES["select"]]) for e, g in grids.items() if g}
    tables = {e: state_table(grids[e][K]) for e, K in selected.items()}

    fs.apply_paper_style()
    figs = [HERE / "fig_r1_scores.svg", HERE / "fig_r1_restarts.svg"]
    fig_scores(grids, lags, gru, figs[0])
    fig_restarts(grids, figs[1])
    figs.append(HERE / "fig_r1_state_map.svg")
    fig_state_map(grids, selected, figs[-1])
    figs.append(HERE / "fig_r1_timescales.svg")
    fig_timescales(grids, figs[-1])
    for e, K in selected.items():
        run = grids[e][K]
        for name in ("weights", "states", "examples"):
            path = HERE / f"fig_r1_{name}_{e}.svg"
            if name == "weights":
                fig_weights(run, path)
            elif name == "states":
                fig_states(run, path)
            else:
                fig_examples(frozen["examples"], run, path)
            figs.append(path)

    text = REPORT.read_text()
    text = replace_block(text, 1, block_scores(grids, lags, gru, selected))
    text = replace_block(text, 2, block_states(grids, selected, tables))
    text = replace_block(text, 3, block_diagnostics(grids))
    REPORT.write_text(text)

    result = {
        "selected_K": selected,
        "scores": {e: {K: {name: r["summary"][key] for name, key in SCORES.items()}
                       for K, r in sorted(g.items())} for e, g in grids.items()},
        "lag_scan": lags,
        "gru_d614": gru,
        "states": tables,
        "restarts": {e: {K: restart_stats(r) for K, r in sorted(g.items())} for e, g in grids.items()},
        "per_subject_matched_selected_vs_k1": {
            e: _paired(grids[e][1], grids[e][K]) for e, K in selected.items() if 1 in grids[e]},
        "per_subject_hurdle_vs_reference": (
            _paired(grids["reference"][selected["reference"]], grids["hurdle"][selected["hurdle"]])
            if len(selected) == 2 else None),
    }
    meta = build_meta("analysis/r1_round_1.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN, GRU_SOURCE])
    write_artifact(HERE / "r1_round_1.json", {"_meta": meta, **result},
                   outputs=figs + [f"{REPORT}#result-{n}" for n in (1, 2, 3)])


if __name__ == "__main__":
    main()
