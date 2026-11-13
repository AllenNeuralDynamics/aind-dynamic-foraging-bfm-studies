#!/usr/bin/env python
"""r4: simplification at three floors, each floor's program on the selection and test subjects. Offline.

Reads the three frozen runs (``simplify/<variant>/run.json``), the floors (``simplify/floors.json``), the
floor-program choice (``simplify_choice.json``), the test scores of ``freeze_test.py`` and the curves and
Halstead efforts of ``freeze_test_curves.py`` (``test/curves.json``), and, through r3's helpers, the
comparators every test score is set against (HB-CTT, study 13 r5; GRU v2 H128, study 01; study 15's
curves). Writes ``r4_simplification.json``, two figures and the report's result blocks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
import r3_search_best as r3  # noqa: E402

VARIANTS = ("simplify-90", "simplify-75", "simplify-50")
RUNS = {v: HERE / "simplify" / v / "run.json" for v in VARIANTS}
FLOORS, CHOICE, CURVES = HERE / "simplify" / "floors.json", HERE / "simplify_choice.json", HERE / "test" / "curves.json"
REPORT = HERE / "reports" / "r4-simplification.md"
FIGS = {"frontier": HERE / "fig_r4_frontier.svg", "curves": HERE / "fig_r4_curves.svg"}
OI = fs.OKABE_ITO
COL = {"simplify-90": OI["blue"], "simplify-75": OI["green"], "simplify-50": OI["vermillion"]}
SHARE = {"simplify-90": "90%", "simplify-75": "75%", "simplify-50": "50%"}
SEARCH_BEST, REFERENCE = r3.CHOSEN, "reference_miller_rhg"

# A short description and the equations of each program on the atlas leaderboard, written from its code
# (the file named here); the atlas shows them in the program's side panel. Notation as study 12's equations.
NOTATION = (r"Notation: trial $t$; choice $c_t \in \{L, R\}$, $u_t$ the side not chosen; reward $r_t \in \{0, 1\}$; "
            r"signed choice $s_t = +1$ (right) or $-1$ (left); $\sigma(x) = 1/(1+e^{-x})$; learning rate "
            r"$\alpha_t = \alpha_+$ after a reward, $\alpha_-$ after none. Latent variables with index $t$ are the ones "
            r"used to choose on trial $t$; they are updated after $c_t, r_t$ are seen.")
CARDS = {
    SEARCH_BEST: {
        "program": "analysis/selection/programs/rank3_e3819885.py",
        "summary": "Values learned at separate rates after reward and no reward. The unchosen side is forgotten "
                   "toward 0 and, after a miss, also learns that the other side pays (counterfactual learning, "
                   "with its own strength in coupled and uncoupled tasks). Fast and slow choice traces and a side "
                   "bias. The weight on values and traces changes with the rewards collected in the session "
                   "(satiety). Its docstring also mentions a bait expectation; the code has none.",
        "equations": NOTATION + r"""

**Update:**

$$\begin{aligned} Q_{t+1}(c_t) &= Q_t(c_t) + \alpha_t\,\big(r_t - Q_t(c_t)\big) \\ Q_{t+1}(u_t) &= (1-\phi)\,Q_t(u_t) + \kappa\,\alpha_t\,\big((1 - r_t) - (1-\phi)\,Q_t(u_t)\big) \\ F_{t+1} &= F_t + \tau_F\,(s_t - F_t) \\ S_{t+1} &= S_t + \tau_S\,(s_t - S_t) \\ n_{t+1} &= n_t + r_t/100 \end{aligned}$$

with $\kappa = \kappa_c$ in coupled tasks and $\kappa_u$ in uncoupled ones; $n_t$ is the rewards collected so far, in hundreds.

**Choice** ($\epsilon = 10^{-4}$):

$$\begin{aligned} g_t &= \exp\!\big(-\gamma\,(n_t - 1)\big) \\ P(c_t = R) &= \epsilon + (1 - 2\epsilon)\,\sigma\Big(g_t\,\big[\beta\,(Q_t(R) - Q_t(L)) + w_F F_t + w_S S_t\big] + b\Big) \end{aligned}$$

**Initial values:** every latent variable is 0 at the start of each session.

**Parameters (12):** $\alpha_+$ `learn_rate_rewarded`, $\alpha_-$ `learn_rate_unrewarded`, $\phi$ `forget_rate_unchosen`, $\beta$ `value_inverse_temperature`, $\tau_F$ `fast_habit_step_size`, $w_F$ `fast_habit_weight`, $\tau_S$ `slow_habit_step_size`, $w_S$ `slow_habit_weight`, $b$ `right_bias`, $\kappa_c$ `counterfactual_weight`, $\kappa_u$ `counterfactual_weight_uncoupled`, $\gamma$ `satiety_gain`.""",
    },
    "simplify-75_c1_141c8a13": {
        "program": "analysis/simplify/simplify-75/programs/c1_141c8a13.py",
        "summary": "The 50% floor's model with one addition: the fast choice trace's weight changes with the "
                   "mouse's training experience (the log of the session's index). Values and traces are kept in "
                   "logit units, so the choice is their plain sum.",
        "equations": NOTATION + r"""

**Session start:** the fast trace's weight for session $k$ (0 for the mouse's first session) is

$$w_F^{(k)} = w_F + \gamma\,\big(\ln(1 + k) - 3\big)$$

($\ln(1+k) = 3$ near the 20th session).

**Update:**

$$\begin{aligned} Q_{t+1}(c_t) &= Q_t(c_t) + \alpha_t\,\big(\beta\,r_t - Q_t(c_t)\big) \\ Q_{t+1}(u_t) &= (1-\phi)\,Q_t(u_t) \\ F_{t+1} &= F_t + \tau_F\,\big(w_F^{(k)}\,s_t/2 - F_t\big) \\ S_{t+1} &= S_t + \tau_S\,\big(w_S\,s_t/2 - S_t\big) \end{aligned}$$

**Choice:**

$$P(c_t = R) = \sigma\big(Q_t(R) - Q_t(L) + F_t + S_t + b\big)$$

**Initial values:** $Q$, $F$ and $S$ are 0 at the start of each session.

**Parameters (10):** $\alpha_+$ `learn_rate_rewarded`, $\alpha_-$ `learn_rate_unrewarded`, $\phi$ `forget_rate_unchosen`, $\beta$ `value_inverse_temperature`, $\tau_F$ `fast_habit_step_size`, $w_F$ `fast_habit_weight`, $\tau_S$ `slow_habit_step_size`, $w_S$ `slow_habit_weight`, $b$ `right_bias`, $\gamma$ `habit_experience_gain`.""",
    },
    "simplify-50_c1_cd994c73": {
        "program": "analysis/simplify/simplify-50/programs/c1_cd994c73.py",
        "summary": "Values learned at separate rates after reward and no reward, with the unchosen side forgotten "
                   "toward 0; fast and slow choice traces that pull toward repeating (or alternating) recent "
                   "choices; a side bias; a softmax over their weighted sum.",
        "equations": NOTATION + r"""

**Update:**

$$\begin{aligned} Q_{t+1}(c_t) &= Q_t(c_t) + \alpha_t\,\big(r_t - Q_t(c_t)\big) \\ Q_{t+1}(u_t) &= (1-\phi)\,Q_t(u_t) \\ F_{t+1} &= F_t + \tau_F\,(s_t - F_t) \\ S_{t+1} &= S_t + \tau_S\,(s_t - S_t) \end{aligned}$$

**Choice:**

$$P(c_t = R) = \sigma\big(\beta\,(Q_t(R) - Q_t(L)) + w_F F_t + w_S S_t + b\big)$$

**Initial values:** every latent variable is 0 at the start of each session.

**Parameters (9):** $\alpha_+$ `learn_rate_rewarded`, $\alpha_-$ `learn_rate_unrewarded`, $\phi$ `unchosen_forgetting`, $\beta$ `beta`, $\tau_F$ `fast_trace_step`, $w_F$ `fast_trace_weight`, $\tau_S$ `slow_trace_step`, $w_S$ `slow_trace_weight`, $b$ `right_bias`.""",
    },
}


def table(head, rows):
    return r3.table(head, rows)


def write_blocks(blocks: dict[int, str]) -> None:
    text = REPORT.read_text()
    for k, body in blocks.items():
        start, end = f"<!-- BEGIN result-{k} -->", f"<!-- END result-{k} -->"
        assert text.count(start) == 1 and text.count(end) == 1, f"result-{k} markers"
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + body.strip() + "\n" + text[j:]
    REPORT.write_text(text)


def test_scores(names: list[str]) -> dict:
    """Pooled likelihood on the 146 shared test subjects for each program, above HB-CTT, with paired SEs."""
    import math

    hb = {r["subject_id"]: (r["heldout/eval_log_lik_sum"], r["heldout/num_eval_trials"])
          for r in json.loads(r3.HB.read_text())["rows"]}
    rows = [r for r in json.loads(r3.GRU.read_text())["rows"] if r["variant"] == "v2" and r["ratio"] == 1.0]
    seeds = sorted({r["seed"] for r in rows})
    gru = {s: {r["subject"]: (r["n_trials"] * math.log(r["ll"]), r["n_trials"]) for r in rows if r["seed"] == s}
           for s in seeds}
    progs = {n: r3.used_ll(json.loads((HERE / "test" / "test" / f"{n}.json").read_text())) for n in names}
    shared = sorted(set(progs[names[0]]) & set(gru[seeds[0]]))
    assert len(shared) == 146
    for n in names:
        assert all(progs[n][s][1] == hb[s][1] == gru[seeds[0]][s][1] for s in shared), f"{n}: trial counts differ"
    n = np.array([hb[s][1] for s in shared], float)
    H = np.array([hb[s][0] for s in shared])
    G = np.array([[gru[k][s][0] for s in shared] for k in seeds])
    P = {k: np.array([progs[k][s][0] for s in shared]) for k in names}

    def stats(idx=None):
        h = r3.pooled(H, n, idx)
        g = float(np.mean([r3.pooled(x, n, idx) for x in G]))
        return h, g, {k: r3.pooled(v, n, idx) for k, v in P.items()}

    h, g, p = stats()
    rng = np.random.default_rng(0)
    boot = [stats(rng.integers(0, len(shared), len(shared))) for _ in range(r3.N_BOOT)]
    gru149 = float(np.mean([np.exp(sum(v[0] for v in gru[s].values()) / sum(v[1] for v in gru[s].values()))
                            for s in seeds]))
    out = {"hb_ctt_x100": 100 * h, "gru_x100": 100 * g, "gru_gru149_x100": 100 * gru149, "n_subjects": len(shared),
           "programs": {}}
    for k in names:
        gap = np.array([(b[2][k] - b[0]) / (b[1] - b[0]) for b in boot])
        out["programs"][k] = {"pooled_x100": 100 * p[k], "above_hb_x100": 100 * (p[k] - h),
                              "above_hb_paired_se_x100": 100 * float(np.std([b[2][k] - b[0] for b in boot])),
                              "gap_closed": (p[k] - h) / (g - h), "gap_closed_ci95": np.percentile(gap, [2.5, 97.5]).tolist(),
                              "gru149_x100": 100 * json.loads((HERE / "test" / "gru_test" / f"{k}.json").read_text())["score"]["score"]
                              if (HERE / "test" / "gru_test" / f"{k}.json").exists() else None}
    return out


def scores_149(names: list[str]) -> dict:
    """Each program's pooled likelihood on all 149 test subjects, the atlas leaderboard's metric: the raw
    2-level EM score (study 12 r3's convention), the score with the per-mouse fallback, and per subject."""
    out = {}
    for k in names:
        d = json.loads((HERE / "test" / "test" / f"{k}.json").read_text())
        s = d["score"]
        out[k] = {"n_subjects": d["n_subjects"], "two_level": s["two_level"], "with_fallback": s["score"],
                  "per_subject": s["per_subject"], "n_fallback": s["n_fallback"], "two_level_failed": s["two_level_failed"]}
    assert {v["n_subjects"] for v in out.values()} == {149}
    for k, card in CARDS.items():
        if k in out:
            assert (STUDY / card["program"]).is_file(), card["program"]
            out[k].update(card)
    return out


def main() -> None:
    runs = {v: json.loads(p.read_text()) for v, p in RUNS.items()}
    floors = json.loads(FLOORS.read_text())
    choice = json.loads(CHOICE.read_text())["floors"]
    curves = json.loads(CURVES.read_text())
    efforts = curves["efforts"]
    chosen = {v: choice[v]["chosen"] for v in VARIANTS}
    tested = [c for c in chosen.values() if c and (HERE / "test" / "test" / f"{c}.json").exists()]
    names = [REFERENCE, SEARCH_BEST] + tested
    ts = test_scores(names)
    s149 = scores_149(names)

    # ---------- result 1: the runs
    rows1 = []
    for v in VARIANTS:
        r = runs[v]
        props = [p for p in r["programs"] if p["generation"] > 0]
        ok = [p for p in props if p["correct"] and p["effort"] is not None]
        low = min(ok, key=lambda p: p["effort"])
        rows1.append([f"`{v}`", f"{r['floor_x100']:.3f}", f"{len(props):,}", f"{len(ok):,}", f"${r['summary']['cost_usd']:.2f}",
                      f"{r['seed_effort_min']:,.0f}", f"**{low['effort']:,.0f}**", f"{low['validation_x100']:.3f}",
                      f"{low['n_params']}"])
    ref_eff, best_eff = efforts[REFERENCE], efforts[SEARCH_BEST]
    b1 = ("![frontier](../fig_r4_frontier.svg)\n\n"
          + table(["run", "floor (×100)", "proposals", "above the floor", "Claude cost", "seeds' lowest effort",
                   "lowest effort above the floor", "its validation score", "parameters"], rows1)
          + f"\n\nFor scale: the search's best has Halstead effort {best_eff:,.0f}, Miller-RHG {ref_eff:,.0f}. Every run "
            f"spent its $40 cap while still finding simpler programs.\n\n"
            "**Figure 1.** (a) Every proposal above its run's floor: validation score against Halstead effort (lower is "
            "simpler), with each run's floor (dashed) and its frontier (the simplest program at each score); star: the "
            "search's best, square: Miller-RHG. (b) On the 146 shared test subjects: pooled likelihood above HB-CTT "
            "against effort for Miller-RHG (2-level EM), the search's best and each floor's program; dashed: the GRU; "
            "bars: paired bootstrap SE.")

    # ---------- result 2: the choice and the test
    rows2 = []
    for v in VARIANTS:
        c = choice[v]
        for x in c["candidates"]:
            mark = "**chosen**" if x["name"] == c["chosen"] else ""
            rows2.append([f"`{v}`", str(x["rank"]), f"{x['effort']:,.0f}", f"{x['score_x100']:.3f}",
                          f"{c['selection_floor_x100']:.3f}", "pass" if x["passes_gate"] else "fail",
                          "yes" if x["reaches_floor"] else "no", mark])
    rows3 = [["Miller-RHG (2-level EM)", f"{ref_eff:,.0f}", f"{ts['programs'][REFERENCE]['pooled_x100']:.3f}",
              f"{r3.fs.num(ts['programs'][REFERENCE]['above_hb_x100'], '+.3f')} ± {ts['programs'][REFERENCE]['above_hb_paired_se_x100']:.3f}",
              f"{ts['programs'][REFERENCE]['gap_closed']:.2f}", "—"]]
    for label, k in [("search's best", SEARCH_BEST)] + [(f"{SHARE[v]} floor", chosen[v]) for v in VARIANTS if chosen[v] in tested]:
        t = ts["programs"][k]
        rows3.append([label, f"{efforts[k]:,.0f}", f"**{t['pooled_x100']:.3f}**",
                      f"{r3.fs.num(t['above_hb_x100'], '+.3f')} ± {t['above_hb_paired_se_x100']:.3f}",
                      f"**{t['gap_closed']:.2f}** [{t['gap_closed_ci95'][0]:.2f}, {t['gap_closed_ci95'][1]:.2f}]",
                      "—" if t["gru149_x100"] is None else f"{t['gru149_x100']:.3f}"])
    missing = [v for v in VARIANTS if not chosen[v]]
    b2 = (table(["run", "candidate", "effort", "selection score", "selection floor", "gate", "reaches floor", ""], rows2)
          + ("\n\n" + "; ".join(f"`{v}`: no candidate reached the selection floor, so no program goes to test"
                                for v in missing) + "." if missing else "")
          + f"\n\nOn the {ts['n_subjects']} shared test subjects (HB-CTT {ts['hb_ctt_x100']:.3f}, GRU {ts['gru_x100']:.3f}); "
            f"the last column is each program on the GRU's own 149 (the GRU there: {ts['gru_gru149_x100']:.3f}):\n\n"
          + table(["program", "Halstead effort", "test (×100)", "above HB-CTT ± paired SE", "gap closed [95% CI]",
                   "GRU's 149 (×100)"], rows3))

    # ---------- result 3: the programs' code
    b3 = []
    for v in VARIANTS:
        if not chosen[v]:
            continue
        f = STUDY / choice[v]["chosen_file"]
        b3.append(f"**{SHARE[v]} floor** (`{chosen[v]}`, effort {efforts.get(chosen[v], float('nan')):,.0f}):\n\n```python\n"
                  + f.read_text().strip() + "\n```")
    b3 = "\n\n".join(b3)

    # ---------- result 4: mouse-average curves
    rm = {}
    for mode, path in r3.S15.items():
        r = json.loads(path.read_text())
        cells = r["models"]["GRU-D614"]
        seeds = [cells[k][r["headline_set"]].get("statistics", cells[k][r["headline_set"]]) for k in sorted(cells)
                 if k.startswith("seed")] or [cells["per_subject"][r["headline_set"]].get("statistics")]
        gru = {}
        for st in seeds[0]:
            pb = [c[st]["per_bin_common"] for c in seeds]
            model = [None if all(x is None for x in xs) else float(np.mean([x for x in xs if x is not None]))
                     for xs in zip(*(q["model"] for q in pb))]
            gru[st] = r3._rmse(model, pb[0]["subject"])
        rm[mode] = {"GRU": gru, **{k: {st: r3._rmse(c["model"], c["subject"]) for st, c in curves["programs"][k][mode].items()}
                                   for k in names}}
    rows4 = []
    for mode in ("replay", "rollout"):
        for st in rm[mode]["GRU"]:
            rows4.append([mode, st] + [f"{rm[mode][k][st]:.4f}" for k in names] + [f"{rm[mode]['GRU'][st]:.4f}"])
    labels = ["Miller-RHG", "search's best"] + [f"{SHARE[v]} floor" for v in VARIANTS if chosen[v] in tested]
    b4 = ("![curves](../fig_r4_curves.svg)\n\n" + table(["mode", "statistic", *labels, "GRU"], rows4)
          + "\n\n**Figure 2.** The per-bin RMSE of the mouse-average curves (study 15's view, its headline sets and "
            "common subjects, as in r3) for each program and the GRU; lower is closer to the mice.")

    # ---------- figures
    fs.apply_paper_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(183, 72), layout="constrained")
    for v in VARIANTS:
        ok = sorted((p for p in runs[v]["programs"] if p["generation"] > 0 and p["correct"] and p["effort"] is not None),
                    key=lambda p: p["effort"])
        a.scatter([p["effort"] for p in ok], [p["validation_x100"] for p in ok], s=3, lw=0, color=COL[v], alpha=0.35)
        front, best = [], -np.inf
        for p in sorted(ok, key=lambda p: p["effort"]):
            if p["validation_x100"] > best:
                best = p["validation_x100"]
                front.append((p["effort"], p["validation_x100"]))
        a.step(*zip(*front), where="post", color=COL[v], lw=0.8, label=f"{SHARE[v]} floor")
        a.axhline(floors["runs"][v]["floor"], color=COL[v], ls="--", lw=0.6)
    va = floors["validation_anchors_x100"]
    a.scatter([best_eff], [va["high"]], marker="*", s=40, color="k", zorder=5)
    a.scatter([ref_eff], [va["low"]], marker="s", s=14, color="k", zorder=5)
    a.annotate("search's best", (best_eff, va["high"]), xytext=(-3, 3), textcoords="offset points", ha="right", fontsize=5)
    a.annotate("Miller-RHG", (ref_eff, va["low"]), xytext=(3, 2), textcoords="offset points", fontsize=5)
    a.set(xlabel="Halstead effort (lower is simpler)", ylabel="Validation score (×100)", ylim=(va["low"] - 0.02, va["high"] + 0.015))
    a.legend(loc="lower right")
    pts = [(REFERENCE, "Miller-RHG", "k", "s"), (SEARCH_BEST, "search's best", "k", "*")] + \
          [(chosen[v], f"{SHARE[v]} floor", COL[v], "o") for v in VARIANTS if chosen[v] in tested]
    for k, lab, c, m in pts:
        t = ts["programs"][k]
        b.errorbar(efforts[k], t["above_hb_x100"], yerr=t["above_hb_paired_se_x100"], fmt=m, ms=5 if m != "*" else 8,
                   color=c, elinewidth=0.6, capsize=0)
        b.annotate(f"{lab}\n{t['gap_closed']:.2f} of gap", (efforts[k], t["above_hb_x100"]), xytext=(4, -2),
                   textcoords="offset points", fontsize=5, va="top")
    b.axhline(ts["gru_x100"] - ts["hb_ctt_x100"], color=r3.GRU_COLOR, ls="--", lw=0.8)
    b.annotate("GRU", (0, ts["gru_x100"] - ts["hb_ctt_x100"]), xytext=(2, 2), textcoords="offset points", fontsize=5,
               color=r3.GRU_COLOR)
    b.axhline(0, color="0.6", lw=0.5)
    b.annotate("HB-CTT", (0, 0), xytext=(2, 2), textcoords="offset points", fontsize=5, color="0.4")
    b.set(xlabel="Halstead effort (lower is simpler)", ylabel="Test likelihood above HB-CTT (×100)",
          xlim=(0, max(efforts[k] for k, *_ in pts) * 1.15))
    fs.panel_label(a, "a"), fs.panel_label(b, "b")
    fs.save_figure(fig, FIGS["frontier"])

    fig, axes = plt.subplots(1, 2, figsize=fs.mm(183, 64), layout="constrained", width_ratios=[4, 5])
    marks = [("GRU", "GRU", r3.GRU_COLOR, "D")] + [(k, lab, c, m) for k, lab, c, m in pts]
    for ax, mode in zip(axes, ("replay", "rollout")):
        stats = list(rm[mode]["GRU"])
        for i, st in enumerate(stats):
            for k, lab, c, m in marks:
                ax.scatter(rm[mode][k][st], i, marker=m, s=18 if m != "*" else 34, color=c, lw=0,
                           label=lab if i == 0 else None, zorder=3)
        ax.set_yticks(range(len(stats)), stats)
        ax.invert_yaxis()
        ax.set_xlim(0, None)
        ax.set_xlabel("Per-bin RMSE of the mouse-average curve")
        ax.set_title(mode, fontsize=6)
    axes[1].legend(loc="upper right", fontsize=5, handletextpad=0.2)
    fs.panel_label(axes[0], "a"), fs.panel_label(axes[1], "b")
    fs.save_figure(fig, FIGS["curves"])

    # ---------- result 5: all 149 test subjects, for the atlas leaderboard
    label = {REFERENCE: "Miller-RHG (2-level EM)", SEARCH_BEST: "search's best (`satiety_engagement`)",
             **{chosen[v]: f"{SHARE[v]} floor" for v in VARIANTS if chosen[v]}}
    rows5 = [[label[k], f"{s149[k]['two_level']:.4f}", f"{s149[k]['with_fallback']:.4f}", f"{s149[k]['per_subject']:.4f}",
              str(s149[k]["n_fallback"])] for k in names]
    b5 = ("Pooled likelihood (geometric-mean probability per choice) on all 149 test subjects, the atlas "
          "leaderboard's metric. The leaderboard reads the raw 2-level EM score, which is study 12 r3's convention "
          "for the literature models. The other columns are the score with the per-mouse fallback (MEvolve ADR 0004; "
          "this study's headline on the 146 shared subjects uses it) and the per-subject fit.\n\n"
          + table(["program", "2-level EM (raw)", "with per-mouse fallback", "per subject", "mice on fallback"], rows5)
          + "".join(f"\n\n### {label[k]}\n\n{c['summary']} Code: [`{c['program']}`](../{c['program'][len('analysis/'):]})."
                    f"\n\n{c['equations']}" for k, c in CARDS.items() if k in s149))

    write_blocks({1: b1, 2: b2, 3: b3, 4: b4, 5: b5})
    inputs = [*RUNS.values(), FLOORS, CHOICE, CURVES, r3.HB, r3.GRU, *r3.S15.values(),
              *(HERE / "test" / "test" / f"{k}.json" for k in names),
              *(p for k in tested if (p := HERE / "test" / "gru_test" / f"{k}.json").exists()),
              *(STUDY / choice[v]["chosen_file"] for v in VARIANTS if chosen[v]),
              *(STUDY / c["program"] for k, c in CARDS.items() if k in s149 and k not in chosen.values())]
    write_artifact(HERE / "r4_simplification.json", {
        "_meta": build_meta("analysis/r4_simplification.py", [f"{v}@20261108-230437" for v in VARIANTS],
                            study_root=STUDY, code=[__file__, HERE / "r3_search_best.py"], inputs=inputs),
        "efforts": efforts, "chosen": chosen, "test": ts, "rmse": rm, "test_149": s149},
        outputs=[*FIGS.values(), *(Path(f"{REPORT}#result-{k}") for k in range(1, 6))])
    print(json.dumps({"chosen": chosen, "test": {k: {kk: v[kk] for kk in ("pooled_x100", "gap_closed", "gap_closed_ci95")}
                                                for k, v in ts["programs"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
