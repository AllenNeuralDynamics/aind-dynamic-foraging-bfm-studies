"""Producer for r1: gate 4, program recovery on simulated mice.

Reads only committed files: the gate 4 freezes in ``analysis/gate4/`` (``runs.json``,
``first_half.json``, ``example_session.json``, ``novelty.json``, ``archives.json``) and the
pre-check results in ``variants/gate4-recovery/``. Writes five paper-style figures, six result
blocks of ``reports/r1-gate4-program-recovery.md`` and ``r1_gate4_program_recovery.json``.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE.parents[2] / "studies" / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

G = HERE / "gate4"
RUNS, FIRST, EXAMPLE, NOVELTY, ARCHIVES = (G / f for f in ("runs.json", "first_half.json", "example_session.json",
                                                           "novelty.json", "archives.json"))
PRECHECK = {s: STUDY / "variants" / "gate4-recovery" / f"precheck_two_timescale_ck_{s}.json" for s in "abc"}
REPORT = HERE / "reports" / "r1-gate4-program-recovery.md"
FIGS = {k: HERE / f"fig_r1_{k}.svg" for k in ("first_half", "example_session", "precheck", "searches", "novelty")}
WANDB_GROUPS = ["gate4-recovery@20261104-132600", "gate4-noseed@20261104-141503"]
TOLERANCE = 0.2                    # the pass tolerance on the x100 scale (PLAN: 0.002 in pooled likelihood)
THRESHOLDS = (0.99, 0.995, 0.999)  # 0.99 is ShinkaEvolve's default code_embed_sim_threshold
BUDGET_USD = 450                   # PLAN.md: the `seeded-islands` search
ARMS = {"seeded": "seeded (4 islands)", "noseed": "no seeds (template only)"}
COLOR = {"seeded": fs.OKABE_ITO["black"], "noseed": fs.OKABE_ITO["orange"]}

# The mechanism of every program that set a new best score, read by hand from its code in
# runs.json on 2026-11-04 (the freeze keeps the code of exactly these programs):
# (arm, proposal, name) -> (slow reward trace, choice kernel, what it is).
MECHANISM = {
    ("seeded", 0, "initial_program"): (False, False, "template: side bias and last choice"),
    ("seeded", 0, "q_learning"): (False, False, "seed: one reward value per side"),
    ("seeded", 0, "choice_kernel"): (False, True, "seed: choice kernel, no reward learning"),
    ("seeded", 0, "two_timescale"): (True, False, "seed: fast and slow reward values"),
    ("seeded", 1, "q_learning_plus_choice_trace"): (False, True, "one reward value with forgetting, choice trace"),
    ("seeded", 2, "q_learning_plus_choice_trace"): (False, True, "the same, rewritten"),
    ("seeded", 3, "two_timescale_q_choice_trace"): (True, True, "fast and slow reward values, choice trace: the generator"),
    ("noseed", 0, "initial_program"): (False, False, "template: side bias and last choice"),
    ("noseed", 1, "add_q_learning_with_choice_trace"): (False, True, "one reward value with forgetting, choice trace"),
    ("noseed", 7, "two_timescale_choice_trace"): (False, True, "adds a slow choice trace"),
    ("noseed", 12, "forgetting_unchosen_q"): (False, True, "unchosen value decays"),
    ("noseed", 23, "fast_slow_trace_value_drift"): (False, True, "crossover: two choice traces, value drift"),
    ("noseed", 26, "reward_gated_choice_kernel"): (False, True, "adds a win-stay kernel"),
    ("noseed", 71, "dual_rate_values_with_choice_traces"): (True, True, "fast and slow reward values, two choice traces"),
    ("noseed", 75, "fast_slow_value_learners"): (True, True, "the same, refined"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def best_steps(programs: list[dict]) -> list[dict]:
    """The starting programs and every program that set a new best score, in proposal order."""
    out, best = [], float("-inf")
    for p in sorted(programs, key=lambda p: p["timestamp"]):
        if p["generation"] == 0 or (p["score"] is not None and p["score"] > best):
            out.append(p)
        if p["score"] is not None:
            best = max(best, p["score"])
    return out


def name(p: dict) -> str:
    return p["mevolve_seed"] or p["patch_name"] or "initial_program"


def table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f3(x) -> str:
    return "–" if x is None else fs.num(x, ".3f")


def first_half(first: dict) -> tuple[str, dict]:
    models = first["models"]
    ncol = max(len(m["bounds"]) for m in models.values())
    fig, axes = plt.subplots(len(models), ncol, figsize=fs.mm(183, 36 * len(models)), squeeze=False)
    rows, out = [], {}
    for i, (model, m) in enumerate(models.items()):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(m["bounds"]):
                ax.axis("off")
                continue
            par = list(m["bounds"])[j]
            t = np.array([s["true_typical"][par] for s in m["subjects"]])
            r = np.array([s["recovered_typical"][par] for s in m["subjects"]])
            lo, hi = min(t.min(), r.min()), max(t.max(), r.max())
            pad = 0.05 * ((hi - lo) or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.7", lw=0.5)
            ax.scatter(t, r, s=4, color="k", lw=0)
            ax.set_xlim(lo - pad, hi + pad), ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(par, fontsize=6), ax.set_xlabel("true")
            if j == 0:
                ax.set_ylabel(f"{model}\nrecovered")
            c = m["params"][par]
            ok = all(c[k] for k in ("mu_ok", "bound_ok", "sigma_ok") if c.get(k) is not None)
            ax.text(0.03, 0.97, "pass" if ok else "fail", transform=ax.transAxes, va="top", fontsize=5,
                    color=fs.OKABE_ITO["green"] if ok else fs.OKABE_ITO["vermillion"])
            rows.append([model, par, c["n_off_bound"], f3(c.get("mu_error_median")),
                         f3(c.get("per_subject_error_median")),
                         "–" if c.get("sigma_ratio_median") is None else fs.num(c["sigma_ratio_median"], ".2f"),
                         "pass" if ok else "fail"])
        out[model] = {"ok": m["ok"], "score": m["score"]}
    fig.tight_layout()
    fs.save_figure(fig, FIGS["first_half"])
    plt.close(fig)
    text = (f"![first half](../fig_r1_first_half.svg)\n\nTrue against recovered typical values of 24 simulated "
            f"subjects per model, 2-level EM (MEvolve {first['mevolve_git_sha'][:7]}, at most {first['max_em']} "
            "EM iterations). Errors are medians on the probit scale. A parameter passes when its 2-level error is "
            "at most the per-subject fit's on the same data (criterion amended 2026-11-03), a parameter on a bound "
            "is recovered within 0.05 of its range, and the median spread ratio is within [1/2, 2].\n\n"
            + table(["model", "parameter", "subjects off bound", "2-level error", "per-subject error",
                     "spread ratio", "criteria"], rows))
    return text, out


def example_session(example: dict) -> str:
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session

    fig = plt.figure(figsize=fs.mm(183, 70))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.15)
    _, axes = plot_foraging_session(np.asarray(example["choices"], float), np.asarray(example["rewards"], float),
                                    np.asarray(example["p_reward"]), fitted_data=np.asarray(example["p_right"]),
                                    ax=fig.add_subplot(grid[0]))
    for a in axes:
        a.set_xlabel(""), a.tick_params(labelbottom=False)
        if a.get_legend():
            a.get_legend().remove()
    ax = fig.add_subplot(grid[1])
    lat = np.asarray(example["latents"])
    styles = {"fast_left": ("r", "-"), "fast_right": ("b", "-"), "slow_left": ("r", "--"), "slow_right": ("b", "--"),
              "kernel": ("0.4", "-")}
    for k, nm in enumerate(example["latent_names"]):
        ax.plot(lat[:, k], color=styles[nm][0], ls=styles[nm][1], lw=0.6, label=nm.replace("_", " "))
    ax.set_xlim(axes[0].get_xlim()), ax.set_xlabel("trial"), ax.set_ylabel("latent variable")
    ax.legend(frameon=False, ncol=5, fontsize=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    p0, p1 = axes[0].get_position(), ax.get_position()
    ax.set_position([p0.x0, p1.y0, p0.width, p1.height])
    fs.save_figure(fig, FIGS["example_session"])
    plt.close(fig)
    params = ", ".join(f"{k} {v:g}" for k, v in example["params"].items())
    return (f"![example session](../fig_r1_example_session.svg)\n\nOne session from the generator at setting a "
            f"({params}), {example['n_trials']} trials, seed {example['seed']}: choices with the generator's "
            "p(right), and its latent variables below (right blue, left red; slow traces dashed, kernel grey).")


def precheck(prechecks: dict) -> tuple[str, dict]:
    order = ["one_timescale_ck", "two_timescale", "q_learning", "choice_kernel", "template"]
    labels = ["reference\n(no slow trace)", "seed\ntwo_timescale", "seed\nq_learning", "seed\nchoice_kernel", "template"]
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=len(order)), 50))
    x = np.arange(len(order))
    for k, (s, c) in enumerate(zip("abc", ["0.15", "0.5", "0.8"])):
        ax.bar(x + (k - 1) * 0.27, [prechecks[s]["margins"][o] for o in order], 0.27, color=c,
               label=f"setting {s}" + (" (used)" if s == "a" else ""))
    ax.axhline(2 * TOLERANCE, color="k", ls="--", lw=0.6)
    ax.set_xticks(x, labels, fontsize=5), ax.set_ylabel("target − program (×100)")
    ax.legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fs.save_figure(fig, FIGS["precheck"])
    plt.close(fig)
    rows = [["target `two_timescale_ck` (its score)", *[f3(prechecks[s]["target_score"]) for s in "abc"]]]
    rows += [[f"`{o}`", *[f3(prechecks[s]["margins"][o]) for s in "abc"]] for o in order]
    text = (f"![pre-check](../fig_r1_precheck.svg)\n\nPre-check on the simulated validation mice: each program's "
            f"margin below the target. The dashed line is the stop rule ({2 * TOLERANCE}). Every program is outside "
            "it in every setting, so the pre-registered order picks setting a.\n\n"
            + table(["program", "setting a", "setting b", "setting c"], rows))
    return text, {s: {"target_score": prechecks[s]["target_score"], "ok": prechecks[s]["ok"]} for s in "abc"}


def searches(runs: dict, target: float) -> tuple[str, dict]:
    pass_mark = target - TOLERANCE
    fig, (a0, a1) = plt.subplots(1, 2, figsize=fs.mm(150, 58))
    rows, out = [], {}
    for arm, run in runs.items():
        P = sorted((p for p in run["programs"] if p["generation"] > 0), key=lambda p: p["generation"])
        g = np.array([p["generation"] for p in P])
        s = np.array([p["score"] if p["score"] is not None else np.nan for p in P])
        start = max(p["score"] for p in run["programs"] if p["generation"] == 0 and p["score"] is not None)
        best = np.fmax.accumulate(np.concatenate([[start], s]))
        for ax in (a0, a1):
            ax.scatter(g, s, s=3, color=COLOR[arm], alpha=0.35, lw=0)
            ax.step(np.concatenate([[0], g]), best, where="post", color=COLOR[arm], lw=1, label=ARMS[arm])
        first_pass = int(g[np.nonzero(s >= pass_mark)[0][0]]) if np.any(s >= pass_mark) else None
        winner = next(p for p in run["programs"] if p["id"] == run["heldout"]["winner_id"])
        h, cfg = run["heldout"]["scores"], run["run_config"]
        o = out[arm] = {
            "best": winner["score"], "winner_proposal": winner["generation"], "first_pass": first_pass,
            "passed": winner["score"] >= pass_mark, "heldout_winner": h["winner"]["heldout_score"],
            "heldout_target": h["target"]["heldout_score"], "calls": run["summary"]["calls"],
            "cost_usd": run["summary"]["cost_usd"], "hours": run["summary"]["seconds"] / 3600,
            "n_proposals": cfg["n_proposals"], "failed_edits": len(run["failed_attempts"]),
            "eval_seconds_median": float(np.median([p["evaluation_seconds"] for p in P if p["evaluation_seconds"]])),
            "islands": 1 + len(cfg["seeds"])}
        rows.append([ARMS[arm], f3(o["best"]), o["winner_proposal"], o["first_pass"], "pass" if o["passed"] else "fail",
                     f3(o["heldout_winner"]), f3(o["heldout_target"]), f"${o['cost_usd']:.2f}", o["calls"],
                     f"{o['hours']:.2f}", o["failed_edits"]])
    for ax in (a0, a1):
        ax.axhline(target, color="0.3", lw=0.6, ls="--"), ax.axhline(pass_mark, color="0.3", lw=0.6, ls=":")
        ax.set_xlabel("proposal")
    a0.set_ylabel("validation score (×100)"), a0.set_ylim(64, 74), a0.legend(frameon=False, loc="lower right")
    a1.set_ylim(72.0, 73.1), a1.set_title("zoom: target dashed, pass mark dotted", fontsize=6)
    fs.panel_label(a0, "a"), fs.panel_label(a1, "b")
    fig.tight_layout()
    fs.save_figure(fig, FIGS["searches"])
    plt.close(fig)
    text = (f"![searches](../fig_r1_searches.svg)\n\nBest validation score so far against proposals (lines) and "
            f"every proposal (dots). Target {f3(target)}, pass mark {f3(pass_mark)} (one-sided). One run per arm: "
            "no run-to-run variance was measured.\n\n"
            + table(["arm", "best", "at proposal", "first at the pass mark", "gate 4", "held-out winner",
                     "held-out target", "cost", "Claude calls", "hours", "failed edits"], rows))
    return text, out


def mechanisms(runs: dict) -> tuple[str, dict]:
    rows, out = [], {}
    for arm, run in runs.items():
        steps = best_steps(run["programs"])
        missing = [(arm, p["generation"], name(p)) for p in steps if (arm, p["generation"], name(p)) not in MECHANISM]
        assert not missing, f"no hand reading for {missing}: read their code in runs.json and add them"
        first_slow = next(p for p in steps if MECHANISM[(arm, p["generation"], name(p))][0])
        out[arm] = {"first_slow_trace": {"proposal": first_slow["generation"], "program": name(first_slow)},
                    "steps": [{"proposal": p["generation"], "program": name(p), "score": p["score"]} for p in steps]}
        for p in steps:
            slow, kernel, what = MECHANISM[(arm, p["generation"], name(p))]
            rows.append([ARMS[arm], p["generation"], f"`{name(p)}`", f3(p["score"]), p["public"].get("n_params"),
                         "yes" if slow else "", "yes" if kernel else "", what])
    text = ("The starting programs and every program that set a new best score. Mechanisms read by hand from "
            "each program's code (kept in `analysis/gate4/runs.json`). The generator has a slow reward trace and "
            "a choice kernel, with 7 parameters.\n\n"
            + table(["arm", "proposal", "program", "score", "parameters", "slow reward trace", "choice kernel",
                     "what it is"], rows))
    return text, out


def novelty(runs: dict, nov: dict) -> tuple[str, dict]:
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(150, 60), sharey=True)
    rows, out = [], {}
    for ax, (arm, run) in zip(axes, runs.items()):
        by_id = {p["id"]: p for p in run["programs"]}
        steps = {p["id"] for p in best_steps(run["programs"]) if p["generation"] > 0}
        pts = []
        for r in nov["arms"][arm]:
            p = by_id[r["id"]]
            if p["generation"] == 0 or r["max_similarity"] is None:
                continue
            parent = by_id.get(p["parent_id"])
            gain = (p["score"] - parent["score"]) if p["score"] is not None and parent and parent["score"] is not None else None
            pts.append((r["max_similarity"], gain, p["id"] in steps, p["generation"]))
        x = np.array([q[0] for q in pts])
        y = np.array([q[1] if q[1] is not None else -1.9 for q in pts])
        ax.scatter(x, y, s=5, color="0.4", lw=0)
        nb = [q for q in pts if q[2]]
        ax.scatter([q[0] for q in nb], [q[1] if q[1] is not None else -1.9 for q in nb], s=14, facecolor="none",
                   edgecolor=fs.OKABE_ITO["vermillion"], lw=0.8, label="new best so far")
        for q in nb:
            ax.annotate(str(q[3]), (q[0], q[1] if q[1] is not None else -1.9), fontsize=5, xytext=(2, 2),
                        textcoords="offset points", color=fs.OKABE_ITO["vermillion"])
        ax.axvline(THRESHOLDS[0], color="k", lw=0.6, ls="--"), ax.axvline(THRESHOLDS[-1], color="k", lw=0.6, ls=":")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlim(0.93, 1.0005), ax.set_ylim(-2, 7.5), ax.set_title(ARMS[arm], fontsize=6)
        ax.set_xlabel("highest similarity to its island (cosine)")
        groups = {}
        for r in nov["arms"][arm]:
            groups.setdefault(r["repeat_key"], []).append(by_id[r["id"]])
        repeats = [g for g in groups.values() if len(g) > 1]
        later = [q for g in repeats for q in sorted(g, key=lambda q: q["timestamp"])[1:] if q["score"] is not None]
        out[arm] = {"proposals": len(pts), "median_similarity": float(np.median(x)), "repeat_groups": len(repeats),
                    "repeats_scored": len(later), "repeat_seconds": float(sum(q["evaluation_seconds"] or 0 for q in later)),
                    "drops": {str(th): {"n": sum(q[0] > th for q in pts),
                                        "new_bests": sorted(q[3] for q in pts if q[0] > th and q[2])} for th in THRESHOLDS}}
    axes[0].set_ylabel("score gain over parent (×100)"), axes[0].legend(frameon=False, loc="upper left", fontsize=5)
    fs.panel_label(axes[0], "a"), fs.panel_label(axes[1], "b")
    fig.tight_layout()
    fs.save_figure(fig, FIGS["novelty"])
    plt.close(fig)
    for th in THRESHOLDS:
        rows.append([f"> {th}" + (" (ShinkaEvolve's default)" if th == 0.99 else ""),
                     *[f"{out[a]['drops'][str(th)]['n']} of {out[a]['proposals']} "
                       f"({out[a]['drops'][str(th)]['n'] / out[a]['proposals']:.0%}); new bests lost: "
                       f"{', '.join(map(str, out[a]['drops'][str(th)]['new_bests'])) or 'none'}" for a in runs]])
    rows.append(["exact repeats (MEvolve's repeat key)",
                 *[f"{out[a]['repeats_scored']} scored repeats in {out[a]['repeat_groups']} groups, "
                   f"{out[a]['repeat_seconds'] / 60:.0f} min of fitting" for a in runs]])
    text = (f"![novelty](../fig_r1_novelty.svg)\n\nPost hoc. x: each proposal's highest code-embedding similarity "
            f"to the programs on its island just before it ({nov['embedding']['model']}); y: its gain over its "
            "parent (failed or unscored at the bottom). ShinkaEvolve's novelty filter drops everything right of its "
            "threshold (dashed 0.99, dotted 0.999). Median similarity: "
            + ", ".join(f"{ARMS[a]} {out[a]['median_similarity']:.3f}" for a in runs) + ".\n\n"
            + table(["filter", *[ARMS[a] for a in runs]], rows))
    return text, out


def compute(search: dict) -> tuple[str, dict]:
    rows, out = [], {}
    for arm, s in search.items():
        per, node_h = s["cost_usd"] / s["n_proposals"], s["hours"] / s["n_proposals"]
        n = BUDGET_USD / per
        out[arm] = {"usd_per_proposal": per, "node_hours_per_proposal": node_h, "proposals_for_budget": n,
                    "node_days_for_budget": n * node_h / 24}
        rows.append([ARMS[arm], f"${per:.4f}", f"{node_h * 60:.1f} min", f"{s['eval_seconds_median'] / 60:.1f} min",
                     f"{n:,.0f}", f"{n * node_h / 24:.0f}"])
    text = (f"Measured per proposal (one 56-core job per run, two evaluations at a time, each on 24 + 24 simulated "
            f"mice), and extrapolated to the PLAN's ${BUDGET_USD} search at the same rates.\n\n"
            + table(["arm", "Claude cost", "node time", "median evaluation", f"proposals for ${BUDGET_USD}",
                     "node-days"], rows))
    return text, out


def write_blocks(blocks: dict[int, str]) -> None:
    text = REPORT.read_text()
    for n, body in blocks.items():
        start, end = f"<!-- BEGIN result-{n} -->", f"<!-- END result-{n} -->"
        assert text.count(start) == 1 and text.count(end) == 1, f"result-{n} markers"
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + body.strip() + "\n" + text[j:]
    REPORT.write_text(text)


def main() -> None:
    fs.apply_paper_style()
    runs = load(RUNS)["runs"]
    prechecks = {s: load(p) for s, p in PRECHECK.items()}
    target = prechecks["a"]["target_score"]
    b1, r1 = first_half(load(FIRST))
    b2 = example_session(load(EXAMPLE))
    b3, r3 = precheck(prechecks)
    b4, r4 = searches(runs, target)
    b5, r5 = mechanisms(runs)
    b6, r6 = novelty(runs, load(NOVELTY))
    b7, r7 = compute(r4)
    write_blocks({1: b1, 2: b2 + "\n\n" + b3, 3: b4, 4: b5, 5: b6, 6: b7})
    meta = build_meta("analysis/r1_gate4_program_recovery.py", WANDB_GROUPS, study_root=STUDY, code=[__file__],
                      inputs=[RUNS, FIRST, EXAMPLE, NOVELTY, ARCHIVES, *PRECHECK.values()])
    write_artifact(HERE / "r1_gate4_program_recovery.json", {
        "_meta": meta, "target": target, "pass_mark": target - TOLERANCE, "first_half": r1, "precheck": r3,
        "searches": r4, "mechanisms": r5, "novelty": r6, "compute": r7,
        "archives": {a: f"{v['artifact']} {v['digest']}" for a, v in load(ARCHIVES)["archives"].items()}},
        outputs=[*FIGS.values(), *(Path(f"{REPORT}#result-{n}") for n in range(1, 7))])


if __name__ == "__main__":
    main()
