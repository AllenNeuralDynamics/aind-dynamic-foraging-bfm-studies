#!/usr/bin/env python
"""r2: the first production search (seeded-islands-2), from its committed freezes. Offline.

Reads ``seeded_islands_2/run.json`` (every program, Claude call and the stopping rule) and
``seeded_islands_2/monitor.json`` (programs rescored on the validation and the 100 monitor subjects, #128).
Writes ``r2_search_curve.json``, three figures and the report's result blocks:

1. the search curve: every scored proposal's validation score by island, the best so far, the stopping
   rule's rises, and the best so far per island against Claude dollars;
2. the drift: score quantiles and parent scores per block of proposals, the share of programs carrying
   each mechanism (tagged by parameter names), and the change from the parent when a child adds one;
3. the monitor rescoring: gains over the best seed on validation against monitor subjects, the final
   archive's top 10 against the early top 10, and validation against monitor rank across the archive.

Monitor scores apply MEvolve's per-subject fallback (ADR 0004, amended 2026-11-08; ``program_score`` at
MEvolve 146153c): a subject whose 2-level log-likelihood is non-finite or more than ``FALLBACK_TOL`` nats
per trial below its per-subject one is scored with its per-subject fit.
"""
from __future__ import annotations

import json
import re
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

WANDB_GROUPS = ["seeded-islands@20261106-114959"]
RUN, MONITOR = HERE / "seeded_islands_2" / "run.json", HERE / "seeded_islands_2" / "monitor.json"
REPORT = HERE / "reports" / "r2-search-curve.md"
FIGS = {k: HERE / f"fig_r2_{k}.svg" for k in ("curve", "drift", "monitor")}
FALLBACK_TOL = 0.005          # MEvolve evaluate.FALLBACK_TOL at 146153c
BLOCK = 250                   # proposals per block in the drift figure
N_BOOT = 2000
OI = fs.OKABE_ITO
ISLAND_COLORS = [OI["black"], OI["blue"], OI["vermillion"], OI["green"], OI["purple"]]
HABIT = r"(habit|trace|kernel|perseverat)"
MECH = {                      # parameter-name patterns; "two habits" needs a fast and a slow one
    "counterfactual update": r"counterfactual",
    "unchosen forgetting": r"forget",
    "two habits": None,
    "bait expectation": r"bait",
    "reward rate": r"reward_rate",
    "experience": r"experience",
    "engagement or lapse": r"engagement|lapse",
    "satiety": r"satiety",
    "win-stay, lose-shift": r"win_stay|wsls|loss_run|stay_weight",
}
CORE = ("counterfactual update", "unchosen forgetting", "two habits")
ADD_ONS = ("bait expectation", "reward rate", "experience", "engagement or lapse", "satiety",
           "win-stay, lose-shift")


def tags(names: list[str]) -> set[str]:
    out = {k for k, pat in MECH.items() if pat and any(re.search(pat, n) for n in names)}
    if any(re.search(r"fast", n) and re.search(HABIT, n) for n in names) and \
            any(re.search(r"slow", n) and re.search(HABIT, n) for n in names):
        out.add("two habits")
    return out


def table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


# ---------------------------------------------------------------- 1. search curve
def curve(run: dict) -> tuple[str, dict]:
    progs = run["programs"]
    seeds = [p for p in progs if p["generation"] == 0]
    islands = {p["island"]: (p["seed"] or "template") for p in seeds}
    top_seed = max((p for p in seeds if p["seed"]), key=lambda p: p["score"])
    best_seed, best_seed_name = top_seed["score"], top_seed["seed"]
    props = sorted((p for p in progs if p["generation"] > 0), key=lambda p: p["generation"])
    scored = [p for p in props if p["correct"]]
    best = max(scored, key=lambda p: p["score"])
    usd = np.cumsum([p["cost_usd"] for p in props])
    stop = run["stop_rule"]

    fs.apply_paper_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(2 * 45 + 2 * fs.AXIS_MM + 30, 66), layout="constrained",
                               width_ratios=[1.6, 1])
    lo = best_seed - 0.35
    for i, name in sorted(islands.items()):
        g = [p for p in scored if p["island"] == i]
        a.scatter([p["generation"] for p in g], np.clip([p["score"] for p in g], lo, None), s=1.5, lw=0,
                  color=ISLAND_COLORS[i], alpha=0.5, label=f"island {i}: {name}")
    bsf, cur = [], best_seed
    for p in props:
        cur = max(cur, p["score"]) if p["correct"] else cur
        bsf.append(cur)
    gens = [p["generation"] for p in props]
    a.step(gens, bsf, where="post", color="0.2", lw=0.8, label="best so far")
    a.axhline(best_seed, color="0.5", ls=":", lw=0.6, label=f"best seed ({best_seed_name})")
    for r in stop["rises"]:
        a.plot(r["proposal"], r["score"], marker="v", ms=3, color=OI["orange"], zorder=4)
    a.axvline(stop["met_at_proposal"], color=OI["orange"], lw=0.6, ls="--")
    a.annotate("stopping rule met", (stop["met_at_proposal"], lo), xytext=(-2, 2), textcoords="offset points",
               ha="right", va="bottom", color=OI["orange"])
    a.set(xlabel="Proposal", ylabel="Validation score (×100)", ylim=(lo, best["score"] + 0.04),
          xlim=(0, gens[-1] + 20))
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=4, markerscale=3, handletextpad=0.2,
               columnspacing=0.8)
    per_island = {}
    for i, name in sorted(islands.items()):
        seed_score = next(p["score"] for p in seeds if p["island"] == i)
        cur, xs, ys = seed_score, [0.0], [seed_score]
        for p, d in zip(props, usd):
            if p["island"] == i and p["correct"] and p["score"] > cur:
                cur = p["score"]
                xs.append(d), ys.append(cur)
        xs.append(usd[-1]), ys.append(cur)
        b.step(xs, ys, where="post", color=ISLAND_COLORS[i], lw=0.8)
        per_island[name] = {"island": i, "seed": seed_score, "best": cur,
                            "proposals": sum(p["island"] == i for p in props)}
    b.axhline(best_seed, color="0.5", ls=":", lw=0.6)
    b.set(xlabel="Claude cost (USD, cumulative)", ylabel="Best validation score so far (×100)",
          ylim=(best_seed - 0.1, best["score"] + 0.02))
    fs.panel_label(a, "a"), fs.panel_label(b, "b")
    fs.save_figure(fig, FIGS["curve"])

    hours = run["summary"]["seconds"] / 3600
    res = {"proposals": len(props), "scored": len(scored), "failed_checks": len(props) - len(scored),
           "cost_usd": run["summary"]["cost_usd"], "calls": run["summary"]["calls"], "hours": hours,
           "evaluation_jobs": run["slurm"]["evaluation_jobs"], "best_seed": best_seed,
           "best_seed_name": best_seed_name,
           "best": {k: best[k] for k in ("generation", "id", "island", "name", "score", "n_params", "n_latent")},
           "gain_over_best_seed": best["score"] - best_seed,
           "stop_rule": {k: stop[k] for k in ("met_at_proposal", "anchor", "se", "window", "rises")},
           "islands": per_island, "above_best_seed": sum(p["score"] > best_seed for p in scored),
           "two_level_failed": sum(bool(p["two_level_failed"]) for p in scored),
           "repeats": sum(bool(p["repeat_of"]) for p in scored)}
    rises = " → ".join(f"{r['score']:.3f} (proposal {r['proposal']:,})" for r in stop["rises"])
    rows = [[f"{v['island']}: {name}", f"{v['proposals']:,}", f"{v['seed']:.3f}", f"{v['best']:.3f}",
             fs.num(v["best"] - v["seed"], "+.3f")] for name, v in per_island.items()]
    ev = res["evaluation_jobs"]
    text = (f"![search curve](../fig_r2_curve.svg)\n\n"
            f"**{res['proposals']:,} proposals** ({res['scored']:,} scored, {res['failed_checks']} rejected by the "
            f"static checks) cost **${res['cost_usd']:.2f}** of Claude in {res['calls']:,} calls over {hours:.1f} h, "
            f"with {ev['n']:,} evaluation jobs ({ev['cpu_hours']:,.0f} CPU-hours). The best program, "
            f"`{best['name']}` (proposal {best['generation']:,}, island {best['island']}), scores "
            f"**{best['score']:.3f}**, {fs.num(res['gain_over_best_seed'], '+.3f')} over the best seed "
            f"({best_seed_name}, {best_seed:.3f}). {res['above_best_seed']:,} of {res['scored']:,} scored proposals "
            f"beat the best seed. {res['repeats']} scored proposals repeated an earlier program and reused its score "
            f"(the flat rows in a).\n\n"
            f"The stopping rule (stop when the best has not risen by {stop['se']} in {stop['window']:,} proposals) "
            f"saw the anchor rise {len(stop['rises'])} times, {rises}, and was met at proposal "
            f"{stop['met_at_proposal']:,}. The one later best (proposal {best['generation']:,}) is "
            f"{fs.num(best['score'] - stop['anchor'], '+.4f')} above the anchor, less than the rule's step.\n\n"
            + table(["island", "proposals", "seed", "best", "gain"], rows)
            + "\n\n**Figure 1.** (a) Validation score (48 subjects, 2-level EM, `max_em` 20) of every scored "
              "proposal, by island; scores below the axis are drawn at its bottom. Line: best so far; triangles: "
              "the stopping rule's rises; dashed: the rule met. (b) Best so far per island against cumulative "
              "Claude cost.")
    return text, res


# ---------------------------------------------------------------- 2. drift and mechanisms
def drift(run: dict) -> tuple[str, dict]:
    progs = run["programs"]
    by_id = {p["id"]: p for p in progs}
    tag = {p["id"]: tags(p["param_names"]) for p in progs}
    props = sorted((p for p in progs if p["generation"] > 0 and p["correct"]), key=lambda p: p["generation"])
    blocks = []
    for i in range(0, props[-1]["generation"], BLOCK):
        b = [p for p in props if i < p["generation"] <= i + BLOCK]
        if len(b) < 50:
            continue
        s = np.array([p["score"] for p in b])
        par = [by_id[p["parent_id"]]["score"] for p in b
               if p["parent_id"] in by_id and by_id[p["parent_id"]]["correct"]]
        blocks.append({"from": i + 1, "to": i + BLOCK, "n": len(b), "x": float(np.mean([p["generation"] for p in b])),
                       "quantiles": np.percentile(s, [10, 25, 50, 75, 90]).tolist(),
                       "median_parent": float(np.median(par)),
                       "mean_params": float(np.mean([p["n_params"] for p in b])),
                       "share": {k: float(np.mean([k in tag[p["id"]] for p in b])) for k in MECH}})
    gains = {k: [] for k in ADD_ONS}
    same = []
    for p in props:
        q = by_id.get(p["parent_id"])
        if not q or not q["correct"]:
            continue
        for k in (tag[p["id"]] - tag[q["id"]]) & set(ADD_ONS):
            gains[k].append(p["score"] - q["score"])
        if tag[p["id"]] == tag[q["id"]]:
            same.append(p["score"] - q["score"])

    fs.apply_paper_style()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=fs.mm(183, 64), layout="constrained", width_ratios=[1, 1, 1.1])
    x = [bl["x"] for bl in blocks]
    Q = np.array([bl["quantiles"] for bl in blocks])
    a.fill_between(x, Q[:, 0], Q[:, 4], color=OI["blue"], alpha=0.15, lw=0, label="10th–90th percentile")
    a.fill_between(x, Q[:, 1], Q[:, 3], color=OI["blue"], alpha=0.3, lw=0, label="25th–75th percentile")
    a.plot(x, Q[:, 2], color=OI["blue"], marker="o", ms=2, label="median child")
    a.plot(x, [bl["median_parent"] for bl in blocks], color=OI["vermillion"], marker="o", ms=2, label="median parent")
    a.set(xlabel=f"Proposal (blocks of {BLOCK})", ylabel="Validation score (×100)")
    a.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
    cols = [OI["green"], OI["sky"], OI["black"], OI["orange"], OI["vermillion"], OI["purple"], OI["blue"], "0.5",
            OI["yellow"]]
    for k, col in zip(MECH, cols):
        b.plot(x, [bl["share"][k] for bl in blocks], marker="o", ms=1.5, lw=0.7, color=col, label=k)
    b.set(xlabel=f"Proposal (blocks of {BLOCK})", ylabel="Share of scored programs", ylim=(0, 1.02))
    b.legend(loc="center right", fontsize=5, handlelength=1, borderaxespad=0.2, frameon=True, framealpha=0.85,
             edgecolor="none")
    data = [same] + [gains[k] for k in ADD_ONS]
    labels = ["no new mechanism"] + list(ADD_ONS)
    c.boxplot(data, widths=0.6, showfliers=False, medianprops={"color": OI["vermillion"]},
              whiskerprops={"lw": 0.5}, boxprops={"lw": 0.5}, capprops={"lw": 0.5})
    c.axhline(0, color="0.5", ls=":", lw=0.6)
    c.set_xticks(range(1, len(data) + 1), labels, rotation=45, ha="right", rotation_mode="anchor")
    c.set_ylabel("Child − parent (×100)")
    top = c.get_ylim()[1]
    for i, d in enumerate(data, 1):
        c.annotate(f"{len(d)}", (i, top), xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                   fontsize=5)
    for ax, letter in zip((a, b, c), "abc"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, FIGS["drift"])

    q = {k: {"n": len(v), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
             "q75": float(np.percentile(v, 75)), "share_improved": float(np.mean(np.array(v) > 0))}
         for k, v in [("no new mechanism", same)] + [(k, gains[k]) for k in ADD_ONS] if v}
    res = {"block": BLOCK, "blocks": blocks, "add_on_changes": q, "mechanism_patterns": MECH}
    first, last = blocks[0], blocks[-1]
    core_first = ", ".join(f"{100 * first['share'][k]:.0f}%" for k in CORE)
    core_last = ", ".join(f"{100 * last['share'][k]:.0f}%" for k in CORE)
    rows = [[k, f"{100 * first['share'][k]:.1f}%", f"{100 * last['share'][k]:.1f}%"] for k in MECH]
    grows = [[k, str(v["n"]), fs.num(v["median"], "+.3f"),
              f"[{fs.num(v['q25'], '+.3f')}, {fs.num(v['q75'], '+.3f')}]", f"{v['share_improved']:.2f}"]
             for k, v in q.items()]
    text = (f"![drift](../fig_r2_drift.svg)\n\n"
            f"The median scored proposal rose from {first['quantiles'][2]:.3f} (proposals {first['from']}–"
            f"{first['to']}) to {last['quantiles'][2]:.3f} (proposals {last['from']:,}–{last['to']:,}), and the "
            f"median parent from {first['median_parent']:.3f} to {last['median_parent']:.3f}. The rise follows the "
            f"spread of a shared core ({', '.join(CORE)}), whose shares went from {core_first} to {core_last}.\n\n"
            + table(["mechanism (parameter names)", f"share, proposals 1–{BLOCK}", "share, last block"], rows)
            + "\n\nWhen a child adds one add-on mechanism to its parent, its score changes by:\n\n"
            + table(["added", "children", "median change", "IQR", "share improved"], grows)
            + f"\n\n**Figure 2.** (a) Score quantiles of scored proposals and the median score of their parents, "
              f"per block of {BLOCK} proposals. (b) Share of scored programs whose parameter names carry each "
              f"mechanism (patterns in `r2_search_curve.json`). (c) Score change from parent to child when the "
              f"child adds an add-on mechanism, against children that keep the parent's mechanisms; boxes are "
              f"quartiles, whiskers 1.5 IQR, outliers not drawn, numbers are children.")
    return text, res


# ---------------------------------------------------------------- 3. monitor rescoring
def used(rows: list) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-subject log-likelihoods the score uses (MEvolve's ``program_score`` fallback), trials, fallbacks."""
    a = np.array(rows, dtype=float)
    two, one, n = a[:, 0], a[:, 1], a[:, 2]
    fell = ~np.isfinite(two) | (two < one - FALLBACK_TOL * n)
    mixed = np.where(fell, one, two)
    if fell.all() or mixed.sum() < one.sum():
        return one, n, int(fell.sum())
    return mixed, n, int(fell.sum())


def score(ll, n, idx=None):
    idx = slice(None) if idx is None else idx
    return 100 * np.exp(ll[idx].sum() / n[idx].sum())


def rank(v):
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v))
    return r


def monitor(mon: dict) -> tuple[str, dict]:
    progs = {p["tag"]: p for p in mon["programs"]}
    rng = np.random.default_rng(0)
    cohorts = ("validation", "monitor")
    u = {t: {c: used(p[c]) for c in cohorts} for t, p in progs.items()}
    boots = {c: [rng.integers(0, len(mon["subjects"][c]), len(mon["subjects"][c])) for _ in range(N_BOOT)]
             for c in cohorts}
    memo: dict = {}

    def gain(t, c):
        if (t, c) not in memo:
            ll, n, _ = u[t][c]
            ls = u["seed"][c][0]
            memo[t, c] = (score(ll, n) - score(ls, n),
                          np.array([score(ll, n, i) - score(ls, n, i) for i in boots[c]]))
        return memo[t, c]

    first = [t for t in sorted(progs) if "first" in progs[t]["batches"] and t != "seed"]
    rows = []
    for t in first:
        (gv, bv), (gm, bm) = gain(t, "validation"), gain(t, "monitor")
        rows.append({"tag": t, "name": progs[t]["name"], "island": progs[t]["island"],
                     "generation": progs[t]["generation"], "gain_validation": gv, "se_validation": float(bv.std()),
                     "gain_monitor": gm, "se_monitor": float(bm.std()),
                     "fallback": {c: u[t][c][2] for c in cohorts}})
    early = [t for t in first if t.startswith("top")]
    arch = sorted((t for t in progs if "final_archive" in progs[t]["batches"]),
                  key=lambda t: progs[t]["archive_rank"])
    late = arch[:10]
    means = [float(np.mean([gain(t, "monitor")[0] for t in g])) for g in (early, late)]
    eb = np.mean([gain(t, "monitor")[1] for t in early], axis=0)
    lb = np.mean([gain(t, "monitor")[1] for t in late], axis=0)
    d = means[1] - means[0]
    ci = np.percentile(lb - eb, [2.5, 97.5])
    val = np.array([progs[t]["search_validation_x100"] for t in arch])
    mons = np.array([score(*u[t]["monitor"][:2]) for t in arch])
    rho = float(np.corrcoef(rank(val), rank(mons))[0, 1])
    overlap = len(set(np.argsort(-val)[:10]) & set(np.argsort(-mons)[:10]))

    fs.apply_paper_style()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=fs.mm(2 * 42 + 25 + 3 * fs.AXIS_MM, 60), layout="constrained",
                                  width_ratios=[1, 0.55, 1])
    for r in rows:
        a.errorbar(r["gain_validation"], r["gain_monitor"], xerr=r["se_validation"], yerr=r["se_monitor"], fmt="o",
                   ms=2.5, color=ISLAND_COLORS[r["island"]], elinewidth=0.5, capsize=0)
        lab = "new" if r["tag"].startswith("new") else str(int(r["tag"][3:]))
        a.annotate(lab, (r["gain_validation"], r["gain_monitor"]), xytext=(2, 1), textcoords="offset points",
                   fontsize=5)
    lim = [0.08, max(max(r["gain_validation"], r["gain_monitor"]) for r in rows) + 0.04]
    a.plot(lim, lim, color="0.6", ls=":", lw=0.6)
    a.set(xlabel="Gain over best seed, validation (×100)", ylabel="Gain over best seed, monitor (×100)",
          xlim=lim, ylim=lim)
    b.bar([0, 1], means, yerr=[eb.std(), lb.std()], width=0.6, color=["0.7", OI["blue"]], error_kw={"elinewidth": 0.6})
    b.set_xticks([0, 1], ["early\ntop 10", "final\ntop 10"])
    b.set_ylabel("Mean monitor gain (×100)")
    for i in range(5):
        m = [k for k, t in enumerate(arch) if progs[t]["island"] == i]
        c.scatter(val[m], mons[m], s=8, color=ISLAND_COLORS[i], lw=0)
    c.set(xlabel="Validation score, search (×100)", ylabel="Monitor score (×100)")
    c.annotate(f"Spearman ρ = {rho:.2f}\ntop-10 overlap {overlap}/10", (0.03, 0.97), xycoords="axes fraction",
               ha="left", va="top", fontsize=5)
    for ax, letter in zip((a, b, c), "abc"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, FIGS["monitor"])

    res = {"evaluator": mon["evaluator"], "fallback_tol": FALLBACK_TOL, "first_batch": rows,
           "late_vs_early": {"early": early, "late": late, "early_mean_gain": means[0], "late_mean_gain": means[1],
                             "difference": d, "ci95": ci.tolist()},
           "final_archive": {"n": len(arch), "spearman": rho, "top10_overlap": overlap}}
    tops = [r for r in rows if r["tag"].startswith("top")]
    gm, sm = [r["gain_monitor"] for r in tops], [r["se_monitor"] for r in tops]
    new = next(r for r in rows if r["tag"].startswith("new"))
    trows = [[("new best" if r["tag"].startswith("new") else r["tag"]), f"`{r['name']}`", str(r["generation"]),
              f"{fs.num(r['gain_validation'], '+.3f')} ± {r['se_validation']:.3f}",
              f"{fs.num(r['gain_monitor'], '+.3f')} ± {r['se_monitor']:.3f}",
              f"{r['fallback']['validation']}, {r['fallback']['monitor']}"] for r in rows]
    gens = [progs[t]["generation"] for t in early]
    text = (f"![monitor](../fig_r2_monitor.svg)\n\n"
            f"**The gain over the best seed holds on unseen subjects.** On the 100 monitor subjects the early top 10 "
            f"gain {fs.num(min(gm), '+.3f')} to {fs.num(max(gm), '+.3f')} (SE {min(sm):.3f}–{max(sm):.3f}) over the "
            f"best seed, and the program that set the stopping rule's anchor (`{new['name']}`, proposal "
            f"{new['generation']}) gains {fs.num(new['gain_monitor'], '+.3f')} ± {new['se_monitor']:.3f}.\n\n"
            f"**The search's late progress does not.** The final archive's top 10 gain {fs.num(means[1], '+.3f')} on "
            f"the monitor subjects, against {fs.num(means[0], '+.3f')} for the early top 10: a difference of "
            f"{fs.num(d, '+.3f')}, 95% CI [{fs.num(ci[0], '+.3f')}, {fs.num(ci[1], '+.3f')}]. Across the final "
            f"archive of {len(arch)}, validation and monitor ranks barely agree (Spearman ρ = {rho:.2f}; {overlap} of "
            f"the validation top 10 are in the monitor top 10).\n\n"
            + table(["program", "name", "proposal", "gain, validation", "gain, monitor",
                     "fallback subjects (val, mon)"], trows)
            + f"\n\n**Figure 3.** Programs rescored with the search's evaluator (MEvolve "
              f"{mon['evaluator']['mevolve']}, 2-level, `max_em` {mon['evaluator']['max_em']}) and the per-subject "
              f"fallback (tolerance {FALLBACK_TOL} nats per trial). (a) Gain over the best seed on the 48 validation "
              f"and 100 monitor subjects, ± paired bootstrap SE over subjects ({N_BOOT:,} resamples); numbers are "
              f"early top-10 ranks. (b) Mean monitor gain of the early top 10 (proposals {min(gens)}–{max(gens)}) "
              f"and of the final archive's top 10, ± bootstrap SE. (c) The final archive's {len(arch)} programs: "
              f"search validation score against monitor score, coloured by island as in Figure 1.")
    return text, res


def write_blocks(blocks: dict[int, str]) -> None:
    text = REPORT.read_text()
    for n, body in blocks.items():
        start, end = f"<!-- BEGIN result-{n} -->", f"<!-- END result-{n} -->"
        assert text.count(start) == 1 and text.count(end) == 1, f"result-{n} markers"
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + body.strip() + "\n" + text[j:]
    REPORT.write_text(text)


def main() -> None:
    run = json.loads(RUN.read_text())
    mon = json.loads(MONITOR.read_text())
    b1, r1 = curve(run)
    b2, r2 = drift(run)
    b3, r3 = monitor(mon)
    write_blocks({1: b1, 2: b2, 3: b3})
    meta = build_meta("analysis/r2_search_curve.py", WANDB_GROUPS, study_root=STUDY, code=[__file__],
                      inputs=[RUN, MONITOR])
    write_artifact(HERE / "r2_search_curve.json", {"_meta": meta, "search": r1, "drift": r2, "monitor": r3},
                   outputs=[*FIGS.values(), *(Path(f"{REPORT}#result-{n}") for n in range(1, 4))])
    print(json.dumps({"best": r1["best"], "late_vs_early": r3["late_vs_early"], "archive": r3["final_archive"]},
                     default=float, indent=1))


if __name__ == "__main__":
    main()
