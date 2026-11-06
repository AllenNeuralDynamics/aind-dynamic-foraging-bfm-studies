#!/usr/bin/env python
"""The seeded-islands-cap12 result (``make cap12-result``): the numbers the variant's notes
pre-registered, from the committed freezes of this run and of seeded-islands-smoke. Offline.

Writes ``analysis/seeded_islands_cap12.json`` and ``analysis/fig_seeded_islands_cap12.svg``: the pass
checks (proposals, dollars, hours, API errors, evaluation steps, archive); against the smoke run, the
use of the extra room (scored proposals above 8 parameters or 6 latent variables), the best program's
counts and the numbers written into its code, the best score and the best by lineage, and throughput
(proposals per hour, CPU-minutes per proposal; CPUs from each launch record's ``search.sbatch``). CPU-minutes
use the search's own time, the only time both freezes hold; cap12's are also given over its whole Slurm
allocation, which adds startup and the end-of-run archive.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

RUNS = {"smoke": HERE / "seeded_islands_smoke" / "run.json", "cap12": HERE / "seeded_islands_cap12" / "run.json"}
SBATCH = {k: STUDY / "variants" / f"seeded-islands-{k}" / "launch_record" / f"seeded-islands-{k}-1" / "search.sbatch"
          for k in RUNS}                                 # as launched, not the live variant
OUT, FIG = HERE / "seeded_islands_cap12.json", HERE / "fig_seeded_islands_cap12.svg"
GROUPS = ["seeded-islands-smoke@20261105-011815", "seeded-islands-cap12@20261105-100249"]
N_PROPOSALS, MAX_USD, MAX_HOURS = 100, 20.0, 6.0      # notes.md, pre-registered
OLD_PARAMS, OLD_LATENT = 8, 6                          # ADR 0005
LINEAGES = ("template", "Miller-RHG", "Lebedeva-PR", "Lopez-DoubleTrace", "Eckstein-BI")
COLORS = {"smoke": fs.OKABE_ITO["blue"], "cap12": fs.OKABE_ITO["vermillion"]}
LABELS = {"smoke": "8 params, 6 latents", "cap12": "12 params, 8 latents"}


def cpus(sbatch: Path) -> int:
    opt = dict(re.findall(r"#SBATCH --(ntasks|cpus-per-task)=(\d+)", sbatch.read_text()))
    return int(opt["ntasks"]) * int(opt["cpus-per-task"])


def constants(code: str) -> list[dict]:
    """Numbers written into the program other than 0 and 1, outside PARAMS, N_LATENT and LATENT_NAMES
    and outside index brackets (``latents[3]`` names a latent variable, it is not a constant)."""
    tree, lines = ast.parse(code), code.splitlines()
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("PARAMS", "N_LATENT", "LATENT_NAMES")
                                                for t in node.targets):
            skip |= {id(n) for n in ast.walk(node)}
        elif isinstance(node, ast.Subscript):
            skip |= {id(n) for n in ast.walk(node.slice)}
    found = [n for n in ast.walk(tree) if id(n) not in skip and isinstance(n, ast.Constant)
             and type(n.value) in (int, float) and n.value not in (0, 1)]
    return [{"line": n.lineno, "value": n.value, "text": lines[n.lineno - 1].strip()}
            for n in sorted(found, key=lambda n: (n.lineno, n.col_offset))]


def summarize(run: dict, n_cpus: int) -> dict:
    progs = run["programs"]
    by_id = {p["id"]: p for p in progs}

    def lineage(p):
        while p["parent_id"] in by_id:
            p = by_id[p["parent_id"]]
        return p["seed"] or "template"

    proposals = [p for p in progs if p["generation"] > 0]
    scored = [p for p in proposals if p["correct"]]
    seeds = {p["seed"] or "template": p["score"] for p in progs if p["generation"] == 0}
    best_seed = max(v for k, v in seeds.items() if k != "template")
    best = max(scored, key=lambda p: p["score"])
    curve, cur = [], best_seed
    for p in sorted(proposals, key=lambda p: p["generation"]):
        cur = max(cur, p["score"]) if p["correct"] else cur
        curve.append([p["generation"], cur])
    hours = run["summary"]["seconds"] / 3600
    return {
        "proposals": len(proposals) + len(run["no_program"]), "cost_usd": run["summary"]["cost_usd"], "hours": hours,
        "api_errors": len(run["summary"]["errors"]), "n_scored": len(scored),
        "throughput": {"cpus": n_cpus, "proposals_per_hour": N_PROPOSALS / hours,
                       "search_cpu_minutes_per_proposal": n_cpus * hours * 60 / N_PROPOSALS,
                       "allocation_cpu_minutes_per_proposal": (n_cpus * run["slurm"]["seconds"] / 60 / N_PROPOSALS
                                                               if "slurm" in run else None)},
        "above_old_limits": {"params": sum(p["n_params"] > OLD_PARAMS for p in scored),
                             "latent": sum(p["n_latent"] > OLD_LATENT for p in scored),
                             "either": sum(p["n_params"] > OLD_PARAMS or p["n_latent"] > OLD_LATENT for p in scored)},
        "n_params": {str(k): sum(p["n_params"] == k for p in scored) for k in range(1, 13)},
        "n_latent": {str(k): sum(p["n_latent"] == k for p in scored) for k in range(1, 9)},
        "best": {"generation": best["generation"], "score": best["score"], "best_seed": best_seed,
                 "gain_over_best_seed": best["score"] - best_seed, "n_params": best["n_params"],
                 "n_latent": best["n_latent"], "lineage": lineage(best)},
        "better_than_parent": sum(1 for p in scored if p["parent_id"] in by_id and by_id[p["parent_id"]]["correct"]
                                  and p["score"] > by_id[p["parent_id"]]["score"]),
        "two_level_failed": sum(bool(p["two_level_failed"]) for p in scored),
        "two_level_failed_by_size": {f"{side}_{OLD_PARAMS}_params": {"failed": sum(bool(p["two_level_failed"]) for p in g), "of": len(g)}
                                     for side, g in (("above", [p for p in scored if p["n_params"] > OLD_PARAMS]),
                                                     ("at_most", [p for p in scored if p["n_params"] <= OLD_PARAMS]))},
        "rejected_by_checks": [{"generation": p["generation"], "reason": p["failure"]} for p in proposals if not p["correct"]],
        "no_program": run["no_program"],
        "lineages": {k: {"proposals": sum(1 for p in proposals if lineage(p) == k), "seed": seeds[k],
                         "best": max([p["score"] for p in scored if lineage(p) == k] + [seeds[k]])} for k in LINEAGES},
        "best_so_far": curve,
    }


def figure(res: dict) -> None:
    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(3 * fs.AXIS_MM + 3 * fs.SQUARE_PANEL_MM, 48), layout="constrained")
    ax = axes[0]
    for k in RUNS:
        g, s = np.array(res[k]["best_so_far"]).T
        ax.step(g, s, where="post", color=COLORS[k], label=LABELS[k])
    seed = res["smoke"]["best"]["best_seed"]                # the same seeds in both runs
    ax.axhline(seed, color="0.5", ls=":", lw=0.6)
    ax.annotate("best seed", (98, seed), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom", color="0.4")
    ax.set(xlabel="Proposal", ylabel="Best validation score so far", xlim=(0, 100))
    ax.set_ylim(bottom=seed - 0.01)
    for ax, key, old, n in ((axes[1], "n_params", OLD_PARAMS, 12), (axes[2], "n_latent", OLD_LATENT, 8)):
        x = np.arange(1, n + 1)
        for i, k in enumerate(RUNS):
            ax.bar(x + (i - 0.5) * 0.4, [res[k][key][str(v)] for v in x], width=0.4, color=COLORS[k], label=LABELS[k])
        ax.axvline(old + 0.5, color="0.5", ls=":", lw=0.6)
        ax.set(xticks=x, ylabel="Scored proposals",
               xlabel="Parameters" if key == "n_params" else "Latent variables")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=2)
    for ax, letter in zip(axes, "abc"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, FIG)


def main() -> None:
    runs = {k: json.loads(p.read_text()) for k, p in RUNS.items()}
    res = {k: summarize(runs[k], cpus(SBATCH[k])) for k in RUNS}
    cap = runs["cap12"]
    steps = cap["slurm"]["steps"]
    n_programs = sum(p["generation"] > 0 or not p["seed"] for p in cap["programs"])   # proposals + template; seeds are cached
    slurm_failures = [p["generation"] for p in cap["programs"] if p["failure"] and re.search(r"srun|slurm", p["failure"], re.I)]
    c = res["cap12"]
    passed = {"proposals": c["proposals"] == N_PROPOSALS, "cost": c["cost_usd"] <= MAX_USD, "hours": c["hours"] <= MAX_HOURS,
              "api_errors": c["api_errors"] == 0,
              "steps": len(steps) == n_programs and all(s["state"] == "COMPLETED" for s in steps) and not slurm_failures,
              "archive": bool(cap["archive"]["digest"])}
    out = {
        "pass": {"passed": all(passed.values()), "checks": passed, "stop_reasons": cap["summary"]["stop_reasons"],
                 "archive": cap["archive"], "programs_evaluated": n_programs,
                 "steps": {"n": len(steps), "states": {s: sum(x["state"] == s for x in steps) for s in {x["state"] for x in steps}},
                           "nodes": len({x["node"] for x in steps}), "median_seconds": float(np.median([x["seconds"] for x in steps]))},
                 "slurm_failures": slurm_failures},
        "best_constants": constants(cap["best_code"]["code"]),
        **res,
    }
    figure(res)
    write_artifact(OUT, {"_meta": build_meta("analysis/seeded_islands_cap12.py", GROUPS, study_root=STUDY,
                                             code=[__file__], inputs=[*RUNS.values(), *SBATCH.values()]),
                         "about": "seeded-islands-cap12 against seeded-islands-smoke: the pre-registered numbers of its notes.",
                         **out}, outputs=[FIG])
    print(json.dumps({"pass": out["pass"]["checks"], "best": c["best"], "above": c["above_old_limits"]}, default=float))


if __name__ == "__main__":
    main()
