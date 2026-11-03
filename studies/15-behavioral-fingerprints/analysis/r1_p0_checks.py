"""Producer for r1: the P0 checks, from the three committed P0 freezes.

- ``p0_golden_mevolve.json``: the library against MEvolve's replay fingerprints;
- ``p0_golden_wrapper.json``: the library against the wrapper's committed rollout numbers;
- ``p0_task_checks.json``: held coupled blocks, reward-pair pools and parameter coverage.

Writes the figure of held blocks by task and stage, ``r1_p0_checks.json`` and the report's
result block. Offline: reads committed files only.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS: list[str] = []
MEVOLVE = HERE / "p0_golden_mevolve.json"
WRAPPER = HERE / "p0_golden_wrapper.json"
TASK = HERE / "p0_task_checks.json"
REPORT = HERE / "reports" / "r1-p0-checks.md"
FIG = HERE / "fig_r1_held_blocks.svg"
STAGES = ["STAGE_1_WARMUP", "STAGE_1", "STAGE_2", "STAGE_3", "STAGE_FINAL", "GRADUATED", "None"]
TASKS = {"Coupled Baiting": "baited", "Coupled Without Baiting": "not baited"}
STAGE_LABELS = {"STAGE_1_WARMUP": "S1 warm-up", "STAGE_1": "S1", "STAGE_2": "S2", "STAGE_3": "S3",
                "STAGE_FINAL": "final", "GRADUATED": "graduated", "None": "no stage"}


def golden_mevolve(d: dict) -> dict:
    models = d["models"]
    return {"passed": d["passed"], "n_models": len(models), "n_subjects": models[0]["n_subjects"],
            "max_library_vs_mevolve": max(m["max_abs_diff"]["library_vs_mevolve"] for m in models),
            "max_library_vs_stored": max(m["max_abs_diff"]["library_vs_stored"] for m in models),
            "replay_differs": {m["literature_model"]: m["n_subjects_replay_differs"]
                               for m in models if m["n_subjects_replay_differs"]},
            "library_git_sha": d["fingerprints_library"]["git_sha"],
            "mevolve_git_sha": d["mevolve_git_sha"]}


def golden_wrapper(d: dict) -> dict:
    return {"passed": d["passed"], "library_git_sha": d["fingerprints_library"]["git_sha"],
            "aliases": {a: {"max_abs_diff": r["max_abs_diff"], "n_subjects": r["n_subjects"],
                            "n_sessions": r["n_sessions"],
                            "count_mismatches": sum(x["n_subject_count_mismatches"]
                                                    for x in r["depths"].values())}
                        for a, r in d["aliases"].items()}}


def held_rows(d: dict) -> list[dict]:
    rows = [r for r in d["held_blocks"]["rows"] if r["task"] in TASKS and r["blocks"]]
    return sorted(rows, key=lambda r: (list(TASKS).index(r["task"]), STAGES.index(r["stage"])))


def draw(rows: list[dict]) -> None:
    fs.apply_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=len(rows)), 55))
    x = range(len(rows))
    with_min = [r.get("over_max_with_min_reward", 0) / r["blocks"] for r in rows]
    without = [r.get("over_max_without_min_reward", 0) / r["blocks"] for r in rows]
    ax.bar(x, without, 0.7, color=fs.OKABE_ITO["blue"], label="no minimum reward set")
    ax.bar(x, with_min, 0.7, bottom=without, color=fs.OKABE_ITO["vermillion"],
           label="minimum reward set")
    for i, r in enumerate(rows):
        ax.text(i, r["share_over_max"] + 0.01, f"{r['blocks']}", ha="center", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{STAGE_LABELS[r['stage']]}\n{TASKS[r['task']]}" for r in rows])
    ax.set_ylabel("coupled blocks longer than block_max")
    ax.set_ylim(0, 0.4)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fs.save_figure(fig, FIG)
    plt.close(fig)


def render(gm: dict, gw: dict, t: dict, rows: list[dict]) -> str:
    total = t["held_blocks"]["coupled_total"]
    rp = t["reward_pairs"]
    cov = t["coverage"]
    lines = [
        "**Library against MEvolve's replay fingerprints** (gate-1b per-subject fits, "
        f"{gm['n_models']} literature programs × {gm['n_subjects']} test subjects, score sessions; "
        f"library `{gm['library_git_sha'][:7]}`, MEvolve `{gm['mevolve_git_sha'][:7]}`):",
        "",
        "| check | max \\|difference\\| | pass |",
        "|---|---|---|",
        f"| library vs MEvolve's `fingerprints()`, same p(right) | {gm['max_library_vs_mevolve']:.1e} | "
        f"{'yes' if gm['max_library_vs_mevolve'] <= 1e-12 else 'no'} |",
        f"| library vs the stored fingerprints, subjects whose replay reproduces the stored log "
        f"likelihood | {gm['max_library_vs_stored']:.1e} | "
        f"{'yes' if gm['max_library_vs_stored'] <= 1e-9 else 'no'} |",
        "",
        "Subjects whose replay from the stored parameters does not reproduce the stored log "
        "likelihood (not compared): " + (", ".join(f"{k} {v}" for k, v in gm["replay_differs"].items())
                                         or "none") + ".",
        "",
        "**Library against the wrapper's committed rollout numbers** (study 05 "
        "`generative-rl-baseline`, 614 subjects, 18,124 sessions per alias; abstract history "
        "patterns at 1–3 trials back, their summaries, and the post-switch summary):",
        "",
        "| alias | max \\|difference\\| | subject-count mismatches |",
        "|---|---|---|",
    ]
    for alias, r in gw["aliases"].items():
        lines.append(f"| {alias} | {r['max_abs_diff']:.1e} | {r['count_mismatches']} |")
    lines += [
        "",
        f"**Held coupled blocks** (study 12's test subjects, score sessions; {total['blocks']} "
        f"blocks with a recorded `block_max`, last block of each session left out): "
        f"{t['held_blocks']['share_over_max']:.1%} ran past `block_max`; "
        f"{total['over_max_without_min_reward']} of {total['over_max']} were in sessions that set "
        "no minimum reward.",
        "",
        "| task | stage | sessions | blocks | past block_max | share |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['task']} | {r['stage']} | {r['sessions']} | {r['blocks']} | "
                     f"{r['over_max']} | {r['share_over_max']:.1%} |")
    within = rp["score_sessions_within_one_family"]
    lines += [
        "",
        f"**Reward pairs and coverage** ({cov['sessions']} score sessions):",
        "",
        "| check | result |",
        "|---|---|",
        f"| sessions with no `block_min` / `block_max` / `block_beta` | {cov['missing_block_max']} |",
        f"| sessions whose probabilities are only 0 and 1 | {cov['binary_probabilities']} |",
        f"| coupled score sessions whose pairs fit one GUI reward family | {within['within']} of "
        f"{within['n']} |",
        f"| (subject, task, stage) pools equal to a family prefix | "
        f"{rp['coupled_pools'] - rp['pool_matches'].get('no family prefix', 0)} of "
        f"{rp['coupled_pools']} |",
        f"| share of its pool a coupled score session shows | mean "
        f"{rp['score_session_share_of_pool_seen']['mean']:.2f}; all of it in "
        f"{rp['score_session_share_of_pool_seen']['share_seeing_all']:.0%} |",
        f"| sessions where a block's pair does not sum to the recorded base sum | "
        f"{rp['pair_sum_vs_base']['mismatched']} of {rp['pair_sum_vs_base']['checked']} |",
    ]
    return "\n".join(lines)


def _rewrite(text: str, block: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {block} -->", f"<!-- END {block} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    gm = golden_mevolve(json.loads(MEVOLVE.read_text()))
    gw = golden_wrapper(json.loads(WRAPPER.read_text()))
    task = json.loads(TASK.read_text())
    rows = held_rows(task)
    draw(rows)
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-1", render(gm, gw, task, rows)))
    meta = build_meta("analysis/r1_p0_checks.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[MEVOLVE, WRAPPER, TASK])
    write_artifact(HERE / "r1_p0_checks.json",
                   {"_meta": meta, "golden_mevolve": gm, "golden_wrapper": gw,
                    "held_blocks": {"share_over_max": task["held_blocks"]["share_over_max"],
                                    "coupled_total": task["held_blocks"]["coupled_total"],
                                    "rows": rows}},
                   outputs=[FIG, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
