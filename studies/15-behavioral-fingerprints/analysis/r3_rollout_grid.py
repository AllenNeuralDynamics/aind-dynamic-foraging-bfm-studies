"""Producer for r3: rollout fingerprints of the literature models, the GLM and the GRU.

Reads the committed P2 freezes: ``p2_rollouts.json`` (the self-replay and GRU gates),
``p2_rollout_gaps.json`` (per model, regime, session set and statistic: the subject-balanced
rollout gap, split-half floor, floor ratio and coverage, and the per-bin subject means),
``p2_gru_scratch.json`` (the GRU against the best literature models per mouse), ``p2_holds.json``
(the hold on/off check) and ``p2_task_specs.json`` (the tasks); and the replay gaps freeze
``p1_replay_gaps.json`` for the other mode's scores, the scatter's models and the cells that did not
reproduce. Draws the figures r2 draws, in the same order (``fingerprint_figures``), then the holds
check, and writes ``r3_rollout_grid.json`` and the report's result blocks. Offline: reads committed
files only.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
import figure_style as fs  # noqa: E402
import fingerprint_figures as ff  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS: list[str] = []
ROLLOUTS = HERE / "p2_rollouts.json"
GAPS = HERE / "p2_rollout_gaps.json"
SCRATCH = HERE / "p2_gru_scratch.json"
HOLDS = HERE / "p2_holds.json"
SPECS = HERE / "p2_task_specs.json"
REPLAY_GAPS = HERE / "p1_replay_gaps.json"
GLM = HERE / "p1_glm.json"
GRU = HERE / "p1_gru.json"
REPORT = HERE / "reports" / "r3-rollout-grid.md"
FIG_HOLDS = HERE / "fig_r3_holds.svg"
HOLDS_BLOCK, GATES_BLOCK = "result-10", "result-11"


def draw_holds(holds: dict) -> None:
    """Per model and statistic, the gap between hold-on and hold-off rollouts as a multiple of the
    subjects' split-half floor on the same sessions."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    stats = list(ff.STAT_LABELS)
    models = list(holds["models"])
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=len(stats)) + 30, 50))
    for k, m in enumerate(models):
        ys = [holds["models"][m]["statistics"][s]["ratio"] for s in stats]
        ax.plot(np.arange(len(stats)) + (k - (len(models) - 1) / 2) * 0.1, ys, "o", ms=2.5,
                color=fs.OKABE_ITO[ff.MODEL_COLORS[m]], label=m)
    ax.axhline(1, color="0.5", lw=0.5, ls="--")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([ff.STAT_LABELS[s] for s in stats], rotation=30, ha="right")
    ax.set_ylabel("hold on vs off: gap ÷ floor")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fs.save_figure(fig, FIG_HOLDS)
    plt.close(fig)


def render_holds(h: dict) -> str:
    return "\n".join([
        f"![holds](../{FIG_HOLDS.name})", "",
        f"**Figure 10.** The `BlockMinReward` check, on the {h['n_sessions']} score sessions of {h['n_subjects']} "
        "subjects that record a positive minimum. The other coupled and warm-up sessions record 0 ("
        + ", ".join(f"{k.split('|')[0]} with minimum {k.split('|')[1]}: {v}" for k, v in h["n_sessions_by_minimum"].items())
        + "). Per model and statistic: the gap between rollouts with the hold on and off (same seeds), over "
        "the subjects' bins, as a multiple of the subjects' split-half floor on those sessions. "
        f"Largest: {h['max_ratio']:.2f}."])


def render_gates(g: dict, t: dict, seed_scores: dict) -> str:
    return (
        "**Gates.** Each agent, replayed on the first rollout it played, returns the probabilities it played "
        f"with (largest difference {max(v.get('max_self_diff', 0) for v in g.values()):.1e}). The numpy GRU "
        f"replays every held-out session to the wrapper's stored p(right) within {g[ff.GRU_MODEL]['max_gate_diff']:.1e}. "
        f"Tasks: {t['n_sessions']} score sessions; {t['n_excluded']} left out (no recorded block parameters, or one "
        f"uncoupled reward probability), which removes every score session of {t['n_subjects_lost']} of the 149 "
        "subjects. Coupled blocks in the gym average "
        f"{t['block_length_check']['coupled']['gym_mean']:.1f} trials against "
        f"{t['block_length_check']['coupled']['real_mean']:.1f} in the real sessions, which count ignored trials "
        "and held blocks. GRU rollout scores per seed: " + ", ".join(f"{v:.2f}" for v in seed_scores.values()) + ".")


def main() -> None:
    rolls, gaps, scratch, holds, specs, rep, glm, gru = (json.loads(p.read_text()) for p in
                                                         (ROLLOUTS, GAPS, SCRATCH, HOLDS, SPECS, REPLAY_GAPS, GLM, GRU))
    if not rolls["passed"]:
        raise SystemExit("a P2 rollout gate failed; fix it before rendering r3")
    mode = ff.rollout_mode(gaps, rep, scratch, ff.likelihoods(glm, gru))
    shown = ff.scatter_models(rep)
    res = ff.draw_all(mode, shown)
    draw_holds(holds)
    kept = {s["subject_id"] for s in specs["sessions"]}
    seed_scores = {s: gaps["models"][ff.GRU_MODEL][s][mode.head]["score"] for s in sorted(gaps["models"][ff.GRU_MODEL])}
    tasks = {**{k: specs[k] for k in ("n_sessions", "n_excluded", "by_kind_and_group", "block_length_check")},
             "n_subjects_lost": len({e["subject_id"] for e in specs["excluded"]} - kept)}
    gates = {m: {k: v for k, v in r.items() if k.startswith("max_") or k == "n_subjects"}
             for m, r in rolls["models"].items()}
    blocks = ff.captions(mode, res, shown)
    blocks[HOLDS_BLOCK] = render_holds(holds)
    blocks[GATES_BLOCK] = render_gates(gates, tasks, seed_scores)
    text = REPORT.read_text()
    for name in sorted(blocks):
        text = ff.rewrite(text, name, blocks[name])
    REPORT.write_text(text)
    meta = build_meta("analysis/r3_rollout_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__, HERE / "fingerprint_figures.py"],
                      inputs=[ROLLOUTS, GAPS, SCRATCH, HOLDS, SPECS, REPLAY_GAPS, GLM, GRU, lit.HEADLINE_SOURCE,
                              lit.LITERATURE_SOURCE, lit.TWO_LEVEL_EM_SOURCE])
    write_artifact(HERE / "r3_rollout_grid.json",
                   {"_meta": meta, "headline_set": mode.head, "order": mode.names,
                    "two_level_failed": sorted(mode.failed), "scatter_models": shown, **res,
                    "gru_seed_scores": seed_scores,
                    "holds": {k: holds[k] for k in ("n_sessions", "n_subjects", "max_ratio",
                                                    "holds_move_less_than_floor", "n_sessions_by_minimum")},
                    "tasks": tasks, "gates": gates},
                   outputs=[*(mode.fig(f) for f in ff.FIGURES), FIG_HOLDS,
                            *(f"{REPORT}#result-{k}" for k in range(1, len(ff.FIGURES) + 3))])


if __name__ == "__main__":
    main()
