"""Producer for r2: replay fingerprints of the literature models, the GLM and the GRU.

Reads the committed P1 freezes: ``p1_replay.json``, ``p1_glm.json`` and ``p1_gru.json`` (the gates),
``p1_replay_gaps.json`` (per model, regime, session set and rate statistic: the subject-balanced
replay gap, binomial floor and floor ratio, and the per-bin subject means) and
``p1_gru_scratch.json`` (the GRU against the best literature models per mouse); and the rollout
gaps freeze ``p2_rollout_gaps.json`` for the other mode's scores. Draws the figures r3 draws, in
the same order (``fingerprint_figures``), and writes ``r2_replay_grid.json`` and the report's result
blocks. Offline: reads committed files only.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
import fingerprint_figures as ff  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS: list[str] = []
REPLAY = HERE / "p1_replay.json"
GLM = HERE / "p1_glm.json"
GRU = HERE / "p1_gru.json"
SCRATCH = HERE / "p1_gru_scratch.json"
GAPS = HERE / "p1_replay_gaps.json"
ROLLOUT_GAPS = HERE / "p2_rollout_gaps.json"
REPORT = HERE / "reports" / "r2-replay-grid.md"
GATES_BLOCK = "result-11"


def gate_summary(replay: dict, glm: dict, gru: dict, gaps: dict) -> dict:
    """The replay gates, over the cells the grid shows; a cell the gaps freeze left out (a failed
    2-level fit that does not reproduce) is listed, not counted."""
    skip = {(e["model"], e["regime"]) for e in gaps["excluded"]}
    out = {"library_git_sha": replay["fingerprints_library"]["git_sha"],
           "mevolve_git_sha": replay["mevolve_git_sha"], "tolerance": replay["tolerance"],
           "n_models": len(replay["models"]), "n_subjects": replay["n_subjects"],
           "max_abs_log_lik_diff": {}, "max_fingerprint_diff": 0.0, "excluded": gaps["excluded"],
           "glm_max_abs_diff": glm["max_abs_diff_sklearn"],
           "glm_pooled_heldout_likelihood": glm["pooled_heldout_likelihood"],
           "gru": {"wandb_group": gru["launch"]["wandb_group"], "tolerance": gru["tolerance"]["likelihood"],
                   "n_seeds": len(gru["seeds"]),
                   "n_subjects_referenced": min(g["n_subjects_referenced"] for g in gru["seeds"].values()),
                   "subjects_unreferenced": gru["seeds"]["seed0"]["subjects_unreferenced"],
                   "subjects_in_training": gru["seeds"]["seed0"]["subjects_in_training"],
                   "max_abs_likelihood_diff": max(g["max_abs_likelihood_diff"] for g in gru["seeds"].values()),
                   "seed_scores": {reg: gaps["scores"][gru["model"]][reg][gaps["headline_set"]]
                                   for reg in gru["seeds"]}}}
    shown = [(m, reg, g) for m, regimes in replay["models"].items() for reg, g in regimes.items()
             if (m, reg) not in skip]
    out["passed"] = glm["passed"] and gru["passed"] and all(g["passed"] for _, _, g in shown)
    for _, reg, g in shown:
        out["max_abs_log_lik_diff"][reg] = max(out["max_abs_log_lik_diff"].get(reg, 0.0),
                                               g["max_abs_log_lik_diff"])
        out["max_fingerprint_diff"] = max(out["max_fingerprint_diff"], g.get("max_fingerprint_diff", 0.0))
    return out


def render_gates(gates: dict, n_subjects: int) -> str:
    return (
        f"**Gates** (library `{gates['library_git_sha'][:7]}`, MEvolve `{gates['mevolve_git_sha'][:7]}`). "
        f"Every replay shown ({gates['n_models']} of study 12's models × {gates['n_subjects']} subjects) "
        "reproduces the held-out log likelihood study 12 stored for the subject: largest difference "
        + ", ".join(f"{ff.REGIMES[k]} {v:.1e}" for k, v in gates["max_abs_log_lik_diff"].items())
        + f" (tolerance {gates['tolerance']['log_lik']:.0e}). "
        + "".join(f"Left out: {e['model']} ({ff.REGIMES[e['regime']]}; {e['n_subjects_replay_differs']} "
                  f"subjects differ): {e['reason']}. " for e in gates["excluded"])
        + "MEvolve's `replay_gap`, rebuilt from the 2-level tallies, equals the "
        f"stored one to {gates['max_fingerprint_diff']:.1e}. The GLM's replay equals scikit-learn's "
        f"prediction to {gates['glm_max_abs_diff']:.1e}; its pooled held-out likelihood is "
        f"{gates['glm_pooled_heldout_likelihood']:.4f}. "
        f"The GRU (study 01's v1 GRU at D = 614, re-fine-tuned on the held-out subjects, launch "
        f"`{gates['gru']['wandb_group']}`, which logs no W&B runs) scores exactly the freeze's sessions and choices; on the "
        f"{gates['gru']['n_subjects_referenced']} subjects study 01 r2 also scored, each seed's "
        f"likelihood equals study 01's to {gates['gru']['max_abs_likelihood_diff']:.1e} (tolerance "
        f"{gates['gru']['tolerance']:.0e}; study 01 scored on GPUs, this re-fine-tune on CPUs). Its "
        f"{len(gates['gru']['subjects_in_training'])} other subjects "
        f"({', '.join(gates['gru']['subjects_in_training'])}) were among the 614 its checkpoint was "
        f"trained on, so the GRU's numbers leave them out: its cells and score are over "
        f"{n_subjects - len(gates['gru']['subjects_in_training'])} subjects, every other "
        f"model's over {n_subjects}. Replay scores per seed: "
        + ", ".join(f"{v:.2f}" for v in gates["gru"]["seed_scores"].values()) + ".")


def main() -> None:
    replay, glm, gru, gaps, scratch, rollout = (json.loads(p.read_text()) for p in
                                                (REPLAY, GLM, GRU, GAPS, SCRATCH, ROLLOUT_GAPS))
    gates = gate_summary(replay, glm, gru, gaps)
    if not gates["passed"]:
        raise SystemExit("a P1 replay gate failed; fix it before rendering r2")
    mode = ff.replay_mode(gaps, rollout, scratch, ff.likelihoods(glm, gru), gru)
    shown = ff.scatter_models(gaps)
    res = ff.draw_all(mode, shown)
    blocks = ff.captions(mode, res, shown)
    blocks[GATES_BLOCK] = render_gates(gates, len(gaps["subjects"]))
    text = REPORT.read_text()
    for name in sorted(blocks):
        text = ff.rewrite(text, name, blocks[name])
    REPORT.write_text(text)
    by_set = {m: {reg: {name: {s: c["floor_ratio"] for s, c in cells.items()} for name, cells in sets.items()}
                  for reg, sets in gaps["models"][m].items()} for m in mode.names}
    meta = build_meta("analysis/r2_replay_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__, HERE / "fingerprint_figures.py"],
                      inputs=[REPLAY, GLM, GRU, GAPS, SCRATCH, ROLLOUT_GAPS, lit.HEADLINE_SOURCE,
                              lit.LITERATURE_SOURCE, lit.TWO_LEVEL_EM_SOURCE])
    write_artifact(HERE / "r2_replay_grid.json",
                   {"_meta": meta, "gates": gates, "headline_set": mode.head, "order": mode.names,
                    "two_level_failed": sorted(mode.failed), "scatter_models": shown,
                    "floor_ratio_by_set": by_set, **res},
                   outputs=[*(mode.fig(f) for f in ff.FIGURES),
                            *(f"{REPORT}#result-{k}" for k in (*range(1, len(ff.FIGURES) + 1), 11))])


if __name__ == "__main__":
    main()
