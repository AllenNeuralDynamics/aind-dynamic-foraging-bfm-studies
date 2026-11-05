#!/usr/bin/env python
"""Freeze gate 4 into committed files (``make freeze-gate4``) for r1.

Reads, read-only, from /allen scratch and writes under ``analysis/gate4/``:

* ``runs.json``: for each search of gate 4's second half (``gate4-recovery-1``, seeded, and
  ``gate4-noseed-1``, no seeds), the run config and summary, one row per program in Shinka's
  database (generation, time, island when proposed and migrations, parent, inspirations, edit
  type and name, score, metrics), the ledger's per-call costs, and the held-out check. Code and
  Claude's description are kept only for the programs a report shows (the starting programs, each
  step of the best score so far, and the winner); everything else stays in the run's W&B artifact
  (``archives.json``), whose database sha256 the freeze checks;
* ``first_half.json``: gate 4's first half (MEvolve 85de84f): per model, the criteria summary and,
  per simulated subject, the true and recovered typical values and spreads;
* ``example_session.json``: one session simulated from the committed generator at hand-set
  setting a, with its block reward probabilities and latent variables, for the report's figure.

Shinka's databases are in WAL mode and their ``-wal`` files are not checkpointed, so each one is
copied with its ``-wal`` to a temporary directory and the copy is opened; the originals are only
read, and their sha256 are checked before and after. Needs /allen, MEvolve (``PYTHONPATH``) and
its environment (jax, aind_behavior_gym), so it runs on a compute node; r1's producer reads only
the committed files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

OUT = HERE / "gate4"
RUNS = Path("/allen/aind/scratch/han.hou/mevolve/runs")
ARMS = {  # arm: (variant, launch id, W&B group)
    "seeded": ("gate4-recovery", "20261104-132600", "gate4-recovery@20261104-132600"),
    "noseed": ("gate4-noseed", "20261104-141503", "gate4-noseed@20261104-141503"),
}
FIRST_HALF = Path("/allen/aind/scratch/han.hou/mevolve/gate4/run_85de84f")
FIRST_HALF_MEVOLVE = "85de84f"          # the MEvolve commit gate4.json records (checked below)
GENERATOR = STUDY / "variants" / "gate4-recovery" / "generator" / "two_timescale_ck.py"
EXAMPLE = {"setting": "a", "n_trials": 400, "seed": 1}
METADATA = ("patch_type", "patch_name", "mevolve_seed")
ARCHIVES = OUT / "archives.json"        # written by archive_gate4_runs.py


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file(path: Path) -> dict:
    return {"type": "file", "id": str(path), "sha256": _sha256(path.read_bytes())}


def rows(db: Path) -> tuple[list[dict], list[dict]]:
    """Every program row and every failed edit, from a copy of the database and its -wal."""
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal"):
            if Path(f"{db}{suffix}").exists():
                shutil.copy(f"{db}{suffix}", Path(tmp) / f"programs.sqlite{suffix}")
        con = sqlite3.connect(Path(tmp) / "programs.sqlite")
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM programs ORDER BY timestamp")]
        failed = [dict(r) for r in con.execute("SELECT * FROM attempt_log")]
        con.close()
    return rows, failed


def _shown(programs: list[dict], winner_id: str) -> set[str]:
    """The programs a report shows: the starting programs, each step of the best score so far, the winner."""
    keep, best = {winner_id}, float("-inf")
    for p in sorted(programs, key=lambda p: p["timestamp"]):
        if p["generation"] == 0 or (p["score"] is not None and p["score"] > best):
            keep.add(p["id"])
        if p["score"] is not None:
            best = max(best, p["score"])
    return keep


def _programs(db: Path) -> tuple[list[dict], list[dict]]:
    rows_, failed = rows(db)
    gen = {r["id"]: r["generation"] for r in rows_}
    programs = []
    for r in rows_:
        meta = json.loads(r["metadata"] or "{}")
        moves = sorted(json.loads(r["migration_history"] or "[]"), key=lambda m: m["timestamp"])
        public = json.loads(r["public_metrics"] or "{}")
        private = json.loads(r["private_metrics"] or "{}")
        programs.append({
            "generation": r["generation"], "id": r["id"], "timestamp": r["timestamp"],
            "island_proposed": moves[0]["from"] if moves else r["island_idx"], "island_final": r["island_idx"],
            "migrations": [{"to": m["to"], "timestamp": m["timestamp"]} for m in moves],
            "parent": gen.get(r["parent_id"]), "parent_id": r["parent_id"],
            "inspirations": [gen.get(i) for i in json.loads(r["archive_inspiration_ids"] or "[]")
                             + json.loads(r["top_k_inspiration_ids"] or "[]")],
            **{k: meta.get(k) for k in METADATA},
            "correct": bool(r["correct"]), "score": r["combined_score"] if r["correct"] else None,
            "public": {k: public.get(k) for k in ("validation_score", "generation_score", "n_params", "n_latent")},
            "evaluation_seconds": private.get("seconds"),
            "patch_description": meta.get("patch_description"), "code": r["code"],
        })
    return programs, [{k: f.get(k) for k in ("generation", "stage", "reason", "error") if k in f} for f in failed]


def freeze_runs() -> tuple[dict, list[dict]]:
    runs, sources = {}, []
    archives = json.loads(ARCHIVES.read_text())["archives"]
    for arm, (variant, launch, group) in ARMS.items():
        d = RUNS / variant / launch
        db = d / "search" / "programs.sqlite"
        before = {p: _sha256(p.read_bytes()) for p in (db, Path(f"{db}-wal"))}
        archived = archives[arm]["sha256"]
        assert before == {db: archived["search/programs.sqlite"], Path(f"{db}-wal"): archived["search/programs.sqlite-wal"]}, \
            f"{db} is not the archived database ({archives[arm]['artifact']})"
        config = json.loads((d / "search" / "run_config.json").read_text())
        summary = json.loads((d / "search" / "run_summary.json").read_text())
        ledger = [json.loads(line) for line in (d / "search" / "ledger.jsonl").read_text().splitlines()]
        heldout = json.loads((d / "heldout" / "heldout.json").read_text())
        programs, failed = _programs(db)
        assert {p: _sha256(p.read_bytes()) for p in before} == before, f"{db} changed while it was read"
        assert config["wandb"]["group"] == group, (arm, config["wandb"]["group"])
        assert summary["calls"] == len(ledger) and abs(summary["cost_usd"] - sum(c["cost"] for c in ledger)) < 1e-9, arm
        winner = next(p for p in programs if p["id"] == heldout["winner_id"])
        assert winner["score"] == max(p["score"] for p in programs if p["correct"]), arm      # ties are repeats
        shown = _shown(programs, winner["id"])
        for p in programs:
            if p["id"] not in shown:
                p["code"] = p["patch_description"] = None
        runs[arm] = {"variant": variant, "launch_id": launch, "wandb_group": group, "run_config": config,
                     "summary": {k: summary[k] for k in ("seconds", "calls", "cost_usd", "stop_reasons", "islands")},
                     "ledger": [{k: c[k] for k in ("time", "seconds", "input_tokens", "output_tokens",
                                                   "cache_creation_input_tokens", "cache_read_input_tokens", "cost")}
                                for c in ledger],
                     "archive": {k: archives[arm][k] for k in ("artifact", "digest")},
                     "heldout": heldout, "programs": programs, "failed_attempts": failed}
        sources += [_file(d / "search" / f) for f in ("run_config.json", "run_summary.json", "ledger.jsonl")]
        sources += [{"type": "file", "id": str(p), "sha256": s} for p, s in before.items()]
        sources.append({"type": "url", "id": f"wandb-artifact://{archives[arm]['artifact']}"})
        sources.append(_file(d / "heldout" / "heldout.json"))
    return runs, sources


def freeze_first_half() -> tuple[dict, list[dict]]:
    import numpy as np
    from mevolve import fitting, literature, program

    result = json.loads((FIRST_HALF / "gate4.json").read_text())
    assert result["mevolve_git_sha"].startswith(FIRST_HALF_MEVOLVE), result["mevolve_git_sha"]
    sources = [_file(FIRST_HALF / "gate4.json")]
    models = {}
    for model, m in result["models"].items():
        prog = program.literature(literature.LITERATURE[model][0])
        sim_path = FIRST_HALF / f"simulated_{model}.json.gz"
        sources.append(_file(sim_path))
        with gzip.open(sim_path, "rt") as f:
            truth = {s["subject_id"]: s for s in json.load(f)["subjects"]}
        subjects = []
        for rec in m["records"]:
            t = truth[rec["subject_id"]]
            true_typical = {k: float(v) for k, v in fitting.to_params(prog, np.asarray(t["mu"])).items()}
            subjects.append({"subject_id": rec["subject_id"], "true_typical": true_typical,
                             "recovered_typical": rec["typical_params"], "true_log_sigma": t["log_sigma"],
                             "recovered_log_sigma": rec["log_sigma"], "converged": rec.get("converged")})
        models[model] = {**{k: m[k] for k in ("program_sha256", "params", "score", "converged", "ok")},
                         "bounds": {k: list(v) for k, v in prog.params.items()}, "subjects": subjects}
    return {"mevolve_git_sha": result["mevolve_git_sha"], "max_em": result["max_em"], "criteria": result["criteria"],
            "models": models}, sources


def freeze_example() -> dict:
    import jax
    import numpy as np
    from aind_behavior_gym.dynamic_foraging.task.coupled_block_task import CoupledBlockTask
    from mevolve import program

    sys.path.insert(0, str(GENERATOR.parents[1]))
    import make_truths

    prog = program.load(GENERATOR)
    p = make_truths.SETTINGS[EXAMPLE["setting"]]
    n, rng = EXAMPLE["n_trials"], np.random.default_rng(EXAMPLE["seed"])
    task = CoupledBlockTask(reward_baiting=True, num_trials=n, seed=EXAMPLE["seed"])
    task.reset()
    lat = prog.init_latents(p, {})
    choices, rewards, p_right, latents = [], [], [], []
    for _ in range(n):
        pr = float(jax.nn.softmax(prog.predict(lat, p))[1])
        latents.append([float(x) for x in np.asarray(lat)])
        c = int(rng.random() < pr)
        _, r, _, _, _ = task.step(c)
        choices.append(c), rewards.append(float(r)), p_right.append(pr)
        lat = prog.update(lat, c, r, p)
    return {**EXAMPLE, "params": p, "generator": str(GENERATOR.relative_to(STUDY)),
            "generator_sha256": _sha256(GENERATOR.read_bytes()), "latent_names": list(prog.latent_names),
            "choices": choices, "rewards": rewards, "p_right": p_right, "latents": latents,
            "p_reward": np.asarray(task.trial_p_reward[:n]).T.tolist()}


def main() -> None:
    runs, run_sources = freeze_runs()
    first_half, first_sources = freeze_first_half()
    example = freeze_example()
    OUT.mkdir(exist_ok=True)
    groups = [g for _, _, g in ARMS.values()]
    write_artifact(OUT / "runs.json", {
        "_meta": build_meta("analysis/freeze_gate4.py", groups, study_root=STUDY, kind="freeze", code=[__file__],
                            inputs=[ARCHIVES], sources=run_sources),
        "about": "Gate 4's two searches: Shinka's program table (read from a copy of each archived database with "
                 "its -wal; code and descriptions only for the programs a report shows), the run config and "
                 "summary, the ledger's per-call costs and the held-out check.",
        "runs": runs})
    write_artifact(OUT / "first_half.json", {
        "_meta": build_meta("analysis/freeze_gate4.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                            sources=first_sources),
        "about": "Gate 4's first half (MEvolve #36): true and recovered typical values (natural scale) and "
                 "log spreads (probit scale) of 24 simulated subjects per model, and gate4.json's criteria.",
        **first_half})
    write_artifact(OUT / "example_session.json", {
        "_meta": build_meta("analysis/freeze_gate4.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                            sources=[{"type": "file", "id": str(GENERATOR), "sha256": example["generator_sha256"]}]),
        "about": "One session simulated from the committed generator at its typical values (no session "
                 "spread), on the coupled block task with baiting.", **example})
    print(f"froze {sum(len(r['programs']) for r in runs.values())} programs, "
          f"{len(first_half['models'])} first-half models and a {example['n_trials']}-trial example to {OUT}")


if __name__ == "__main__":
    main()
