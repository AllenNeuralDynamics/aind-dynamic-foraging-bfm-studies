#!/usr/bin/env python
"""Freeze the paired standard errors of real search steps (``make freeze-stop-se``), for the stopping rule.

The `seeded-islands` search stops when 2,000 proposals pass without the best validation score rising by
more than one standard error of a paired difference between two programs. Gate 2 measured that error
between literature models that differ a lot (0.03 to 0.06, ×100); the rule compares a new best with
an earlier best of the same search, usually a close relative. This script measures it on such pairs:
in `seeded-islands-smoke` and `seeded-islands-cap12`, every new best and the best before it (the best
seed for the first), in the order the programs finished, as the rule counts them.

Each program's code is read from its run's W&B archive (downloaded at the digest the committed freeze
records) and evaluated again exactly as in the search (2-level EM on the generation and validation
mice, the run's ``max_em`` and ``n_draws``), keeping each validation mouse's log-likelihood; the
reproduced score must match the score in the run's database within 1e-4 (×100): on another node and
worker count the floating-point order differs, and scores agreed to between 1e-9 and 1e-5. The paired standard error is
``mevolve.gate2.paired_se`` (subjects as the unit, delta method). Writes ``analysis/stop_se/steps.json``.

Needs W&B (``WANDB_API_KEY``), /allen and MEvolve (``PYTHONPATH=<mevolve>/src``), on a compute node. Each
program's per-mouse records go to a gitignored cache (``analysis/_cache_stop_se/``, keyed by the program's code
and by the evaluation's inputs: MEvolve's commit, the cohorts, the data freeze, the fit settings) as they are made, so the
evaluations can run as a Slurm array, one program per task (``STOP_SE_ONLY=<run>:<generation>``), and a last
run writes the freeze from the cache. ``stop_se.py`` reads only the committed file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

RUNS = {"smoke": HERE / "seeded_islands_smoke" / "run.json", "cap12": HERE / "seeded_islands_cap12" / "run.json"}
GROUPS = {"smoke": "seeded-islands-smoke@20261105-011815", "cap12": "seeded-islands-cap12@20261105-100249"}
COHORTS = STUDY / "reference" / "cohorts_seed0.json"
OUT = HERE / "stop_se" / "steps.json"
CACHE = HERE / "_cache_stop_se"
TOLERANCE = 1e-4      # ×100: far below any paired standard error here (0.01 and up)
WORKERS = max(1, len(os.sched_getaffinity(0)) // 2)   # one per core; the score does not depend on it


def new_bests(programs: list[dict], finished: dict[str, float]) -> list[dict]:
    """The best seed, then every proposal that beat the best so far, in the order they finished
    (``finished``: program id -> its database timestamp), as the stopping rule sees them."""
    seeds = [p for p in programs if p["generation"] == 0 and p["seed"]]
    cur = max(seeds, key=lambda p: p["score"])
    out = [cur]
    for p in sorted((p for p in programs if p["generation"] > 0 and p["correct"]), key=lambda p: finished[p["id"]]):
        if p["score"] > cur["score"]:
            out.append(p)
            cur = p
    return out


def main() -> None:
    import wandb
    from mevolve import evaluate
    from mevolve.data import Freeze
    from mevolve.gate2 import paired_se

    mevolve_sha = subprocess.run(["git", "-C", str(Path(evaluate.__file__).parents[2]), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    cohorts = json.loads(COHORTS.read_text())["cohorts"]
    only = os.environ.get("STOP_SE_ONLY")           # "<run>:<generation>": evaluate one program into the cache
    sources, out = [], {"mevolve_git_sha": mevolve_sha, "runs": {}}
    for name, frozen in RUNS.items():
        run = json.loads(frozen.read_text())
        art_name, digest = run["archive"]["artifact"], run["archive"]["digest"]
        with tempfile.TemporaryDirectory() as tmp:
            art = wandb.Api().artifact(art_name)
            assert art.digest == digest, f"{art_name} has digest {art.digest}, the freeze recorded {digest}"
            root = Path(art.download(root=str(Path(tmp) / "run")))
            sources.append({"type": "url", "id": f"wandb-artifact://{art_name}"})
            config = json.loads((root / "run_config.json").read_text())
            con = sqlite3.connect(f"file:{root / 'programs.sqlite'}?mode=ro", uri=True)
            code = {i: c for i, c in con.execute("SELECT id, code FROM programs")}
            finished = {i: ts for i, ts in con.execute("SELECT id, timestamp FROM programs")}
            con.close()
            steps = new_bests(run["programs"], finished)
            inputs_key = hashlib.sha256(json.dumps(       # what else decides a program's likelihoods
                [mevolve_sha, hashlib.sha256(COHORTS.read_bytes()).hexdigest(),
                 hashlib.sha256((Path(config["freeze"]) / "manifest.json").read_bytes()).hexdigest(),
                 config["generation"], config["validation"], config["max_em"], config["n_draws"]]).encode()).hexdigest()
            freeze = Freeze(Path(config["freeze"]))
            gen = [evaluate.subject_data(freeze, s) for s in cohorts[config["generation"]]]
            val = [evaluate.subject_data(freeze, s) for s in cohorts[config["validation"]]]
            progs = []
            for p in steps:
                if only and only != f"{name}:{p['generation']}":
                    continue
                sha = hashlib.sha256(code[p["id"]].encode()).hexdigest()
                cached = CACHE / f"{name}_{p['generation']}_{sha[:16]}_{inputs_key[:16]}.json"
                if not cached.exists():
                    path = Path(tmp) / f"gen_{p['generation']}_{p['id'][:8]}.py"
                    path.write_text(code[p["id"]])
                    records = evaluate.evaluate(path, gen + val, regime="2level", workers=WORKERS,
                                                max_em=config["max_em"], n_draws=config["n_draws"])
                    v = records[len(gen):]
                    score = evaluate.program_score(v)
                    failed = bool(score["two_level_failed"])
                    reproduced = 100 * score["score"]
                    # the same evaluation on another node and worker count: equal to about 1e-5 (float order)
                    assert abs(reproduced - p["score"]) < TOLERANCE, f"{name} gen {p['generation']}: {reproduced} != {p['score']}"
                    key = "per_subject_log_lik" if failed else "log_lik"   # the likelihood its score was made of
                    CACHE.mkdir(exist_ok=True)
                    cached.write_text(json.dumps({
                        "generation": p["generation"], "id": p["id"], "seed": p["seed"], "score": p["score"],
                        "reproduced_score": reproduced, "two_level_failed": failed, "code_sha256": sha,
                        "validation": [{"subject_id": r["subject_id"], "log_lik": r[key], "n_trials": r["n_trials"]}
                                       for r in v]}))
                    print(f"{name} gen {p['generation']}: {reproduced:.6f} reproduced", flush=True)
                entry = json.loads(cached.read_text())
                entry.setdefault("reproduced_score", None)        # cached before the field: equal within 1e-9
                progs.append(entry)
            if only:
                continue
        pairs = [{"from": a["generation"], "to": b["generation"], **paired_se(b["validation"], a["validation"])}
                 for a, b in zip(progs, progs[1:])]
        out["runs"][name] = {"archive": run["archive"], "max_em": config["max_em"], "n_draws": config["n_draws"],
                             "programs": progs, "pairs": pairs}
    if only:
        return
    sources.append({"type": "file", "id": str(Path(config["freeze"]) / "manifest.json"),
                    "sha256": hashlib.sha256((Path(config["freeze"]) / "manifest.json").read_bytes()).hexdigest()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(OUT, {
        "_meta": build_meta("analysis/freeze_stop_se.py", list(GROUPS.values()), study_root=STUDY, kind="freeze",
                            code=[__file__], inputs=[*RUNS.values(), COHORTS], sources=sources),
        "about": "Paired standard errors (x100) between consecutive new bests of seeded-islands-smoke and "
                 "seeded-islands-cap12, from programs re-evaluated as in the search.",
        **out})
    print(json.dumps({n: [(q["from"], q["to"], round(q["score_diff_x100"], 4), round(q["se_x100"], 4)) for q in r["pairs"]]
                      for n, r in out["runs"].items()}))


if __name__ == "__main__":
    main()
