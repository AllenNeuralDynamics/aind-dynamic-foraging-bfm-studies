#!/usr/bin/env python
"""Freeze the paired standard errors of real search steps (``make freeze-stop-se``), for the stopping rule.

The `seeded-islands` search stops when 2,000 proposals pass without the best validation score rising by
more than one standard error of a paired difference between two programs. Gate 2 measured that error
between literature models that differ a lot (0.03 to 0.06, ×100); the rule compares a new best with
an earlier best of the same search, usually a close relative. This script measures it on such pairs:
in `seeded-islands-smoke` and `seeded-islands-cap12`, every new best and the best before it (the best
seed for the first).

Each program's code is read from its run's W&B archive (downloaded at the digest the committed freeze
records) and evaluated again exactly as in the search (2-level EM on the generation and validation
mice, the run's ``max_em`` and ``n_draws``), keeping each validation mouse's log-likelihood; the
reproduced score must equal the score in the run's database. The paired standard error is
``mevolve.gate2.paired_se`` (subjects as the unit, delta method). Writes ``analysis/stop_se/steps.json``.

Needs W&B (``WANDB_API_KEY``), /allen and MEvolve (``PYTHONPATH=<mevolve>/src``); run it on a full compute
node with ``sbatch`` (about an hour for 15 programs). ``stop_se.py`` reads only the committed file.
"""
from __future__ import annotations

import hashlib
import json
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
WORKERS = 56                                      # one per core on a 112-CPU node


def new_bests(programs: list[dict]) -> list[dict]:
    """The best seed, then every proposal that beat the best so far, in proposal order."""
    seeds = [p for p in programs if p["generation"] == 0 and p["seed"]]
    cur = max(seeds, key=lambda p: p["score"])
    out = [cur]
    for p in sorted((p for p in programs if p["generation"] > 0 and p["correct"]), key=lambda p: p["generation"]):
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
    sources, out = [], {"mevolve_git_sha": mevolve_sha, "runs": {}}
    for name, frozen in RUNS.items():
        run = json.loads(frozen.read_text())
        art_name, digest = run["archive"]["artifact"], run["archive"]["digest"]
        steps = new_bests(run["programs"])
        with tempfile.TemporaryDirectory() as tmp:
            art = wandb.Api().artifact(art_name)
            assert art.digest == digest, f"{art_name} has digest {art.digest}, the freeze recorded {digest}"
            root = Path(art.download(root=str(Path(tmp) / "run")))
            sources.append({"type": "url", "id": f"wandb-artifact://{art_name}"})
            config = json.loads((root / "run_config.json").read_text())
            con = sqlite3.connect(f"file:{root / 'programs.sqlite'}?mode=ro", uri=True)
            code = {i: c for i, c in con.execute("SELECT id, code FROM programs")}
            con.close()
            freeze = Freeze(Path(config["freeze"]))
            gen = [evaluate.subject_data(freeze, s) for s in cohorts[config["generation"]]]
            val = [evaluate.subject_data(freeze, s) for s in cohorts[config["validation"]]]
            progs = []
            for p in steps:
                path = Path(tmp) / f"gen_{p['generation']}_{p['id'][:8]}.py"
                path.write_text(code[p["id"]])
                records = evaluate.evaluate(path, gen + val, regime="2level", workers=WORKERS,
                                            max_em=config["max_em"], n_draws=config["n_draws"])
                v = records[len(gen):]
                score = evaluate.program_score(v)
                failed = bool(score["two_level_failed"])
                reproduced = 100 * score["score"]
                assert abs(reproduced - p["score"]) < 1e-9, f"{name} gen {p['generation']}: {reproduced} != {p['score']}"
                key = "per_subject_log_lik" if failed else "log_lik"   # the likelihood its score was made of
                progs.append({"generation": p["generation"], "id": p["id"], "seed": p["seed"], "score": p["score"],
                              "two_level_failed": failed,
                              "code_sha256": hashlib.sha256(code[p["id"]].encode()).hexdigest(),
                              "validation": [{"subject_id": r["subject_id"], "log_lik": r[key], "n_trials": r["n_trials"]}
                                             for r in v]})
                print(f"{name} gen {p['generation']}: {reproduced:.6f} reproduced", flush=True)
        pairs = [{"from": a["generation"], "to": b["generation"], **paired_se(b["validation"], a["validation"])}
                 for a, b in zip(progs, progs[1:])]
        out["runs"][name] = {"archive": run["archive"], "max_em": config["max_em"], "n_draws": config["n_draws"],
                             "programs": progs, "pairs": pairs}
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
