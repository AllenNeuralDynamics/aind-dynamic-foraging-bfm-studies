"""Gate 4, second half, after the search: the winner and the target, each scored once on held-out mice.

The winner is picked on the simulated validation mice, so its validation score is biased upward
(the winner's curse). Scoring it and the generating program once on 24 more simulated mice, with
the search's settings, shows how much (notes.md, pre-registered step 4).

    python heldout.py --run <search out_dir> --config run.json --heldout <simulated file> \
        --target <literature name or program file> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from mevolve.search import seeds, shinka_eval
    from mevolve.search.run import load_config

    for key in shinka_eval.SECRETS:          # candidate code runs below
        os.environ.pop(key, None)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the search's out_dir (programs.sqlite)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--target", required=True, help="a literature program's name, or a program file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    con = sqlite3.connect(f"file:{args.run / 'programs.sqlite'}?mode=ro", uri=True)
    pid, code, score, generation = con.execute(
        "SELECT id, code, combined_score, generation FROM programs WHERE correct = 1 "
        "ORDER BY combined_score DESC LIMIT 1").fetchone()
    con.close()
    args.out.mkdir(parents=True, exist_ok=True)
    from_file = Path(args.target).is_file()
    name = Path(args.target).stem if from_file else args.target
    programs = {"winner": args.out / "winner.py", "target": args.out / f"{name}.py"}
    programs["winner"].write_text(code)
    programs["target"].write_text(seeds.wrap(Path(args.target).read_text()) if from_file
                                  else seeds.literature_source(args.target))
    generation_file = Path(config["simulated"]["generation"])
    workers = config["max_evaluation_jobs"] * config["workers"]
    scores = {}
    for role, path in programs.items():
        m = shinka_eval.evaluate_program(path, None, None, None, None, workers, config["max_em"], config["n_draws"],
                                         (generation_file, args.heldout))
        scores[role] = {"heldout_score": m["combined_score"], "two_level_failed": m["public"]["two_level_failed"],
                        "generation_score": m["public"]["generation_score"]}
        print(f"{role}: held-out {m['combined_score']:.3f}", flush=True)
    result = {"winner_id": pid, "winner_generation": generation, "winner_validation_score": score,
              "target": name, "heldout_file": str(args.heldout), "scores": scores,
              "heldout_gap": scores["target"]["heldout_score"] - scores["winner"]["heldout_score"]}
    (args.out / "heldout.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"winner {pid} (proposal {generation}): validation {score:.3f}; held-out gap to {name} "
          f"{result['heldout_gap']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
