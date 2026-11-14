#!/usr/bin/env python
"""Prepare the scoring launch: download every model in models.yaml, write held-out mouse lists, and
write jobs.csv (one row per model x pass x session-lambda mode).

- Models go to ``$S17_SCORING/models/<name>/run`` via the wrapper's ``hydrate_model_dir``, each with
  its own entity's W&B key (AIND-disRNN: the ~/.netrc key; NeuralDynamics: ~/.config/wandb/
  api_key.NeuralDynamics), so the scoring jobs read local copies (``--model-dir``).
- The best_eval checkpoint (highest eval_likelihood in outputs/checkpoints/index.json, the wrapper's
  rule) and its session_curriculum_lambda are recorded per model. Every model is scored at its own
  lambda (``checkpoint``); a model whose lambda is below 1 is also scored at lambda 1 (PLAN.md).
- Held-out mice = reference/folds.json ``folds[heldout]`` minus ``exclude``.

Run on a compute node with the wrapper (merge of PR #128) on PYTHONPATH:
    PYTHONPATH=<wrapper>/code S17_SCORING=/allen/aind/scratch/han.hou/s17_scoring \\
      <mevolve-search env>/bin/python variants/scoring/prepare.py
"""
from __future__ import annotations

import csv
import json
import netrc
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
ROOT = Path(os.environ["S17_SCORING"])
KEYS = {"AIND-disRNN": netrc.netrc().authenticators("api.wandb.ai")[2],
        "NeuralDynamics": Path("~/.config/wandb/api_key.NeuralDynamics").expanduser().read_text().strip()}


def best_eval(model_dir: Path) -> tuple[int, float]:
    index = json.loads((model_dir / "outputs" / "checkpoints" / "index.json").read_text())
    best = max(index["checkpoints"], key=lambda c: float(c.get("eval_likelihood", float("-inf"))))
    return int(best["step"]), float(best["session_curriculum_lambda"])


def main() -> None:
    from post_training_analysis.wandb_model_dir import hydrate_model_dir

    models = yaml.safe_load((HERE / "models.yaml").read_text())["models"]
    folds = json.loads((STUDY / "reference" / "folds.json").read_text())["folds"]
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in models:
        model_dir = ROOT / "models" / m["name"] / "run"
        if not (model_dir / "outputs" / "checkpoints" / "index.json").exists():
            os.environ["WANDB_API_KEY"] = KEYS[m["entity"]]
            hydrate_model_dir(m["run_id"], project=m["project"], entity=m["entity"], dest=model_dir.parent)
        step, lam = best_eval(model_dir)
        mice = [s for s in folds[m["heldout"]] if s not in set(m["exclude"])]
        subjects = ROOT / "inputs" / f"{m['name']}.txt"
        subjects.write_text("\n".join(mice) + "\n")
        for mode in ["checkpoint"] + (["1"] if lam < 1 else []):
            for p in ("A", "B"):
                jobs.append({"job": len(jobs), "name": m["name"], "pass": p, "session_lambda": mode,
                             "best_eval_step": step, "checkpoint_lambda": lam, "n_mice": len(mice),
                             "model_dir": str(model_dir), "subjects": str(subjects),
                             "out": f"{m['name']}/pass{p}_lam-{mode}"})
        print(f"{m['name']:22s} step {step:6d} lambda {lam:.2f} mice {len(mice)}", flush=True)
    with open(HERE / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0]))
        w.writeheader()
        w.writerows(jobs)
    print(f"{len(jobs)} jobs")


if __name__ == "__main__":
    main()
