#!/usr/bin/env python
"""Log a passed held-out fine-tune to study 30's W&B project (PLAN.md, decision (b)).

One run in group ``heldout-mice@<launch_id>`` carrying the launch's meta and revisions, the final
held-out metrics and the reproduction deltas, plus one artifact with the fine-tuned parameters,
subject maps, embeddings and metrics (overall and per mouse). Refuses unless ``reproduction.json``
passed. Writes ``<out_root>/wandb_upload.json`` (run id, artifact version and digest). Run with an
env whose wandb can write artifacts (mevolve-search), on a compute node:

    python upload_wandb.py <out_root> launch_record/<label>/hpc_<launch_id>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
ENTITY, PROJECT = "NeuralDynamics", "dfbfm_30-gru-vs-glmhmm-latents"
KEY_FILE = Path.home() / ".config" / "mevolve" / "wandb_api_key"   # the NeuralDynamics account
SOURCE_RUN = "nxd-3way-20260707-075401-80a6d102"
FILES = ("params.json", "subject_index_map.json", "subject_embeddings.pkl", "multisubject_metadata.json",
         "output_summary.json", "checkpoint_metrics.json", "per_subject_likelihood.json")


def main() -> None:
    out_root, record_path = Path(sys.argv[1]), HERE / sys.argv[2]
    repro = json.loads((out_root / "reproduction.json").read_text())
    if not repro["pass"]:
        sys.exit("reproduction.json did not pass; nothing is uploaded")
    record = json.loads(record_path.read_text())
    group = record["wandb_group"]
    launch_id = group.split("@")[1]
    (outputs,) = [p.parent for p in (out_root / "heldout").rglob("outputs/output_summary.json")]
    run_dir = outputs.parent
    os.environ["WANDB_API_KEY"] = KEY_FILE.read_text().strip()
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=group, job_type="heldout-finetune",
        name=f"heldout-ft-{SOURCE_RUN}", dir=str(out_root),
        config={"meta": {"study": HERE.parents[1].name, "variant": HERE.name, "launch_id": launch_id,
                         "label": record["label"], "note": record["note"]},
                "revisions": record["revisions"], "launch_record": str(record_path.relative_to(HERE)),
                "source_run": f"AIND-disRNN/mice_ignore_scaling/{SOURCE_RUN}",
                "prior": json.loads((out_root / "prior" / "prior.json").read_text()),
                "resolved_config": (run_dir / "resolved_config.yaml").read_text()},
    )
    summary = json.loads((outputs / "output_summary.json").read_text())["metrics"]
    run.summary.update({f"heldout/final/{k}": v for k, v in summary.items()})
    run.summary.update({f"reproduction/diff/{k}": r["diff"] for k, r in repro["metrics"].items()})
    run.summary["reproduction/pass"] = True
    art = wandb.Artifact(f"gru-heldout-finetune-{launch_id}", type="heldout-finetune",
                         metadata={"source_run": SOURCE_RUN, "wandb_group": group})
    for name in FILES:
        art.add_file(str(outputs / name), name=name)
    art.add_file(str(run_dir / "resolved_config.yaml"), name="resolved_config.yaml")
    art.add_file(str(out_root / "reproduction.json"), name="reproduction.json")
    run.log_artifact(art).wait()
    info = {"run": f"{ENTITY}/{PROJECT}/{run.id}", "artifact": art.qualified_name, "artifact_version": art.version,
            "artifact_digest": art.digest}
    run.finish()
    (out_root / "wandb_upload.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
