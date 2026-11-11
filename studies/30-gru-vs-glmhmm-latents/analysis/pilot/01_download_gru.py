"""Read-only download of the study-02 H256/D614 seed-0 GRU artifact and its run config.

Run with the mevolve-search env (wandb 0.31.0): wandb 0.23.1 in the shared env fails the
artifact GraphQL queries with HTTP 400. No wandb.init, no writes to W&B.
"""
import json, sys
from pathlib import Path
import wandb

rid = "nxd-3way-20260707-075401-80a6d102"
dest = Path(sys.argv[1])
api = wandb.Api(timeout=120)
arts = list(api.artifact_collection("training-output",
                                    f"AIND-disRNN/mice_ignore_scaling/gru-output-{rid}").artifacts())
art = arts[0]
art.download(root=str(dest / "outputs"))
run = api.run(f"AIND-disRNN/mice_ignore_scaling/{rid}")
info = {"run_id": rid, "artifact": f"{art.name}", "artifact_digest": art.digest,
        "artifact_size": art.size, "group": run.group, "state": run.state,
        "config": run.config,
        "summary_subset": {k: v for k, v in run.summary.items()
                           if isinstance(v, (int, float, str)) and ("likelihood" in k or "step" in k)}}
(dest / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
print("done", art.name, art.digest, run.group, run.state)
