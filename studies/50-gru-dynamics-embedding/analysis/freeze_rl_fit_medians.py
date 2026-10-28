"""Freeze the median fitted learning rates r6 cites into ``rl_fit_medians.json``.

r6 quotes these medians next to the GRU's implied update fraction. The per-subject fits live in
the downloaded baseline outputs under ``_cache/models/rl/<run>/`` (fetched by
``c2_rl_rollouts.py``), which CI does not have, so ``report_r6.py`` reads this file instead.

    python analysis/freeze_rl_fit_medians.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent
STUDY = ANALYSIS.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

OUT = ANALYSIS / "rl_fit_medians.json"
MANIFEST = ANALYSIS / "source_runs.json"
# baseline (key in source_runs.json["rl_baselines"]["models"]) -> fitted parameter quoted in r6
PARAMS = {"bari": "learn_rate", "ctt": "learn_rate"}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rl = manifest["rl_baselines"]
    medians, sources = {}, []
    for name, param in PARAMS.items():
        model = rl["models"][name]
        path = ANALYSIS / "_cache" / "models" / "rl" / model["run_id"] / "baseline_rl_output.json"
        fits = json.loads(path.read_text())["fitted_params_per_subject"]
        values = [f["fitted_params"][param] for f in fits.values()]
        medians[name] = {"run_id": model["run_id"], "param": param, "median": float(np.median(values)),
                         "n_subjects": len(values)}
        sources.append({"type": "wandb_run", "id": f"{manifest['entity']}/{rl['project']}/{model['run_id']}",
                        "sha256": file_digest(path)})
    # Study 01 launched these runs: they are sources here, not this study's groups.
    meta = build_meta("analysis/freeze_rl_fit_medians.py", [], study_root=STUDY,
                      kind="freeze", code=[__file__], inputs=[MANIFEST], sources=sources)
    write_artifact(OUT, {"_meta": meta, "medians": medians})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
