"""A1: per-trial GRU hidden states for one source run (stage A — needs weights, data, JAX).

    python analysis/extract_trial_states.py C-s0

Reads the model_dir written by ``fetch_models.py``, runs the wrapper's
``extract_gru_trial_states`` over the training mice's eval split, adds each session's
``task`` from the pinned snapshot's session table (``curriculum_name`` is the training
curriculum, not the task run, and is None for many older sessions), and writes
  * ``analysis/_cache/trial_states_<key>.parquet`` — the full table (gitignored);
  * ``analysis/trial_states_<key>.json`` — counts, checks and provenance (committed).

Fails unless the pooled likelihood recomputed from the table reproduces the eval likelihood
the training run logged for the same checkpoint, and the readout reconstructs every logit
from its row's state. Needs the wrapper's ``code/`` on PYTHONPATH.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

CACHE = STUDY / "analysis" / "_cache"
SNAPSHOT_ROOT = "aind-scratch-data/aind-dynamic-foraging-cache/snapshots"
SPLIT = "eval"
LIKELIHOOD_TOLERANCE = 1e-4
# The readout check guards row alignment, not precision. JAX's default GPU matmul (TF32)
# leaves ~1e-3 abs error on C-s0 (median 1.5e-4, max 1.2e-3, logits up to |4.3|), whereas
# pairing each logit with the previous trial's state gives median 0.18, max 3.8.
READOUT_TOLERANCE = 1e-2


def _wrapper_checkout() -> dict:
    import post_training_analysis  # noqa: PLC0415

    repo = Path(post_training_analysis.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    return {"path": str(repo), "head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _inputs_data_cfg(model_dir: Path) -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load((model_dir / "inputs.yaml").read_text())["data"]


def _logged_eval_likelihood(model_dir: Path, step: int) -> float:
    index = json.loads((model_dir / "outputs" / "checkpoints" / "index.json").read_text())
    return next(float(c["eval_likelihood"]) for c in index["checkpoints"] if int(c["step"]) == step)


def _session_tasks(snapshot: str) -> "pd.DataFrame":
    import pandas as pd  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    import s3fs  # noqa: PLC0415

    fs = s3fs.S3FileSystem(anon=True)  # the snapshot bucket is publicly readable
    path = f"{SNAPSHOT_ROOT}/{snapshot}/session_table.parquet"
    table = pq.read_table(fs.open(path), columns=["_session_id", "task"]).to_pandas()
    if table["_session_id"].duplicated().any():
        raise AssertionError(f"{path} has duplicate _session_id values")
    return pd.DataFrame({"session_id": table["_session_id"].astype(str), "task": table["task"]})


def main(key: str) -> None:
    from post_training_analysis.gru_trial_states import extract_gru_trial_states  # noqa: PLC0415

    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    model_key, _ = key.split("-")
    source = manifest["models"][model_key]
    run = source["runs"][key]
    model_dir = CACHE / "models" / key / "run"

    trial_df, info = extract_gru_trial_states(
        model_dir, split=SPLIT, checkpoint_policy=manifest["checkpoint_policy"]
    )
    snapshot = str(_inputs_data_cfg(model_dir)["snapshot"])
    n_rows = len(trial_df)
    trial_df = trial_df.merge(_session_tasks(snapshot), on="session_id", how="left", validate="m:1")
    if len(trial_df) != n_rows:
        raise AssertionError("Joining session tasks changed the row count")
    logged = _logged_eval_likelihood(model_dir, int(info["checkpoint_step"]))
    checks = {
        "logged_eval_likelihood": logged,
        "recomputed_eval_likelihood": info["pooled_trial_likelihood"],
        "likelihood_abs_diff": abs(info["pooled_trial_likelihood"] - logged),
        "readout_max_abs_error": info["readout_max_abs_error"],
    }

    out_parquet = CACHE / f"trial_states_{key}.parquet"
    trial_df.to_parquet(out_parquet, index=False)
    sessions = trial_df.drop_duplicates("ses_idx")
    summary = {
        "_meta": {
            **build_meta("analysis/extract_trial_states.py", [source["group"]], study_root=STUDY, kind="freeze",
                         code=[__file__], inputs=[STUDY / "analysis" / "source_runs.json"],
                         sources=[{"type": "wandb_run", "id": f"{manifest['entity']}/{manifest['project']}/{run['run_id']}"}]),
            "wrapper_checkout": _wrapper_checkout(),
        },
        "source": {"key": key, **run},
        "extraction": {k: v for k, v in info.items() if k not in ("model_dir", "params_path")},
        "checks": checks,
        "snapshot": snapshot,
        "sessions_by_task": sessions["task"].fillna("<missing>").value_counts().to_dict(),
        "sessions_by_session_curriculum": sessions["session_curriculum_name"].value_counts().to_dict(),
        "subjects_by_subject_curriculum": trial_df.drop_duplicates("subject_id")["curriculum_name"].value_counts().to_dict(),
        "columns": list(trial_df.columns),
        "cache_file": str(out_parquet.relative_to(STUDY)),
    }
    write_artifact(STUDY / "analysis" / f"trial_states_{key}.json", summary)
    print(json.dumps({"checks": checks, **summary["extraction"]}, indent=2))

    if checks["likelihood_abs_diff"] > LIKELIHOOD_TOLERANCE:
        raise AssertionError(f"Recomputed likelihood does not match the logged value: {checks}")
    if checks["readout_max_abs_error"] > READOUT_TOLERANCE:
        raise AssertionError(f"Readout does not reconstruct the logits: {checks}")


if __name__ == "__main__":
    main(sys.argv[1])
