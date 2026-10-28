"""A1: per-trial GRU hidden states for one source run (stage A — needs weights, data, JAX).

    python analysis/extract_trial_states.py R-s0

Reads the model_dir written by ``fetch_models.py``, runs the wrapper's
``extract_gru_trial_states`` over the training mice's eval split, adds each session's
``task`` from the pinned snapshot's session table, adds the response input the GRU
received on each row, and writes
  * ``analysis/_cache/trial_states_<key>.parquet`` — the full table (gitignored);
  * ``analysis/trial_states_<key>.json`` — counts, checks and provenance (committed).

Response columns (row ``t``; see the wrapper's ``RESPONSE_COLUMNS``):
  * ``reaction_time`` — trial ``t``'s raw RT in seconds (NaN when unmatched);
  * ``log_reaction_time`` — trial ``t``'s encoded RT: log, clipped, globally standardized;
  * ``prev_log_rt_input`` — the RT input the GRU received on row ``t``, i.e. the previous
    engaged trial's ``log_reaction_time``, and the padding value -1 on a session's first
    trial (as for every input channel). Only written when the run feeds RT.
  * ``n_lick_left`` / ``n_lick_right`` — trial ``t``'s ENCODED lick counts (standardized; the
    encoder overwrites the raw counts); ``lick_left_count`` / ``lick_right_count`` — the raw
    counts recovered from them; ``prev_lick_left_input`` / ``prev_lick_right_input`` — the
    lick inputs the GRU received on row ``t`` (-1 on a session's first trial). The last four
    are only written when the run feeds lick counts.

Fails unless the pooled likelihood recomputed from the table reproduces the eval likelihood
the training run logged for the same checkpoint, and the readout reconstructs every logit
from its row's state. Needs the wrapper's ``code/`` on PYTHONPATH.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402

CACHE = STUDY / "analysis" / "_cache"
SNAPSHOT_ROOT = "aind-scratch-data/aind-dynamic-foraging-cache/snapshots"
SPLIT = "eval"
LIKELIHOOD_TOLERANCE = 1e-4
# The readout check guards row alignment, not precision (Study 50: GPU TF32 leaves ~1e-3,
# a one-row misalignment gives ~0.2 median).
READOUT_TOLERANCE = 1e-2
# Study 51's extraction peaked at 31 GB; refuse to start without headroom.
MIN_AVAILABLE_GB = 40
# ``create_disrnn_dataset`` pads row 0 of every session's inputs with -1.
FIRST_TRIAL_INPUT = -1.0
TIMING_KEYS = ("enabled", "shuffle", "reaction_time", "lick_counts")
LICK_CENTER, LICK_SCALE = 3.139, 4.159  # wrapper ``trial_timing_features`` constants


def _require_memory() -> None:
    """Fail fast, before loading anything, if the host cannot hold the extraction's peak."""
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available_gb = int(meminfo["MemAvailable"].split()[0]) / 2**20
    if available_gb < MIN_AVAILABLE_GB:
        raise MemoryError(
            f"{available_gb:.0f} GB available, extraction needs ~{MIN_AVAILABLE_GB} GB; "
            "wait for other jobs to finish rather than run alongside them"
        )


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


def _add_prev_input(trial_df: "pd.DataFrame", column: str, name: str) -> "pd.DataFrame":
    """``name``: the previous row's ``column`` (what the GRU saw on each row), −1 on row 0."""
    ordered = trial_df.sort_values(["ses_idx", "trial_idx"], kind="stable")
    if not (ordered.groupby("ses_idx", sort=False)["trial_idx"].diff().dropna() == 1).all():
        raise AssertionError("trial_idx is not a contiguous per-session timestep")
    previous = ordered.groupby("ses_idx", sort=False)[column].shift(1)
    # Assigned by index, so the table keeps the extractor's row order.
    trial_df[name] = previous.fillna(FIRST_TRIAL_INPUT).astype(np.float32)
    return trial_df


def _add_rt_input(trial_df: "pd.DataFrame") -> "pd.DataFrame":
    """``prev_log_rt_input``: the RT channel the GRU saw on each row."""
    return _add_prev_input(trial_df, "log_reaction_time", "prev_log_rt_input")


def _add_lick_inputs(trial_df: "pd.DataFrame") -> "pd.DataFrame":
    """Raw lick counts of each trial and the two lick channels the GRU saw on each row."""
    for side in ("left", "right"):
        counts = trial_df[f"n_lick_{side}"] * LICK_SCALE + LICK_CENTER
        if np.max(np.abs(counts - np.rint(counts))) > 1e-3:
            raise AssertionError(f"n_lick_{side} does not decode to integer counts")
        trial_df[f"lick_{side}_count"] = np.rint(counts).astype(np.int32)
        trial_df = _add_prev_input(trial_df, f"n_lick_{side}", f"prev_lick_{side}_input")
    return trial_df


def _lick_summary(trial_df: "pd.DataFrame") -> dict:
    chosen = np.where(trial_df["action"] == 1, trial_df["lick_right_count"], trial_df["lick_left_count"])
    other = np.where(trial_df["action"] == 1, trial_df["lick_left_count"], trial_df["lick_right_count"])
    by_reward = {}
    for r in (0, 1):
        m = trial_df["reward"].to_numpy() == r
        by_reward[str(r)] = {"mean_chosen": float(chosen[m].mean()), "mean_other": float(other[m].mean()),
                             "fraction_zero_licks": float(np.mean((chosen[m] + other[m]) == 0))}
    return {"by_reward": by_reward, "fraction_zero_licks": float(np.mean((chosen + other) == 0)),
            "max_count": int(max(trial_df["lick_left_count"].max(), trial_df["lick_right_count"].max()))}


def _rt_summary(trial_df: "pd.DataFrame") -> dict:
    rt = trial_df["reaction_time"].to_numpy(dtype=float)
    finite = np.isfinite(rt)
    quantiles = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    return {
        "n_rt_missing": int((~finite).sum()),
        "reaction_time_s_quantiles": dict(zip(map(str, quantiles), np.quantile(rt[finite], quantiles).tolist())),
        "log_reaction_time_mean": float(trial_df["log_reaction_time"].mean()),
        "log_reaction_time_sd": float(trial_df["log_reaction_time"].std()),
        "n_first_trial_rows": int((trial_df["trial_idx"] == 1).sum()),
    }


def main(key: str) -> None:
    _require_memory()
    from post_training_analysis.gru_trial_states import extract_gru_trial_states  # noqa: PLC0415

    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    model_key, _ = key.split("-")
    source = manifest["models"][model_key]
    run = source["runs"][key]
    model_dir = CACHE / "models" / key / "run"
    data_cfg = _inputs_data_cfg(model_dir)
    timing = {k: data_cfg["timing_features"].get(k) for k in TIMING_KEYS}
    if timing != source["timing_features"]:
        raise AssertionError(f"{key} model_dir timing_features {timing} != manifest {source['timing_features']}")

    trial_df, info = extract_gru_trial_states(
        model_dir, split=SPLIT, checkpoint_policy=manifest["checkpoint_policy"]
    )
    if info["checkpoint_step"] != run["checkpoint_step"]:
        raise AssertionError(f"Loaded step {info['checkpoint_step']}, W&B selected {run['checkpoint_step']}")
    snapshot = str(data_cfg["snapshot"])
    n_rows = len(trial_df)
    trial_df = trial_df.merge(_session_tasks(snapshot), on="session_id", how="left", validate="m:1")
    if len(trial_df) != n_rows:
        raise AssertionError("Joining session tasks changed the row count")
    if timing["reaction_time"]:
        trial_df = _add_rt_input(trial_df)
    if timing["lick_counts"]:
        trial_df = _add_lick_inputs(trial_df)
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
            **build_meta("analysis/extract_trial_states.py", [source["group"]], study_root=STUDY),
            "wrapper_checkout": _wrapper_checkout(),
        },
        "source": {"key": key, **run},
        "timing_features": timing,
        "extraction": {k: v for k, v in info.items() if k not in ("model_dir", "params_path")},
        "checks": checks,
        "snapshot": snapshot,
        "response_inputs": _rt_summary(trial_df) if timing["reaction_time"] else None,
        "lick_inputs": _lick_summary(trial_df) if timing["lick_counts"] else None,
        "sessions_by_task": sessions["task"].fillna("<missing>").value_counts().to_dict(),
        "sessions_by_session_curriculum": sessions["session_curriculum_name"].value_counts().to_dict(),
        "subjects_by_subject_curriculum": trial_df.drop_duplicates("subject_id")["curriculum_name"].value_counts().to_dict(),
        "columns": list(trial_df.columns),
        "cache_file": str(out_parquet.relative_to(STUDY)),
    }
    (STUDY / "analysis" / f"trial_states_{key}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"checks": checks, **summary["extraction"], "response_inputs": summary["response_inputs"],
                      "lick_inputs": summary["lick_inputs"]}, indent=2))

    if checks["likelihood_abs_diff"] > LIKELIHOOD_TOLERANCE:
        raise AssertionError(f"Recomputed likelihood does not match the logged value: {checks}")
    if checks["readout_max_abs_error"] > READOUT_TOLERANCE:
        raise AssertionError(f"Readout does not reconstruct the logits: {checks}")


if __name__ == "__main__":
    main(sys.argv[1])
