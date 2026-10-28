"""Shared helpers for Study 50 producers: paths, loading the trial table, the PCA fit, style.

Producers import this with ``sys.path.insert(0, <analysis dir>)``; plotting helpers that are
not study-specific come from the wrapper (``post_training_analysis``, ``evaluation``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
# Which source run the producers analyse. The default (C-s0) writes to ``analysis/`` as
# before; any other key (phase G seed checks: ``S12_MODEL_KEY=C-s1``) reads and writes its
# JSONs, figures and explore outputs under ``analysis/seeds/<key>/`` so seed 0's results are
# never overwritten. The cache stays shared; its per-run files carry the key in the name.
MODEL_KEY = os.environ.get("S12_MODEL_KEY", "C-s0")
DEFAULT_KEY = "C-s0"
CACHE = STUDY / "analysis" / "_cache"
if MODEL_KEY == DEFAULT_KEY:
    ANALYSIS = STUDY / "analysis"
    EXPLORE = CACHE / "explore"
else:
    ANALYSIS = STUDY / "analysis" / "seeds" / MODEL_KEY
    EXPLORE = CACHE / "explore" / "seeds" / MODEL_KEY
    ANALYSIS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402,F401

HIDDEN = 128
STATE_COLUMNS = [f"rnn_state_{i}" for i in range(HIDDEN)]
N_PCS_SAVED = 20
# Trial-count variables (trials since reward / since a side was chosen) are capped here so a
# few long gaps don't dominate colour ranges and regressions.
CAP = 20

# Tasks in the analysed split (session table ``task``), largest first; Random Walk (4
# sessions) and sessions with no task are left out of per-task splits.
TASKS = ["Uncoupled Baiting", "Coupled Baiting", "Uncoupled Without Baiting", "Coupled Without Baiting"]
TASK_COLORS = {
    "Uncoupled Baiting": "#4C72B0",
    "Coupled Baiting": "#C44E52",
    "Uncoupled Without Baiting": "#55A868",
    "Coupled Without Baiting": "#8172B3",
}
LEFT_COLOR, RIGHT_COLOR, MODEL_COLOR = "#D55E00", "#0072B2", "#222222"


# What this process read, for _meta v2: committed files (``inputs``) and cache files
# (``sources`` of a freeze), by path; filled by read_json() and the cache loaders below.
_INPUTS: dict[str, Path] = {}
_CACHE_READS: dict[str, dict] = {}


def _rel(path: Path) -> str:
    """Repository-relative path, without resolving the _cache symlink out of the repository."""
    return os.path.relpath(path, STUDY.parents[1])


def _read_cache(path: Path) -> Path:
    if str(path) not in _CACHE_READS:
        _CACHE_READS[str(path)] = {"type": "file", "id": _rel(path), "sha256": file_digest(path)}
    return path


def read_json(path: Path | str) -> dict:
    """Parse a committed JSON and record it as an input of whatever this process writes."""
    path = Path(path)
    _INPUTS[str(path)] = path
    return json.loads(path.read_text())


def model_sources(key: str = MODEL_KEY) -> list[dict]:
    """The source run (and its artifact digest) every A1-derived output is computed from."""
    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    run = manifest["models"][key.split("-")[0]]["runs"][key]
    return [{"type": "wandb_run", "id": f"{manifest['entity']}/{manifest['project']}/{run['run_id']}"}]


def wandb_groups() -> list[str]:
    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    return [manifest["models"][MODEL_KEY.split("-")[0]]["group"]]


def trial_table_path(key: str = MODEL_KEY) -> Path:
    return CACHE / f"trial_states_{key}.parquet"


def load_trials(
    columns: list[str] | None = None, key: str = MODEL_KEY, sessions: list[str] | None = None
) -> pd.DataFrame:
    """The A1 table, sorted by subject / session / trial (its stored order).

    ``sessions`` restricts to those ``ses_idx`` values at read time (cheap for wide reads).
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 (freeze-only; CI's offline `all` has no pyarrow)

    filters = [("ses_idx", "in", list(sessions))] if sessions is not None else None
    return pq.read_table(_read_cache(trial_table_path(key)), columns=columns, filters=filters).to_pandas()


def pca_path(key: str = MODEL_KEY) -> Path:
    return CACHE / f"pca_{key}.npz"


def load_pca(key: str = MODEL_KEY) -> dict[str, np.ndarray]:
    """PCA written by ``b3_pca.py``: mean, components (PCs x units), eigenvalues, scores."""
    with np.load(_read_cache(pca_path(key))) as data:
        return {k: data[k] for k in data.files}


def rl_groups() -> list[str]:
    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    return [m["group"] for m in manifest["rl_baselines"]["models"].values()]


def write_json(name: str, payload: dict, produced_by: str, extra_groups: list[str] = (), *,
               kind: str = "freeze", inputs: list[Path | str] = (), sources: list[dict] = ()) -> Path:
    """Write ``analysis/<name>`` with a v2 ``_meta``.

    Every producer behind a ``freeze-*`` target is a freeze (the default): its sources are the
    model's source run plus every cache file read through the loaders here, and ``sources``
    adds any other external object it read. An offline producer that only reads committed JSON
    (``read_json``) passes ``kind="producer"``.
    """
    path = ANALYSIS / name
    read = list(dict.fromkeys([*map(Path, inputs), *_INPUTS.values()]))
    extra = {} if kind == "producer" else {
        "sources": model_sources() + list(_CACHE_READS.values()) + list(sources)}
    meta = build_meta(produced_by, wandb_groups() + list(extra_groups), study_root=STUDY, kind=kind,
                      code=[STUDY / produced_by], inputs=read, **extra)
    write_artifact(path, {"_meta": meta, **payload}, default=_json_default)
    return path


def figure_path(name: str) -> Path:
    """Committed figure: ``analysis/fig_<slug>.png`` per the posthoc-reporting layout."""
    return ANALYSIS / name


def explore_path(name: str) -> Path:
    """Exploration-only output (every colouring / condition), gitignored under ``_cache/``."""
    EXPLORE.mkdir(parents=True, exist_ok=True)
    return EXPLORE / name


def keyed_cache(name: str) -> Path:
    """A cache file that predates per-run naming: ``name`` for C-s0, ``<stem>_<key><ext>`` otherwise."""
    if MODEL_KEY == DEFAULT_KEY:
        return CACHE / name
    path = Path(name)
    return CACHE / f"{path.stem}_{MODEL_KEY}{path.suffix}"


def subsample_rows(n_rows: int, n: int, seed: int = 0) -> np.ndarray:
    """Sorted random row indices, for scatter plots of a 4.3M-trial table."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(n, n_rows), replace=False))


def best_params(key: str = MODEL_KEY) -> dict:
    """Saved params of the best_eval checkpoint (highest eval likelihood in checkpoints/index.json)."""
    outputs = CACHE / "models" / key / "run" / "outputs"
    index = json.loads(_read_cache(outputs / "checkpoints" / "index.json").read_text())
    best = max(index["checkpoints"], key=lambda c: c["eval_likelihood"])
    return json.loads(_read_cache(outputs / "checkpoints" / f"step_{best['step']}" / "params.json").read_text())


def subject_embeddings(key: str = MODEL_KEY) -> pd.DataFrame:
    """The learned 4-d subject embedding table (best_eval checkpoint), indexed by subject_id.

    This is the subject-level part of the model's context; the session-conditioning delta
    is added on top per session.
    """
    outputs = CACHE / "models" / key / "run" / "outputs"
    index_map = json.loads(_read_cache(outputs / "subject_index_map.json").read_text())["subject_id_to_index"]
    table = np.asarray(best_params(key)["multisubject_gru"]["subject_embeddings"])
    subjects = sorted(index_map, key=index_map.get)
    return pd.DataFrame(table[[index_map[s] for s in subjects]], index=pd.Index(subjects, name="subject_id"),
                        columns=[f"e{i}" for i in range(table.shape[1])])


def example_sessions(trials: pd.DataFrame) -> dict[str, str]:
    """One ``ses_idx`` per task: the session whose length is closest to that task's median.

    Deterministic (ties broken by ``ses_idx``) so B1, B2 and B3e show the same sessions.
    ``trials`` needs ``ses_idx``, ``task`` and ``trial_idx``.
    """
    lengths = trials.groupby("ses_idx").agg(task=("task", "first"), n=("trial_idx", "max")).reset_index()
    chosen = {}
    for task in TASKS:
        sessions = lengths[lengths["task"] == task].copy()
        sessions["gap"] = (sessions["n"] - sessions["n"].median()).abs()
        chosen[task] = str(sessions.sort_values(["gap", "ses_idx"]).iloc[0]["ses_idx"])
    return chosen


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))
