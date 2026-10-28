"""Shared helpers for Study 51 producers: paths, loading the 3-way trial table, the two readouts.

Producers import this with ``sys.path.insert(0, <analysis dir>)``. Mirrors Study 50's
``_pk01.py``; Study 50's own table and figures are read from its folder for side-by-side
comparison, restricted to the sessions both eval splits share (``split_alignment.json``).

The 3-way head's logits are re-expressed as two readouts (PLAN.md):
  * ``logit_diff``   = logit_R − logit_L = log P(R | engaged) / P(L | engaged);
  * ``engage_logit`` = logsumexp(logit_L, logit_R) − logit_ignore = log P(engaged) / P(ignore).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ANALYSIS = STUDY / "analysis"
CACHE = ANALYSIS / "_cache"
EXPLORE = CACHE / "explore"
STUDY12 = STUDY.parent / "50-gru-dynamics-embedding"
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402,F401

MODEL_KEY = "C3-s0"
TWIN_KEY = "C-s0"  # Study 50's 2-way model on the same config
HIDDEN = 128
STATE_COLUMNS = [f"rnn_state_{i}" for i in range(HIDDEN)]
N_PCS_SAVED = 20
IGNORE = 2
CAP = 20

TASKS = ["Uncoupled Baiting", "Coupled Baiting", "Uncoupled Without Baiting", "Coupled Without Baiting"]
TASK_COLORS = {
    "Uncoupled Baiting": "#4C72B0",
    "Coupled Baiting": "#C44E52",
    "Uncoupled Without Baiting": "#55A868",
    "Coupled Without Baiting": "#8172B3",
}
LEFT_COLOR, RIGHT_COLOR, MODEL_COLOR = "#D55E00", "#0072B2", "#222222"
IGNORE_COLOR = "#7F7F7F"
ENGAGE_COLOR = "#8E44AD"  # model P(ignore) traces


# What this process read, for _meta v2: committed files (``inputs``) and cache files
# (``sources`` of a freeze), by path; filled by read_json() and the cache loaders below.
_INPUTS: dict[str, Path] = {}
_CACHE_READS: dict[str, dict] = {}
_TWIN_READ = False


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


def _run_source(study: Path, key: str) -> dict:
    manifest = json.loads((study / "analysis" / "source_runs.json").read_text())
    run = manifest["models"][key.split("-")[0]]["runs"][key]
    return {"type": "wandb_run", "id": f"{manifest['entity']}/{manifest['project']}/{run['run_id']}"}


def model_sources() -> list[dict]:
    """The source run(s) every A1-derived output is computed from: C3, plus C when its table was read."""
    return [_run_source(STUDY, MODEL_KEY)] + ([_run_source(STUDY12, TWIN_KEY)] if _TWIN_READ else [])


def wandb_groups() -> list[str]:
    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    return [manifest["models"][MODEL_KEY.split("-")[0]]["group"]]


def twin_wandb_groups() -> list[str]:
    manifest = json.loads((STUDY12 / "analysis" / "source_runs.json").read_text())
    return [manifest["models"][TWIN_KEY.split("-")[0]]["group"]]


def trial_table_path(key: str = MODEL_KEY) -> Path:
    return CACHE / f"trial_states_{key}.parquet"


def twin_table_path() -> Path:
    return STUDY12 / "analysis" / "_cache" / f"trial_states_{TWIN_KEY}.parquet"


def add_readouts(trials: pd.DataFrame) -> pd.DataFrame:
    """``engage_logit`` and ``p_right_engaged`` from the logits / probabilities present."""
    if {"choice_logit_0", "choice_logit_1", "choice_logit_2"} <= set(trials.columns):
        lr = np.logaddexp(trials["choice_logit_0"].to_numpy(np.float64), trials["choice_logit_1"].to_numpy(np.float64))
        trials["engage_logit"] = lr - trials["choice_logit_2"].to_numpy(np.float64)
    if {"p_model1_left", "p_model1_right", "p_model1_ignore"} <= set(trials.columns):
        trials["p_right_engaged"] = trials["p_model1_right"] / (trials["p_model1_left"] + trials["p_model1_right"])
    return trials


def load_trials(
    columns: list[str] | None = None, key: str = MODEL_KEY, sessions: list[str] | None = None
) -> pd.DataFrame:
    """The A1 table (all timesteps, ignored trials as ``action == 2``), in stored order.

    ``sessions`` restricts to those ``ses_idx`` values at read time. Fills ``session_id`` on
    all-ignore sessions (empty in tables extracted before the wrapper fix) and adds the two
    readouts when their inputs are among ``columns``.
    """
    filters = [("ses_idx", "in", list(sessions))] if sessions is not None else None
    wanted = None if columns is None else list(dict.fromkeys(
        [*columns, *(["source_ses_idx"] if "session_id" in columns else [])]))
    import pyarrow.parquet as pq  # noqa: PLC0415 (freeze-only; CI's offline `all` has no pyarrow)

    trials = pq.read_table(_read_cache(trial_table_path(key)), columns=wanted, filters=filters).to_pandas()
    if "session_id" in trials.columns and "source_ses_idx" in trials.columns:
        trials["session_id"] = trials["session_id"].fillna(trials["source_ses_idx"])
    return add_readouts(trials)


def load_twin(columns: list[str], sessions: list[str] | None = None) -> pd.DataFrame:
    """Study 50's C-s0 table (2-way, engaged trials only), optionally restricted to ``ses_idx``."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    global _TWIN_READ
    _TWIN_READ = True
    filters = [("ses_idx", "in", list(sessions))] if sessions is not None else None
    return pq.read_table(_read_cache(twin_table_path()), columns=columns, filters=filters).to_pandas()


def unshared_sessions() -> dict[str, set[str]]:
    """``session_id``s to drop from each table so C and C3 cover the same sessions
    (``split_alignment.json``; the shared sessions hold identical engaged trials)."""
    alignment = read_json(ANALYSIS / "split_alignment.json")
    return {
        "C": {s for v in alignment["sessions_only_in_C"].values() for s in v},
        "C3": {s for v in alignment["sessions_only_in_C3"].values() for s in v}
        | set(alignment["C3_all_ignore_eval_sessions"]),
    }


RL_MODELS = ("bari", "ctt", "hattori")


def engaged_comparison_table(key: str = MODEL_KEY) -> pd.DataFrame:
    """One row per engaged trial of the shared sessions, with every model's P(right).

    Columns: subject_id, ses_idx, trial (original numbering), task, action (0/1), reward,
    ``gap`` (ignored trials since the previous engaged trial of the session, or since its
    start), ``p_C3`` (P(right | engaged)), ``p_C`` (Study 50's C), ``p_<rl>`` for Study 01's
    per-mouse RL fits (replayed on C's trials by Study 50's ``c2_rl_rollouts.py``).
    The shared sessions hold identical engaged trials (``split_alignment.json``), checked here.
    ``key`` picks the C3 seed; ``split_alignment.json`` was computed for seed 0, and the join
    below fails unless the seed's eval split holds the same engaged trials.
    """
    drop = unshared_sessions()
    c3 = load_trials(["subject_id", "ses_idx", "session_id", "trial", "task", "action", "reward",
                      "p_model1_left", "p_model1_right", "p_model1_ignore"], key=key)
    c3 = c3[~c3["session_id"].isin(drop["C3"])]
    # Ignores since the previous engaged trial = cumulative ignores now minus at that trial.
    cum = (c3["action"] == IGNORE).astype(int).groupby(c3["ses_idx"], sort=False).cumsum()
    engaged = c3[c3["action"] != IGNORE].rename(columns={"p_right_engaged": "p_C3"}).copy()
    cum_e = cum[engaged.index]
    engaged["gap"] = (cum_e - cum_e.groupby(engaged["ses_idx"], sort=False).shift(1)).fillna(cum_e).astype(int)
    c = load_twin(["ses_idx", "session_id", "trial_idx", "trial", "action", "p_model1_right"])
    c = c[~c["session_id"].isin(drop["C"])]
    import pyarrow.parquet as pq  # noqa: PLC0415

    for model in RL_MODELS:
        rl = pq.read_table(_read_cache(STUDY12 / "analysis" / "_cache" / f"rl_trials_{model}_{TWIN_KEY}.parquet"),
                           columns=["ses_idx", "trial_idx", "p_rl_right"]).to_pandas()
        c = c.merge(rl.rename(columns={"p_rl_right": f"p_{model}"}), on=["ses_idx", "trial_idx"],
                    how="left", validate="1:1")
    out = engaged[["subject_id", "ses_idx", "trial", "task", "action", "reward", "gap", "p_C3"]].merge(
        c.rename(columns={"p_model1_right": "p_C", "action": "action_C"}).drop(columns=["session_id", "trial_idx"]),
        on=["ses_idx", "trial"], how="inner", validate="1:1")
    if len(out) != len(engaged) or len(out) != len(c) or (out["action"] != out["action_C"]).any():
        raise AssertionError(f"Shared engaged trials do not line up: C3 {len(engaged)}, C {len(c)}, joined {len(out)}")
    return out.drop(columns="action_C").reset_index(drop=True)


def pca_path(key: str = MODEL_KEY) -> Path:
    return CACHE / f"pca_{key}.npz"


def load_pca(key: str = MODEL_KEY) -> dict[str, np.ndarray]:
    """PCA written by ``b3_pca.py``: mean, components (PCs x units), eigenvalues, scores."""
    with np.load(_read_cache(pca_path(key))) as data:
        return {k: data[k] for k in data.files}


def write_json(name: str, payload: dict, produced_by: str, extra_groups: list[str] = (), *,
               kind: str = "freeze", inputs: list[Path | str] = (), sources: list[dict] = ()) -> Path:
    """Write ``analysis/<name>`` with a v2 ``_meta``.

    Every producer behind a ``freeze-*`` target is a freeze (the default): its sources are the
    model's source run(s) plus every cache file read through the loaders here, and ``sources``
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
    """Exploration-only output, gitignored under ``_cache/``."""
    EXPLORE.mkdir(parents=True, exist_ok=True)
    return EXPLORE / name


def subsample_rows(n_rows: int, n: int, seed: int = 0) -> np.ndarray:
    """Sorted random row indices, for scatter plots of a 4.8M-trial table."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(n, n_rows), replace=False))


def best_params(key: str = MODEL_KEY) -> dict:
    """Saved params of the best_eval checkpoint (highest eval likelihood in checkpoints/index.json)."""
    outputs = CACHE / "models" / key / "run" / "outputs"
    index = json.loads(_read_cache(outputs / "checkpoints" / "index.json").read_text())
    best = max(index["checkpoints"], key=lambda c: c["eval_likelihood"])
    return json.loads(_read_cache(outputs / "checkpoints" / f"step_{best['step']}" / "params.json").read_text())


def readout_directions(key: str = MODEL_KEY) -> dict[str, np.ndarray]:
    """The two readout directions in state space (unit-free; not normalised).

    ``side`` = w_R − w_L: logit_diff = side·h + const exactly. ``engage`` = (w_L + w_R)/2 − w_ignore:
    the linear part of engage_logit (the log-sum-exp over L/R makes the readout itself non-linear).
    """
    readout = next(v for k, v in best_params(key).items() if k.endswith("/readout"))
    w = np.asarray(readout["w"], dtype=np.float64)
    return {"side": w[:, 1] - w[:, 0], "engage": (w[:, 0] + w[:, 1]) / 2 - w[:, IGNORE]}


def example_sessions() -> dict[str, str]:
    """Study 50's B1 example sessions (one per task); all four are in both eval splits."""
    shown = json.loads((STUDY12 / "analysis" / "b1_example_sessions.json").read_text())["sessions"]
    return {task: shown[task]["ses_idx"] for task in TASKS}


def ignore_runs(action: np.ndarray, ses: np.ndarray) -> pd.DataFrame:
    """Maximal runs of consecutive ignored trials within sessions (rows in stored order).

    One row per run: ``start`` / ``stop`` row positions (stop exclusive), ``length``, and
    whether the run touches the session start or end (then its true length is unknown).
    """
    ignored = action == IGNORE
    new_session = np.r_[True, ses[1:] != ses[:-1]]
    ends_session = np.r_[ses[1:] != ses[:-1], True]
    starts = np.flatnonzero(ignored & (new_session | ~np.r_[False, ignored[:-1]]))
    stops = np.flatnonzero(ignored & (ends_session | ~np.r_[ignored[1:], False])) + 1
    return pd.DataFrame({
        "start": starts, "stop": stops, "length": stops - starts,
        "at_session_start": new_session[starts], "at_session_end": ends_session[stops - 1],
    })


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))
