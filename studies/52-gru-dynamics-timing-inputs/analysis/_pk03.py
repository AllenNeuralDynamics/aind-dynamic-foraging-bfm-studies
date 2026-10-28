"""Shared helpers for Study 52 producers: paths, the same-trial R/C table, binning, style."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STUDY = Path(__file__).resolve().parents[1]
ANALYSIS = STUDY / "analysis"
CACHE = ANALYSIS / "_cache"
STUDY12 = STUDY.parent / "50-gru-dynamics-embedding"
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402,F401

MODEL_KEY = "R-s0"
BASELINE_KEY = "C-s0"
N_BINS = 10
N_BOOT = 200
SEED_KEY = "R-s1"
# The wrapper's encoder clips RT to this range before the log (``trial_timing_features.RT_CLIP_S``);
# a handful of raw RTs are exactly 0.
RT_CLIP_S = (1e-3, 10.0)
# Colours (Study 50's side-collapsed convention); unrewarded is also dashed.
REWARDED_COLOR, UNREWARDED_COLOR = "#0072B2", "#C44E52"
ANIMAL_COLOR = "#222222"
MODEL_LABELS = {"mice": "Mice", "R": "RT model (R)", "C": "Choice/reward model (C)"}


def wandb_groups() -> list[str]:
    manifest = json.loads((STUDY / "analysis" / "source_runs.json").read_text())
    return [manifest["models"][MODEL_KEY.split("-")[0]]["group"]]


def write_json(name: str, payload: dict, produced_by: str) -> Path:
    path = ANALYSIS / name
    out = {"_meta": build_meta(produced_by, wandb_groups(), study_root=STUDY), **payload}
    path.write_text(json.dumps(out, indent=2, default=_json_default) + "\n")
    return path


def figure_path(name: str) -> Path:
    return ANALYSIS / name


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def load_paired_trials(rt_present_only: bool = True, with_seed: bool = False, key: str = MODEL_KEY,
                       seed_key: str = SEED_KEY, licks_present_only: bool = False) -> pd.DataFrame:
    """R-s0 and C-s0 on the same eval trials, with per-trial history columns.

    Rows are engaged trials in session order. Columns: ``p_right`` / ``p_right_C`` (each
    model's P(right) for trial t), ``log_p`` / ``log_p_C`` (log P(chosen)), ``dlog`` (R − C),
    ``rt`` (trial t's RT, s), ``rt_lag{k}`` / ``reward_lag{k}`` / ``action_lag{k}`` for
    k = 1..3 (trial t−k in the same session), ``switch`` (action differs from t−1) and
    model P(switch). ``rt_present_only`` keeps sessions where every trial has an RT.
    ``rt`` is clipped to ``RT_CLIP_S`` as the model's encoder does. ``with_seed`` adds
    ``log_p_seed``, log P(chosen) under ``seed_key`` (a second seed of the same model).
    ``key`` selects the model compared with C (default R-s0). For a lick model the raw counts
    ``lick_left_count`` / ``lick_right_count`` are kept, plus ``licks_chosen`` / ``licks_other``
    (relative to that trial's choice) and their lags; ``licks_present_only`` drops sessions
    with no licks recorded at all.
    """
    cols = ["ses_idx", "trial_idx", "subject_id", "task", "action", "reward", "reaction_time",
            "p_model1_right", "logit_diff"]
    available = set(pq.read_schema(CACHE / f"trial_states_{key}.parquet").names)
    licks = {"lick_left_count", "lick_right_count"} <= available
    cols += [c for c in ("prev_log_rt_input", "lick_left_count", "lick_right_count") if c in available]
    r = pd.read_parquet(CACHE / f"trial_states_{key}.parquet", columns=cols)
    c = pd.read_parquet(STUDY12 / "analysis" / "_cache" / f"trial_states_{BASELINE_KEY}.parquet",
                        columns=["ses_idx", "trial_idx", "p_model1_right", "logit_diff"])
    df = r.merge(c, on=["ses_idx", "trial_idx"], suffixes=("", "_C"), validate="1:1")
    if len(df) != len(r):
        raise AssertionError("R and C tables do not share every trial (run a2 first)")
    df = df.rename(columns={"p_model1_right": "p_right", "p_model1_right_C": "p_right_C",
                            "reaction_time": "rt", "logit_diff": "logit", "logit_diff_C": "logit_C"})
    df["rt"] = df["rt"].clip(*RT_CLIP_S)
    if with_seed:
        seed = pd.read_parquet(CACHE / f"trial_states_{seed_key}.parquet",
                               columns=["ses_idx", "trial_idx", "p_model1_right"])
        df = df.merge(seed.rename(columns={"p_model1_right": "p_right_seed"}), on=["ses_idx", "trial_idx"], validate="1:1")
        if len(df) != len(r):
            raise AssertionError(f"{seed_key} does not share every trial")
    df = df.sort_values(["ses_idx", "trial_idx"], kind="stable").reset_index(drop=True)
    if rt_present_only:
        complete = df["rt"].notna().groupby(df["ses_idx"]).transform("all")
        df = df[complete].reset_index(drop=True)
    if licks:
        right_now = df["action"].to_numpy() == 1
        df["licks_chosen"] = np.where(right_now, df["lick_right_count"], df["lick_left_count"])
        df["licks_other"] = np.where(right_now, df["lick_left_count"], df["lick_right_count"])
        if licks_present_only:
            any_lick = (df["licks_chosen"] + df["licks_other"]).groupby(df["ses_idx"]).transform("sum") > 0
            df = df[any_lick].reset_index(drop=True)
    right = df["action"].to_numpy() == 1
    df["log_p"] = np.log(np.where(right, df["p_right"], 1 - df["p_right"]))
    df["log_p_C"] = np.log(np.where(right, df["p_right_C"], 1 - df["p_right_C"]))
    df["dlog"] = df["log_p"] - df["log_p_C"]
    if with_seed:
        df["log_p_seed"] = np.log(np.where(right, df["p_right_seed"], 1 - df["p_right_seed"]))
    grouped = df.groupby("ses_idx", sort=False)
    for k in (1, 2, 3):
        for col in ("rt", "reward", "action", *(("licks_chosen", "licks_other") if licks else ())):
            df[f"{col}_lag{k}"] = grouped[col].shift(k)
    has_prev = df["action_lag1"].notna()
    prev_right = df["action_lag1"].to_numpy() == 1
    df["switch"] = np.where(has_prev, (df["action"] != df["action_lag1"]).astype(float), np.nan)
    # P(switch) = P(the side not chosen on t−1).
    for suffix in ("", "_C", "_seed") if with_seed else ("", "_C"):
        p = df[f"p_right{suffix}"].to_numpy()
        df[f"p_switch{suffix}"] = np.where(has_prev, np.where(prev_right, 1 - p, p), np.nan)
    return df


def rt_bins(df: pd.DataFrame, column: str, *, within_mouse: bool, n_bins: int = N_BINS) -> pd.Series:
    """Decile index (0..n_bins−1) of ``column``: pooled, or ranked within each mouse."""
    x = np.log(df[column])
    if within_mouse:
        rank = x.groupby(df["subject_id"]).rank(pct=True, method="first")
        return np.minimum((rank * n_bins).fillna(-1).astype(int), n_bins - 1).where(x.notna())
    edges = np.nanquantile(x, np.linspace(0, 1, n_bins + 1))
    return pd.Series(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1), index=df.index).where(x.notna())


def binned_means(df: pd.DataFrame, by: list[str], values: list[str], seed: int = 0) -> pd.DataFrame:
    """Trial-weighted mean of each value per group, with a 95% CI from resampling mice.

    Per-mouse sums and counts are computed once, then mice are resampled with replacement
    ``N_BOOT`` times. Returns one row per group with ``<v>``, ``<v>_lo``, ``<v>_hi``, ``n``.
    """
    sub = df.dropna(subset=by)
    sums = sub.groupby(["subject_id", *by], observed=True)[values].sum()
    counts = sub.groupby(["subject_id", *by], observed=True).size().rename("n")
    table = pd.concat([sums, counts], axis=1).reset_index()
    mice = table["subject_id"].unique()
    mouse_index = pd.Index(mice)
    table["m"] = mouse_index.get_indexer(table["subject_id"])
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))
                        for _ in range(N_BOOT)])  # (N_BOOT, n_mice)
    out = []
    for key, g in table.groupby(by, observed=True):
        w = weights[:, g["m"].to_numpy()]  # (N_BOOT, rows)
        n = g["n"].to_numpy()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n"] = int(n.sum())
        for v in values:
            s = g[v].to_numpy()
            row[v] = s.sum() / n.sum()
            boot = (w @ s) / np.maximum(w @ n, 1)
            row[f"{v}_lo"], row[f"{v}_hi"] = np.quantile(boot, [0.025, 0.975])
        out.append(row)
    return pd.DataFrame(out)


def bin_centres_s(df: pd.DataFrame, column: str, bins: pd.Series) -> np.ndarray:
    """Median RT (s) of each pooled bin, for axis ticks."""
    return df.groupby(bins)[column].median().to_numpy()


RT_TICKS_S = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def rt_log_axis(ax, axis: str = "x") -> None:
    """Log RT axis with plain-second tick labels."""
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target.set_major_locator(FixedLocator(RT_TICKS_S))
    target.set_major_formatter(FormatStrFormatter("%g"))
    target.set_minor_locator(NullLocator())
