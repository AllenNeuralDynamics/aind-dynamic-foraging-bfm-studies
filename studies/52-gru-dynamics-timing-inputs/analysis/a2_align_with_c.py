"""A2: align a Study 52 trial table with Study 50's C-s0 on the same trials.

    python analysis/a2_align_with_c.py R-s0

Checks that both eval splits hold the same sessions and trials (same actions and rewards),
then breaks the pooled likelihood difference (key − C) down by RT availability. RT is
missing for whole sessions (older rigs), where the GRU's RT input is the constant 0; the
split shows whether any gain comes from that session-level flag rather than from RT.
Writes ``analysis/a2_alignment_<key>.json``. Reads both cached A1 tables (~3 GB RAM).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "analysis"))
from _pk03 import _read_cache, write_json  # noqa: E402

C_TABLE = STUDY.parent / "50-gru-dynamics-embedding" / "analysis" / "_cache" / "trial_states_C-s0.parquet"
KEYS = ["ses_idx", "trial_idx"]
PROBS = ["p_model1_left", "p_model1_right"]


def _log_p_chosen(df: pd.DataFrame, suffix: str = "") -> np.ndarray:
    return np.log(np.where(df["action"] == 1, df[f"p_model1_right{suffix}"], df[f"p_model1_left{suffix}"]))


def _likelihoods(df: pd.DataFrame, total_dlog: float) -> dict:
    model, base = np.exp(df["log_p"].mean()), np.exp(df["log_p_C"].mean())
    return {
        "n_trials": int(len(df)),
        "n_sessions": int(df["ses_idx"].nunique()),
        "n_subjects": int(df["subject_id"].nunique()),
        "likelihood": float(model),
        "likelihood_C": float(base),
        "diff": float(model - base),
        "share_of_total_log_gain": float((df["log_p"] - df["log_p_C"]).sum() / total_dlog),
    }


def main(key: str) -> None:
    table = STUDY / "analysis" / "_cache" / f"trial_states_{key}.parquet"
    model = pd.read_parquet(_read_cache(table), columns=[*KEYS, "subject_id", "action", "reward", "reaction_time", "task", *PROBS])
    base = pd.read_parquet(_read_cache(C_TABLE), columns=[*KEYS, "action", "reward", *PROBS])
    merged = model.merge(base, on=KEYS, how="outer", suffixes=("", "_C"), indicator=True, validate="1:1")
    alignment = {
        "rows": {"model": len(model), "C": len(base), "shared": int((merged["_merge"] == "both").sum())},
        "sessions": {"model": int(model["ses_idx"].nunique()), "C": int(base["ses_idx"].nunique())},
        "same_actions": bool((merged["action"] == merged["action_C"]).all()),
        "same_rewards": bool((merged["reward"] == merged["reward_C"]).all()),
    }
    if not (alignment["rows"]["shared"] == len(model) == len(base) and alignment["same_actions"] and alignment["same_rewards"]):
        raise AssertionError(f"{key} and C-s0 eval splits differ: {alignment}")

    merged["log_p"] = _log_p_chosen(merged)
    merged["log_p_C"] = _log_p_chosen(merged, "_C")
    total_dlog = float((merged["log_p"] - merged["log_p_C"]).sum())
    missing = merged["reaction_time"].isna()
    session_missing = missing.groupby(merged["ses_idx"]).mean()
    merged["rt_sessions"] = pd.cut(
        merged["ses_idx"].map(session_missing), [-np.inf, 0, 1 - 1e-12, np.inf],
        labels=["rt_present", "rt_partial", "rt_absent"],
    )
    subject_missing = missing.groupby(merged["subject_id"]).mean()
    summary = {
        "key": key,
        "alignment": alignment,
        "overall": _likelihoods(merged, total_dlog),
        "by_rt_availability": {
            str(name): _likelihoods(group, total_dlog)
            for name, group in merged.groupby("rt_sessions", observed=True)
        },
        "rt_missing": {
            "trials": int(missing.sum()),
            "sessions_all_missing": int((session_missing == 1).sum()),
            "sessions_partial": int(((session_missing > 0) & (session_missing < 1)).sum()),
            "subjects_all_missing": int((subject_missing == 1).sum()),
            "fraction_by_task": missing.groupby(merged["task"].fillna("<missing>")).mean().round(4).to_dict(),
        },
    }
    write_json(f"a2_alignment_{key}.json", summary, "analysis/a2_align_with_c.py", keys=[key, "C-s0"])
    print(json.dumps({k: summary[k] for k in ("alignment", "overall", "by_rt_availability", "rt_missing")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
