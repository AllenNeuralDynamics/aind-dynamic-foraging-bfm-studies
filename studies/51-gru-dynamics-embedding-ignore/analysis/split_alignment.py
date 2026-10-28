"""Which eval trials do C (Study 50, 2-way) and C3 (3-way) share? → ``split_alignment.json``.

    python analysis/split_alignment.py

Reads the two cached A1 tables (Study 50's ``trial_states_C-s0.parquet`` and this study's
``trial_states_C3-s0.parquet``, key columns only); no W&B, no model.

The eval split takes every other session per mouse (``eval_every_n=2``), counted over the
sessions that survive ``_align_raw_df_with_valid_sessions``. With ``ignore_policy=exclude``
that drops sessions in which every trial was ignored; with ``include`` it keeps them. A
mouse with such a session therefore has its later sessions' train/eval assignment shifted
by one between C and C3. This writes the session lists on each side and checks that the
shared sessions hold identical engaged trials, so C-vs-C3 comparisons can restrict to them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "analysis"))
from _pk02 import _read_cache, write_json  # noqa: E402
import _pk02  # noqa: E402

C_TABLE = STUDY.parent / "50-gru-dynamics-embedding" / "analysis" / "_cache" / "trial_states_C-s0.parquet"
C3_TABLE = STUDY / "analysis" / "_cache" / "trial_states_C3-s0.parquet"
COLUMNS = ["subject_id", "ses_idx", "session_id", "source_ses_idx", "trial", "action", "reward"]
IGNORE = 2


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(_read_cache(path), columns=COLUMNS)
    # Extractions before the wrapper fix leave session_id empty on all-ignore sessions.
    df["session_id"] = df["session_id"].fillna(df["source_ses_idx"])
    return df


def main() -> None:
    c = _load(C_TABLE)
    c3 = _load(C3_TABLE)
    engaged3 = c3[c3["action"] != IGNORE]

    per_session3 = c3.groupby("session_id").agg(
        subject_id=("subject_id", "first"), n=("action", "size"),
        n_engaged=("action", lambda a: int((a != IGNORE).sum())),
    )
    all_ignore = per_session3[per_session3["n_engaged"] == 0]
    sessions_c, sessions_c3 = set(c["session_id"]), set(engaged3["session_id"])
    shared = sorted(sessions_c & sessions_c3)

    keys = ["session_id", "trial", "action", "reward"]
    merged = c.loc[c["session_id"].isin(shared), keys].merge(
        engaged3.loc[engaged3["session_id"].isin(shared), keys],
        on=["session_id", "trial"], how="outer", suffixes=("_c", "_c3"), indicator=True,
    )
    matched = merged[merged["_merge"] == "both"]
    checks = {
        "shared_trials_only_in_C": int((merged["_merge"] == "left_only").sum()),
        "shared_trials_only_in_C3": int((merged["_merge"] == "right_only").sum()),
        "action_mismatches": int((matched["action_c"] != matched["action_c3"]).sum()),
        "reward_mismatches": int((matched["reward_c"] != matched["reward_c3"]).sum()),
    }
    if any(checks.values()):
        raise AssertionError(f"Shared sessions do not hold identical engaged trials: {checks}")

    subject_of = dict(zip(c["session_id"], c["subject_id"])) | dict(zip(c3["session_id"], c3["subject_id"]))

    def by_subject(sessions: set[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in sorted(sessions):
            out.setdefault(str(subject_of[s]), []).append(s)
        return out

    summary = {
        "inputs": {
            "C": str(C_TABLE.relative_to(STUDY.parent)),
            "C3": str(C3_TABLE.relative_to(STUDY.parent)),
        },
        "cause": (
            "eval_every_n=2 alternates over each mouse's sessions after "
            "_align_raw_df_with_valid_sessions, which drops all-ignore sessions only under "
            "ignore_policy=exclude; such a session shifts the mouse's later train/eval assignment."
        ),
        "counts": {
            "C_sessions": len(sessions_c),
            "C3_sessions_with_engaged_trials": len(sessions_c3),
            "C3_all_ignore_sessions": int(len(all_ignore)),
            "shared_sessions": len(shared),
            "shared_engaged_trials": int(len(matched)),
            "C_engaged_trials": int(len(c)),
            "C3_engaged_trials": int(len(engaged3)),
        },
        "checks_on_shared_sessions": checks,
        "subjects_affected": sorted({str(subject_of[s]) for s in sessions_c ^ sessions_c3}),
        "sessions_only_in_C": by_subject(sessions_c - sessions_c3),
        "sessions_only_in_C3": by_subject(sessions_c3 - sessions_c),
        "C3_all_ignore_eval_sessions": {
            s: {"subject_id": str(r.subject_id), "n_trials": int(r.n)} for s, r in all_ignore.iterrows()
        },
    }
    _pk02._TWIN_READ = True   # read C's table directly: C-s0 is a source too
    write_json("split_alignment.json", summary, "analysis/split_alignment.py")
    print(json.dumps({k: summary[k] for k in ("counts", "checks_on_shared_sessions", "subjects_affected")}, indent=2))


if __name__ == "__main__":
    main()
