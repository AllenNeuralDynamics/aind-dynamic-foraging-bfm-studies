"""C1a (stage A): the model's reward kernel by counterfactual — flip one trial's reward.

    python analysis/c1_impulse.py        # JAX_PLATFORMS=cpu works if no GPU is visible

For N_FLIPS random trials t (with K later trials in the session), re-run the GRU with trial
t's reward flipped and every other input as observed, and record Δlogit(right) on t+1…t+K.
Reported as the reward effect signed by the choice on t: c · (logit if rewarded − logit if
unrewarded), c = +1 right / −1 left — the quantity a logistic regression
estimates as β_RewC(k) − β_UnrC(k) (Su2022 design, C1b), so the two are directly comparable:
both hold the later history fixed.

Also re-checks that the counterfactual path's unedited forward pass reproduces the A1
logits. Writes ``c1_impulse.json`` and ``_cache/c1_impulse_flips_<key>.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_FLIPS = 40_000
K = 10
SEED = 0
# CPU float32 vs the A1 table's GPU TF32 run: the unedited replay differs by ~1e-3 (median),
# accumulated through the recurrence; a one-trial misalignment gives ~0.18.
REPLAY_TOLERANCE_MEDIAN = 5e-3


def summarise(effect: np.ndarray) -> dict:
    n = np.sum(np.isfinite(effect), axis=0)
    return {"mean": np.nanmean(effect, axis=0), "sem": np.nanstd(effect, axis=0, ddof=1) / np.sqrt(n), "n": n,
            "median": np.nanmedian(effect, axis=0), "frac_negative": np.nanmean(effect < 0, axis=0)}


def main() -> None:
    from post_training_analysis.gru_counterfactual import (  # noqa: PLC0415
        choice_logit_diff, load_gru_split, reward_flip_effects)

    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    trials = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "task", "action", "reward", "logit_diff"])
    col = trials["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = trials["trial_idx"].to_numpy() - 1

    baseline = choice_logit_diff(split)
    replay_err = np.abs(baseline[tau, col] - trials["logit_diff"].to_numpy())
    if np.median(replay_err) > REPLAY_TOLERANCE_MEDIAN:
        raise AssertionError(f"Unedited replay does not match the A1 logits (median err {np.median(replay_err)})")

    eligible = np.flatnonzero(tau + K < split.n_trials[col])
    rows = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_FLIPS, replace=False))
    delta = reward_flip_effects(split, col[rows], tau[rows], k_max=K, baseline=baseline)
    rewarded = trials["reward"].to_numpy()[rows] > 0
    sign = np.where(trials["action"].to_numpy()[rows] == 1, 1.0, -1.0)
    # rewarded − unrewarded logit: −Δ when the flip removed a reward, +Δ when it added one.
    effect = sign[:, None] * np.where(rewarded[:, None], -delta, delta)

    flips = trials.iloc[rows][["subject_id", "ses_idx", "trial_idx", "task", "action", "reward"]].reset_index(drop=True)
    flips = pd.concat([flips, pd.DataFrame(effect, columns=[f"effect_k{k}" for k in range(1, K + 1)])], axis=1)
    flips.to_parquet(s12.CACHE / f"c1_impulse_flips_{s12.MODEL_KEY}.parquet", index=False)

    by_task = {task: summarise(effect[(flips["task"] == task).to_numpy()]) for task in s12.TASKS}
    s12.write_json("c1_impulse.json", {
        "n_flips": N_FLIPS, "k": np.arange(1, K + 1), "seed": SEED,
        "definition": "c·(logit(right | t rewarded) − logit(right | t unrewarded)) on t+k, later inputs as observed",
        "replay_check": {"median_abs_err": float(np.median(replay_err)), "max_abs_err": float(np.max(replay_err))},
        "all": summarise(effect),
        "by_original_outcome": {"was_rewarded": summarise(effect[rewarded]), "was_unrewarded": summarise(effect[~rewarded])},
        "by_task": by_task,
    }, "analysis/c1_impulse.py")
    print("reward effect k=1..K:", np.round(np.nanmean(effect, axis=0), 3))


if __name__ == "__main__":
    main()
