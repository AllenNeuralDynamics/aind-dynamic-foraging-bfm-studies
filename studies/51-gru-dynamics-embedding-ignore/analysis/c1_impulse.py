"""C1a (stage A): what one input does inside C3 — counterfactual edits of a single trial.

    python analysis/c1_impulse.py

Re-runs C3 with one trial's input edited and every other input as observed (wrapper
``gru_counterfactual.input_edit_effects``), recording Δ of both readouts on the next K steps
(steps include ignored trials). Edits, N_EDITS each:
  * ``reward_flip``: an engaged trial's reward flipped. Side effect reported as Study 50 does,
    c·(logit if rewarded − logit if unrewarded), c = +1 right / −1 left, so it sits next to
    C's cached flips (Study 50 ``_cache/c1_impulse_flips_C-s0.parquet``); engage effect as
    (rewarded − unrewarded).
  * ``to_ignore``: an engaged trial replaced by an ignore (choice 2, reward 0). Side effect
    signed by the original choice (negative = less preference for the side just chosen);
    engage effect as is (negative = toward disengaging). Split by the original reward.
  * ``to_engaged``: an ignored trial replaced by an unrewarded choice of a random side. Side
    effect signed by the inserted side; engage effect as is.
Checks first that the unedited replay reproduces the A1 readouts and that ignored trials
enter the network as (choice 2, reward 0). Full float32 matmuls: on GPU the default TF32
makes edited and baseline runs differ by ~5e-4 even without an edit.
Writes ``c1_impulse.json`` and ``_cache/c1_impulse_edits_C3-s0.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

N_EDITS = 20_000
K = 10
SEED = 0
REPLAY_TOLERANCE_MEDIAN = 5e-3


def summarise(effect: np.ndarray) -> dict:
    n = np.sum(np.isfinite(effect), axis=0)
    return {"mean": np.nanmean(effect, axis=0), "sem": np.nanstd(effect, axis=0, ddof=1) / np.sqrt(n), "n": n,
            "median": np.nanmedian(effect, axis=0), "frac_negative": np.nanmean(effect < 0, axis=0)}


def main() -> None:
    from post_training_analysis.gru_counterfactual import input_edit_effects, load_gru_split, logits, readouts  # noqa: PLC0415

    split = load_gru_split(s13.CACHE / "models" / s13.MODEL_KEY / "run")
    if split.n_actions != 3:
        raise AssertionError(f"Expected a 3-action split, got {split.n_actions}")
    trials = s13.load_trials(["subject_id", "ses_idx", "trial_idx", "task", "action", "reward", "logit_diff",
                              "choice_logit_0", "choice_logit_1", "choice_logit_2"])
    col = trials["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    if np.isnan(col.astype(float)).any():
        raise AssertionError("Some A1 sessions are missing from the split")
    tau = trials["trial_idx"].to_numpy() - 1
    action = trials["action"].to_numpy()

    baseline = readouts(logits(split))
    err = {name: np.abs(baseline[name][tau, col] - trials[column].to_numpy(float))
           for name, column in (("side", "logit_diff"), ("engage", "engage_logit"))}
    for name, e in err.items():
        if np.median(e) > REPLAY_TOLERANCE_MEDIAN:
            raise AssertionError(f"Unedited replay does not match A1 {name} (median err {np.median(e)})")
    # Row τ+1 carries trial τ's outcome: an ignored trial must enter as (2, 0).
    has_next = tau + 1 < split.n_trials[col]
    ign = (action == s13.IGNORE) & has_next
    enc = split.xs[tau[ign] + 1, col[ign]][:, [split.choice_col, split.reward_col]]
    if not (np.all(enc[:, 0] == 2) and np.all(enc[:, 1] == 0)):
        raise AssertionError("Ignored trials do not enter the network as (choice 2, reward 0)")

    rng = np.random.default_rng(SEED)
    eligible = tau + K < split.n_trials[col]
    engaged_rows = np.sort(rng.choice(np.flatnonzero(eligible & (action != s13.IGNORE)), N_EDITS, replace=False))
    ignored_rows = np.sort(rng.choice(np.flatnonzero(eligible & (action == s13.IGNORE)), N_EDITS, replace=False))
    reward = trials["reward"].to_numpy()

    def run(rows: np.ndarray, choice: np.ndarray, new_reward: np.ndarray) -> dict[str, np.ndarray]:
        return input_edit_effects(split, col[rows], tau[rows], choice=choice, reward=new_reward, k_max=K,
                                  baseline=baseline)

    out, frames = {}, []
    # Reward flip on engaged trials.
    rows = engaged_rows
    rewarded = reward[rows] > 0
    d = run(rows, action[rows].astype(np.float32), (~rewarded).astype(np.float32))
    sign = np.where(action[rows] == 1, 1.0, -1.0)
    rw = np.where(rewarded[:, None], -1.0, 1.0)  # rewarded − unrewarded
    flip = {"side": sign[:, None] * rw * d["side"], "engage": rw * d["engage"]}
    out["reward_flip"] = {name: summarise(v) for name, v in flip.items()}
    frames.append(("reward_flip", rows, flip))

    # Engaged trial → ignore.
    d = run(rows, np.full(len(rows), 2.0, np.float32), np.zeros(len(rows), np.float32))
    to_ignore = {"side": sign[:, None] * d["side"], "engage": d["engage"]}
    out["to_ignore"] = {name: summarise(v) for name, v in to_ignore.items()}
    out["to_ignore_by_original_reward"] = {
        label: {name: summarise(v[mask]) for name, v in to_ignore.items()}
        for label, mask in (("was_rewarded", rewarded), ("was_unrewarded", ~rewarded))}
    frames.append(("to_ignore", rows, to_ignore))

    # Ignored trial → unrewarded choice of a random side.
    rows = ignored_rows
    side = rng.integers(0, 2, size=len(rows))
    d = run(rows, side.astype(np.float32), np.zeros(len(rows), np.float32))
    inserted_sign = np.where(side == 1, 1.0, -1.0)
    to_engaged = {"side": inserted_sign[:, None] * d["side"], "engage": d["engage"]}
    out["to_engaged"] = {name: summarise(v) for name, v in to_engaged.items()}
    frames.append(("to_engaged", rows, to_engaged))

    parts = []
    for edit, rows, effects in frames:
        f = trials.iloc[rows][["subject_id", "ses_idx", "trial_idx", "task", "action", "reward"]].reset_index(drop=True)
        f.insert(0, "edit", edit)
        for name, v in effects.items():
            f = pd.concat([f, pd.DataFrame(v, columns=[f"{name}_k{k}" for k in range(1, K + 1)])], axis=1)
        parts.append(f)
    pd.concat(parts, ignore_index=True).to_parquet(s13.CACHE / f"c1_impulse_edits_{s13.MODEL_KEY}.parquet", index=False)

    s13.write_json("c1_impulse.json", {
        "n_edits_each": N_EDITS, "k": np.arange(1, K + 1), "seed": SEED, "steps": "GRU timesteps (ignored trials count)",
        "matmul_precision": "highest",
        "replay_check": {name: {"median_abs_err": float(np.median(e)), "max_abs_err": float(np.max(e))} for name, e in err.items()},
        "ignore_encoding_check": {"n_ignored_with_next_row": int(ign.sum()), "choice": 2, "reward": 0},
        "definitions": {
            "reward_flip": "side: c·(rewarded − unrewarded); engage: rewarded − unrewarded",
            "to_ignore": "engaged trial → (2, 0); side signed by the original choice; engage as is",
            "to_engaged": "ignored trial → unrewarded random side; side signed by the inserted side; engage as is"},
        **out,
    }, "analysis/c1_impulse.py")
    for edit in ("reward_flip", "to_ignore", "to_engaged"):
        print(edit, {n: np.round(out[edit][n]["mean"][:5], 3).tolist() for n in ("side", "engage")})


if __name__ == "__main__":
    main()
