"""Stage 4: one row per pilot eval trial, GRU outputs joined to GLM-HMM posteriors.

Keys: subject_id and ses_idx cast to str, trial cast to int, on every side. Asserts zero
dropped rows. Writes out/joined_trials.parquet and out/untrained_latents.parquet (same row
order, keys included).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
KEYS = ["subject_id", "ses_idx", "trial"]


def keyed(df):
    return df.assign(subject_id=df["subject_id"].astype(str), ses_idx=df["ses_idx"].astype(str),
                     trial=df["trial"].astype(int))


def main():
    ev = keyed(pd.read_parquet(OUT / "pilot_eval_raw.parquet"))
    gru = keyed(pd.read_parquet(OUT / "gru_trained_outputs.parquet"))
    unt = keyed(pd.read_parquet(OUT / "gru_untrained_outputs.parquet"))
    glm = keyed(pd.read_parquet(OUT / "glmhmm_pilot_posteriors.parquet"))
    n = len(ev)
    assert len(gru) == len(unt) == len(glm) == n, (n, len(gru), len(unt), len(glm))

    base = ev.rename(columns={"animal_response": "choice", "earned_reward": "reward",
                              "current_stage_actual": "stage"})
    base["trial_in_session"] = base.groupby("ses_idx", sort=False).cumcount()
    j = base.merge(gru, on=KEYS, how="inner", validate="one_to_one")
    j = j.merge(glm, on=KEYS, how="inner", validate="one_to_one")
    assert len(j) == n, f"dropped {n - len(j)} rows in the join"
    assert (j["choice"].to_numpy() == j["glm_choice"].to_numpy()).all()
    assert (j["reward"].astype(int).to_numpy() == j["glm_reward"].to_numpy()).all()
    j = j.drop(columns=["glm_choice", "glm_reward"])
    # Keep the frame in the GRU eval order (session order, then trial).
    j = j.merge(base[KEYS].reset_index(), on=KEYS).sort_values("index").drop(columns="index").reset_index(drop=True)
    u = base[KEYS].merge(unt, on=KEYS, how="inner", validate="one_to_one")
    assert len(u) == n and (u[KEYS].to_numpy() == j[KEYS].to_numpy()).all()

    c = j["choice"].to_numpy().astype(int)
    p_gru = j[["choice_prob_0", "choice_prob_1", "choice_prob_2"]].to_numpy()[np.arange(n), c]
    lp_glm = j[["logp_left", "logp_right", "logp_ignore"]].to_numpy()[np.arange(n), c]
    pig_gru = j["choice_prob_2"].to_numpy()
    pig_glm = np.exp(j["logp_ignore"].to_numpy())
    chk = {
        "n_trials": int(n), "n_sessions": int(j["ses_idx"].nunique()), "n_mice": int(j["subject_id"].nunique()),
        "dropped_rows": 0,
        "gru_normalized_likelihood": float(np.exp(np.mean(np.log(p_gru)))),
        "glmhmm_normalized_likelihood": float(np.exp(np.mean(lp_glm))),
        "frac_trials_gru_beats_glmhmm": float(np.mean(np.log(p_gru) > lp_glm)),
        "corr_p_ignore_gru_vs_glmhmm": float(np.corrcoef(pig_gru, pig_glm)[0, 1]),
        "ignore_rate": float(np.mean(c == 2)),
        "mean_pred_occupancy": [float(j[f"pred_{k}"].mean()) for k in range(3)],
        "stage_counts": j["stage"].astype(str).value_counts().to_dict(),
    }
    print(json.dumps(chk, indent=1))
    j.to_parquet(OUT / "joined_trials.parquet")
    u.to_parquet(OUT / "untrained_latents.parquet")
    (OUT / "join_checks.json").write_text(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
