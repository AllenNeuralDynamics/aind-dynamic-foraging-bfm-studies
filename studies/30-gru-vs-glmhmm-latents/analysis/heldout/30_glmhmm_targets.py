"""Held-out stage 3: GLM-HMM predictive state posteriors for the 149 held-out mice.

Study 14 run u4j7ukti (hurdle, K = 3, AIND-disRNN/mice_glmhmm) refits each held-out mouse (MAP,
study 14 protocol) and saves its filtered posteriors alpha_t = p(z_t | trials <= t) per trial in
heldout_trials.parquet. r1's pilot (stage 3, step 1) recomputed them from a held-out mouse's
refit parameters and matched the saved columns, so the saved filtered_k include trial t.

The decoding target is the one-step predictive posterior, as in r1:
    pi_t = alpha_{t-1} A_mouse  (pi_0 = the mouse's initial_probs at session start).
Each mouse's refit states are put in the global state order by emission weights
(``match_by_weights``, study 14's method): target column j is the mouse's state permutation[j].
Every mouse gets its best permutation; whether it kept the global order (identity) and its
matched weight distance are recorded per mouse, for the r2 decision on badly matched mice.

Writes <out_dir>/glmhmm_heldout_targets.parquet (eval sessions) and glmhmm_heldout_alignment.json.

    python 30_glmhmm_targets.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aind_dynamic_foraging_models.glm_hmm import match_by_weights

GLM_DIR = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb/run-20261027_213114-u4j7ukti/files/outputs/glmhmm")
OUT = Path(sys.argv[1])


def hurdle_weights(emissions):
    return np.concatenate([np.asarray(emissions["side_weights"]), np.asarray(emissions["engage_weights"])], 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fit = json.loads((GLM_DIR / "fit.json").read_text())
    K = int(fit["config"]["num_states"])
    assert fit["config"]["emission"] == "hurdle" and K == 3, fit["config"]
    g = hurdle_weights(fit["global"]["emissions"])
    df = pd.read_parquet(GLM_DIR / "heldout_trials.parquet")
    df = df[df["split"] == "eval"].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["ses_idx"] = df["ses_idx"].astype(str)
    df = df.sort_values(["subject_id", "ses_idx", "trial"], kind="stable").reset_index(drop=True)
    filt = df[[f"filtered_{k}" for k in range(K)]].to_numpy(np.float64)

    pred = np.empty_like(filt)
    alignment = {}
    for sid, idx in df.groupby("subject_id", sort=False).indices.items():
        p = fit["subjects"][sid]
        A, pi0 = np.asarray(p["transition_matrix"]), np.asarray(p["initial_probs"])
        perm, cost = match_by_weights(hurdle_weights(p["emissions"]), g)
        alignment[sid] = {"permutation": perm.tolist(), "identity": bool((perm == np.arange(K)).all()),
                          "rms_weight_distance": float(np.sqrt(cost[perm, np.arange(K)].sum() / g.size))}
        ses = df["ses_idx"].to_numpy()[idx]
        first = np.r_[True, ses[1:] != ses[:-1]]
        prev = np.vstack([pi0[None, :], filt[idx[:-1]] @ A])   # alpha_{t-1} A, row by row
        prev[first] = pi0
        pred[idx] = prev[:, perm]
    assert np.allclose(pred.sum(1), 1, atol=1e-5), np.abs(pred.sum(1) - 1).max()

    out = df[["subject_id", "ses_idx", "trial", "choice", "logp_left", "logp_right", "logp_ignore"]].copy()
    for j in range(K):
        out[f"pred_{j}"] = pred[:, j].astype(np.float32)
    out.to_parquet(OUT / "glmhmm_heldout_targets.parquet")
    summary = {"source": str(GLM_DIR), "wandb_run": "AIND-disRNN/mice_glmhmm/u4j7ukti",
               "n_mice": len(alignment), "n_eval_trials": int(len(out)),
               "n_eval_sessions": int(out.groupby(["subject_id", "ses_idx"]).ngroups),
               "n_identity": int(sum(a["identity"] for a in alignment.values())),
               "rms_weight_distance_median": float(np.median([a["rms_weight_distance"] for a in alignment.values()])),
               "per_mouse": alignment}
    (OUT / "glmhmm_heldout_alignment.json").write_text(json.dumps(summary, indent=1) + "\n")
    print({k: v for k, v in summary.items() if k != "per_mouse"})


if __name__ == "__main__":
    main()
