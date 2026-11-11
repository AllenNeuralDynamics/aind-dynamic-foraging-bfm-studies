"""K scan, GLM-HMM side: one-step predictive state posteriors for every K, global parameters.

Usage: kscan_glmhmm.py <emission>   (hurdle | reference)

Finds every finished run with fit.json under the run root whose fit.json["config"] matches
(emission, num_states, n_lags = 10), as study 14's find_runs() does, and keeps those in the
pinned W&B group (hurdle@20261027-212905 / reference@20261027-213006). For each K, on the
pilot's 378,624 eval trials (same order as out/joined_trials.parquet):
  pred_k = p(z_t | y_1..t-1) = filtered_{t-1} A (pi at t = 0), global parameters for every mouse;
  logp of the observed choice (GLM-HMM one-step predictive).
Checks: sum_k pred_k p(y | z = k, x_t) reproduces the predictive; K = 3 hurdle reproduces the
pilot's pred_k.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
KOUT = OUT / "kscan"
RUN_ROOT = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb")
GROUPS = {"hurdle": "hurdle@20261027-212905", "reference": "reference@20261027-213006"}
sys.path.insert(0, str(Path(os.environ["WRAPPER_ROOT"]) / "code"))

import jax  # noqa: E402
from aind_dynamic_foraging_models.glm_hmm import GLMHMM, make_buckets, predictive_log_probs, state_posteriors  # noqa: E402
from aind_dynamic_foraging_models.glm_hmm.design import build_inputs  # noqa: E402
from model_trainers.glmhmm_trainer import _sessions  # noqa: E402


def find_runs(emission):
    runs = {}
    for fit_path in sorted(RUN_ROOT.glob("run-2026102*/files/outputs/glmhmm/fit.json")):
        files = fit_path.parents[2]
        cfg = json.loads(fit_path.read_text())["config"]
        if cfg["emission"] != emission or int(cfg["n_lags"]) != 10:
            continue
        text = "".join((files / n).read_text() for n in ("config.yaml", "wandb-metadata.json", "run_hpc.log")
                       if (files / n).exists())
        if GROUPS[emission] not in text:
            continue
        K = int(cfg["num_states"])
        assert K not in runs, f"two runs for {emission} K={K}"
        runs[K] = {"run_id": files.parent.name.split("-")[-1], "fit": str(fit_path)}
    return runs


def main():
    emission = sys.argv[1]
    KOUT.mkdir(exist_ok=True)
    keys = pd.read_parquet(OUT / "joined_trials.parquet", columns=["subject_id", "ses_idx", "trial", "choice", "reward"])
    ev = keys.rename(columns={"choice": "animal_response", "earned_reward": "x"}).assign(earned_reward=keys["reward"])
    order = list(dict.fromkeys(ev["ses_idx"].tolist()))
    recs = _sessions(ev, order)  # one record per session, in joined-table order
    assert sum(len(r["choices"]) for r in recs) == len(ev)
    sess_codes = np.repeat(np.arange(len(recs)), [len(r["choices"]) for r in recs])
    runs = find_runs(emission)
    print(emission, {K: v["run_id"] for K, v in sorted(runs.items())}, flush=True)
    summary = {}
    for K in sorted(runs):
        if K < 2:
            continue
        fit = json.loads(Path(runs[K]["fit"]).read_text())
        cfg, g = fit["config"], fit["global"]
        model = GLMHMM(K, int(cfg["n_lags"]), cfg["emission"], cfg["num_newton_steps"])
        params, _ = model.initialize(initial_probs=np.asarray(g["initial_probs"]),
                                     transition_matrix=np.asarray(g["transition_matrix"]),
                                     **{k: np.asarray(v) for k, v in g["emissions"].items()})
        buckets = make_buckets([(r["choices"], r["rewards"]) for r in recs], int(cfg["n_lags"]),
                               cfg["design"], cfg["bucket_size"])
        lp = predictive_log_probs(model, params, buckets)
        post = state_posteriors(model, params, buckets)
        A, pi = np.asarray(g["transition_matrix"]), np.asarray(g["initial_probs"])
        preds, lpo, mix = [], [], 0.0
        from jax.experimental import enable_x64
        for i, r in enumerate(recs):
            f = post[i]["filtered"]
            pred = np.vstack([pi[None, :], f[:-1] @ A])
            x = build_inputs(r["choices"], r["rewards"], int(cfg["n_lags"]))
            with enable_x64():
                em = jax.tree_util.tree_map(lambda a: jax.numpy.asarray(a, dtype=jax.numpy.float64), params.emissions)
                cp = np.exp(np.asarray(model.emission_component.class_log_probs(em, x)))
            mix = max(mix, float(np.max(np.abs(np.einsum("tk,tkc->tc", pred, cp) - np.exp(lp[i])))))
            preds.append(pred)
            lpo.append(lp[i][np.arange(len(r["choices"])), r["choices"]])
        P = np.vstack(preds)
        lpo = np.concatenate(lpo)
        df = pd.DataFrame({"subject_id": keys["subject_id"], "ses_idx": keys["ses_idx"], "trial": keys["trial"],
                           "glm_logp_obs": lpo, **{f"pred_{k}": P[:, k] for k in range(K)}})
        df.to_parquet(KOUT / f"glmhmm_{emission}_K{K}.parquet")
        sm = pd.DataFrame(P).groupby(sess_codes).transform("mean").to_numpy()
        tot = ((P - P.mean(0)) ** 2).sum()
        within = ((P - sm) ** 2).sum()
        Pc = np.clip(P, 1e-12, 1)
        s = {"run_id": runs[K]["run_id"], "glmhmm_likelihood_pilot": float(np.exp(lpo.mean())),
             "logged_train_mice_eval_likelihood": fit["summary"].get("glmhmm/select/eval_likelihood"),
             "target_entropy": float(-(P * np.log(Pc)).sum(1).mean()),
             "within_session_share_pooled": float(within / tot),
             "occupancy": P.mean(0).tolist(), "max_mix_vs_predictive": mix}
        if emission == "hurdle" and K == 3:
            j = pd.read_parquet(OUT / "joined_trials.parquet", columns=["pred_0", "pred_1", "pred_2"]).to_numpy()
            s["maxabs_vs_pilot_pred"] = float(np.max(np.abs(j - P)))
        summary[K] = s
        print(emission, K, json.dumps(s), flush=True)
        jax.clear_caches()
    (KOUT / f"glmhmm_{emission}_summary.json").write_text(json.dumps(
        {"emission": emission, "group": GROUPS[emission], "runs": {K: v for K, v in runs.items()}, "per_K": summary}, indent=1))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
