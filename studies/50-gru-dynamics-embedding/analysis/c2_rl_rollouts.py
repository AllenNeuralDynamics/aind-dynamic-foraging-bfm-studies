"""C2 (stage A): per-trial P(right) and Q values of the fitted RL baselines on C's trial table.

    python analysis/c2_rl_rollouts.py [bari ctt hattori]

For each pinned baseline in ``source_runs.json["rl_baselines"]``: download its output
artifact (refusing unless ``:latest`` is the pinned digest), replay every mouse's fitted
agent over the same eval sessions as the A1 table (wrapper ``baseline_rl_trial_rollout``),
and check the replay against the run's own logged eval likelihoods, per mouse and pooled.
Writes ``_cache/rl_trials_<model>_<key>.parquet`` (row-aligned to the A1 table; gitignored)
and ``c2_rl_rollouts.json`` (checks; committed).
"""

from __future__ import annotations

import json
import netrc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

WORKERS = 24
TOLERANCE = 1e-6


def _fetch(model: dict, project: str, entity: str) -> dict:
    target = s12.CACHE / "models" / "rl" / model["run_id"]
    path = target / "baseline_rl_output.json"
    if not path.exists():
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")[2]
        import wandb  # noqa: PLC0415

        artifact = wandb.Api().artifact(f"{entity}/{project}/{model['artifact_name']}:latest")
        if artifact.digest != model["artifact_digest"]:
            raise AssertionError(f"{model['artifact_name']} is {artifact.digest}, pinned {model['artifact_digest']}")
        artifact.download(root=str(target))
    return json.loads(path.read_text())


def main(names: list[str]) -> None:
    from post_training_analysis.baseline_rl_trial_rollout import pooled_likelihood, rollout_baseline_rl  # noqa: PLC0415

    manifest = json.loads((s12.STUDY / "analysis" / "source_runs.json").read_text())
    baselines = manifest["rl_baselines"]
    trials = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward"])
    summary_path = s12.ANALYSIS / "c2_rl_rollouts.json"
    previous = json.loads(summary_path.read_text()).get("models", {}) if summary_path.exists() else {}
    results = dict(previous)
    for name in names or list(baselines["models"]):
        model = baselines["models"][name]
        output = _fetch(model, baselines["project"], manifest["entity"])
        rl = rollout_baseline_rl(output, trials, workers=WORKERS)

        per_subject = []
        for subject, rows in trials.groupby("subject_id", sort=False).indices.items():
            ours = pooled_likelihood(trials["action"].to_numpy()[rows], rl["p_rl_right"].to_numpy()[rows])
            per_subject.append(abs(ours - output["fitted_params_per_subject"][subject]["eval_likelihood"]))
        pooled = pooled_likelihood(trials["action"].to_numpy(), rl["p_rl_right"].to_numpy())
        checks = {"pooled_eval_likelihood": pooled, "logged_pooled_eval_likelihood": output["pooled_eval_trial_likelihood"],
                  "pooled_abs_diff": abs(pooled - output["pooled_eval_trial_likelihood"]),
                  "max_subject_abs_diff": float(np.max(per_subject)), "n_subjects": len(per_subject)}
        kwargs = output["agent_kwargs"]
        if "q_rl_right" in rl.columns and kwargs.get("choice_kernel") == "none" and kwargs.get("action_selection") == "softmax":
            # Q values must reproduce the policy: P(right) = σ(β (Q_R − Q_L) − biasL).
            fits = output["fitted_params_per_subject"]
            subject = trials["subject_id"].astype(str)
            beta = subject.map({k: v["fitted_params"]["softmax_inverse_temperature"] for k, v in fits.items()})
            bias = subject.map({k: v["fitted_params"]["biasL"] for k, v in fits.items()})
            policy = 1 / (1 + np.exp(-(beta.to_numpy() * (rl["q_rl_right"] - rl["q_rl_left"]).to_numpy() - bias.to_numpy())))
            checks["q_policy_max_abs_diff"] = float(np.max(np.abs(policy - rl["p_rl_right"].to_numpy())))
            if checks["q_policy_max_abs_diff"] > TOLERANCE:
                raise AssertionError(f"{name}: Q values do not reproduce the policy: {checks}")
        if checks["pooled_abs_diff"] > TOLERANCE or checks["max_subject_abs_diff"] > TOLERANCE:
            raise AssertionError(f"{name}: replay does not reproduce the logged likelihoods: {checks}")
        rl.to_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet", index=False)
        results[name] = {"run_id": model["run_id"], "group": model["group"], "agent_class": output["agent_class"],
                         "agent_kwargs": output["agent_kwargs"], "parameter_columns": output["parameter_columns"],
                         "q_values": "q_rl_left" in rl.columns, "checks": checks}
        print(name, json.dumps(checks))

    groups = s12.wandb_groups() + [m["group"] for k, m in baselines["models"].items() if k in results]
    out = {"_meta": s12.build_meta("analysis/c2_rl_rollouts.py", groups, study_root=s12.STUDY), "models": results}
    summary_path.write_text(json.dumps(out, indent=2, default=s12._json_default) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
