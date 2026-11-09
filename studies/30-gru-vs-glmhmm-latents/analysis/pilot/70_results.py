"""Stage 7: assemble out/results.json (every number + provenance) and print the results table."""
import json
import subprocess
from importlib import metadata
from pathlib import Path

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
GRU_DIR = PILOT / "wandb_dl/gru_D614_s0"


def sha(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main():
    rd = lambda n: json.loads((OUT / n).read_text())  # noqa: E731
    pm, gs, gl, jc, pr = (rd("pilot_meta.json"), rd("gru_forward_summary.json"), rd("glmhmm_summary.json"),
                          rd("join_checks.json"), rd("probe_results.json"))
    run_info = json.loads((GRU_DIR / "run_info.json").read_text())
    gru_cfg = json.loads((GRU_DIR / "outputs/gru_config.json").read_text())
    ckpt = json.loads((GRU_DIR / "outputs/checkpoints/index.json").read_text())
    fit = json.loads(Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb/run-20261027_213114-u4j7ukti/"
                          "files/outputs/glmhmm/fit.json").read_text())
    res = {
        "_meta": {
            "kind": "scratch pilot (study 30 not scaffolded); not a study artifact",
            "produced_by": [str(p) for p in sorted((PILOT / "scripts").glob("*.py"))],
            "wrapper_worktree": "/home/han.hou/code/wt/wrapper-study30-pilot",
            "wrapper_git_sha": sha("/home/han.hou/code/wt/wrapper-study30-pilot"),
            "dispatcher_worktree": "/home/han.hou/code/wt/dispatcher-study30-pilot",
            "dispatcher_git_sha": sha("/home/han.hou/code/wt/dispatcher-study30-pilot"),
            "aind_dynamic_foraging_models_version": metadata.version("aind-dynamic-foraging-models"),
            "jax_version": metadata.version("jax"), "sklearn_version": metadata.version("scikit-learn"),
            "env": "/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu",
            "wandb_download_env": "/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search (wandb 0.31.0; "
                                  "0.23.1 in the shared env fails artifact queries with HTTP 400). Read-only.",
        },
        "gru": {
            "wandb": "AIND-disRNN/mice_ignore_scaling", "run_id": run_info["run_id"], "group": run_info["group"],
            "cell": "H256, D614 (all 614 training mice), seed 0, 3-way, ignore_policy=include, snapshot 20260603",
            "artifact": run_info["artifact"], "artifact_digest": run_info["artifact_digest"],
            "checkpoint": "params.json == checkpoints/step_90000/params.json (byte-identical); final = best eval "
                          "(early-stopped at 90000; eval likelihood rose at every checkpoint)",
            "checkpoint_eval_likelihoods": {r["step"]: r.get("eval_likelihood") for r in ckpt["checkpoints"]},
            "logged_train_mice_eval_likelihood": run_info["summary_subset"].get("checkpoint/eval_likelihood"),
            "logged_heldout_mice_finetune_eval_likelihood": run_info["summary_subset"].get("heldout/eval_likelihood"),
            "architecture": gru_cfg["architecture"], "output_size": gru_cfg["output_size"],
            "forward": gs,
            "sessions_held_out": "Yes: eval_every_n=2, so every 2nd session (positions 2, 4, ...) of each training "
                                 "mouse is never trained on. Same split as the GLM-HMM. But early stopping selected "
                                 "the checkpoint on eval_likelihood over these sessions, and the subject embedding "
                                 "and session-delta MLP were trained on the same mice's other sessions.",
        },
        "glmhmm": {
            "wandb": "AIND-disRNN/mice_glmhmm", "run_id": "u4j7ukti", "group": "hurdle@20261027-212905",
            "config": fit["config"], "wrapper_git_sha_of_fit": fit.get("wrapper_git_sha"),
            "params_used": "GLOBAL hurdle K=3 parameters for every pilot mouse (pilot simplification: one shared "
                           "set of state labels, no cross-mouse alignment)",
            "logged_train_mice_eval_likelihood_global": fit["summary"]["glmhmm/select/eval_likelihood"],
            "state_names": rd("state_names.json"), "state_description": rd("state_description.json"),
        },
        "step1_filtered_convention": gl["step1"],
        "pilot": {"mice": pm["pilot_mice"], "curricula": pm["curricula"], "quota": pm["quota"],
                  "pick_seed": pm["pick_seed"], "n_mice": jc["n_mice"], "n_eval_sessions": jc["n_sessions"],
                  "n_eval_trials": jc["n_trials"], "data_overrides": pm["data_overrides"]},
        "verification": {"join": jc, "glmhmm_pilot": gl["pilot"]},
        "probes": pr,
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    r = pr["results"]
    print("target|set  n_feat | LOGISTIC r2 S0 S1 S2 | within S0 S1 S2 | CE KL | RIDGE r2 S0 S1 S2 | within | CE")
    for key, v in r.items():
        lg, rg = v["logistic"], v["ridge_alr"]
        f = lambda a: " ".join(f"{x:5.2f}" for x in a)  # noqa: E731
        print(f"{key:28s} {v['n_features']:3d} | {f(lg['r2'])} | {f(lg['r2_within'])} | {lg['ce']:.3f} {lg['kl']:.3f} "
              f"| {f(rg['r2'])} | {f(rg['r2_within'])} | {rg['ce']:.3f}  bacc {lg['balanced_acc_argmax']:.2f} const {v['constant_baseline_ce']:.3f}")
    print("variance split", json.dumps(pr["variance_split"]))


if __name__ == "__main__":
    main()
