"""Held-out stage 2: GRU latents on the 149 held-out mice (variant heldout-mice).

Rebuilds the fine-tune's held-out bundle with the wrapper's own functions, from the launch's
heldout.yaml and source model_dir, so the inputs are exactly the trials the fine-tune used. Then
runs two parameter sets over every full session, as the fine-tune's checkpoint evaluation does
(sessions are independent columns, so a session's latents do not depend on the split):

* trained: the fine-tune's params.json (network frozen, 149 fine-tuned embedding rows);
* untrained: the source run's initialization params, expanded with mean-initialized rows for the
  149 mice as the fine-tune does at step 0, run with session-curriculum lambda 0 (r1's control).

Check before anything is written: the trained eval likelihood must equal the fine-tune's
output_summary eval_likelihood to 1e-6. Writes per-trial latents and logits for the eval (odd)
sessions to <out_root>/analysis/. Row t's latent is the state that predicts trial t (r1, stage 2).

    WRAPPER_ROOT=<wrapper checkout> python 20_gru_latents.py <out_root>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_ROOT = Path(sys.argv[1])
sys.path.insert(0, str(Path(os.environ["WRAPPER_ROOT"]) / "code"))

from disentangled_rnns.library import rnn_utils  # noqa: E402
from models.gru_network import make_gru_network  # noqa: E402
from models.session_conditioning import resolve_session_conditioning_from_architecture  # noqa: E402
from post_training_analysis.heldout_finetuning import (  # noqa: E402
    _build_global_heldout_bundle, _load_config, _load_params, _make_gru_runtime,
    _normalize_mapping, _resolve_runtime_config, resolve_model_run)
from evaluation.gru_evaluation import add_gru_model_results  # noqa: E402
from utils.multisubject import expand_local_multisubject_params  # noqa: E402

CHUNK = 128   # sessions per forward pass


def forward(make_net, params, xs):
    ys_, ss_ = [], []
    for start in range(0, xs.shape[1], CHUNK):
        y, s = rnn_utils.eval_network(make_net, params, xs[:, start:start + CHUNK, :])
        ys_.append(np.asarray(y, dtype=np.float32))
        ss_.append(np.asarray(s, dtype=np.float32))
    return np.concatenate(ys_, axis=1), np.concatenate(ss_, axis=1)


def main():
    (summary_path,) = (OUT_ROOT / "heldout").rglob("outputs/output_summary.json")
    ft_outputs = summary_path.parent
    logged_eval = json.loads(summary_path.read_text())["metrics"]["eval_likelihood"]
    config = _load_config(str(OUT_ROOT / "heldout.yaml"))
    source_cfg = _normalize_mapping(config.get("source_run"))
    source_run = resolve_model_run(source_cfg["model_dir"], split="train",
                                   checkpoint_policy=str(source_cfg["checkpoint_policy"]))
    resolved = _resolve_runtime_config(config, source_run=source_run, output_root_override=None)
    work = OUT_ROOT / "analysis"
    work.mkdir(exist_ok=True)
    _, architecture, _, model_config, _ = _make_gru_runtime(source_run=source_run, output_dir=work / "runtime")
    bundle, heldout_ids, appended, _, _ = _build_global_heldout_bundle(
        source_run=source_run, heldout_selector=_normalize_mapping(resolved["heldout_subjects"]),
        fine_tune_cfg=resolved["heldout_finetuning"], architecture=architecture)
    meta = bundle.metadata
    session_cfg = resolve_session_conditioning_from_architecture(
        architecture=architecture, metadata=meta, multisubject=True,
        max_n_subjects=int(meta["num_subjects"]),
        subject_embedding_size=int(architecture["subject_embedding_size"]),
        context="study 30 held-out latents")
    make_network = make_gru_network(
        hidden_size=int(architecture["hidden_size"]),
        output_size=int(_normalize_mapping(model_config).get("output_size", 3)),
        multisubject=True, max_n_subjects=int(meta["num_subjects"]),
        subject_embedding_size=int(architecture["subject_embedding_size"]),
        subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
        session_encoding_type=str(session_cfg["session_encoding_type"]),
        session_integration_type=str(session_cfg["session_integration_type"]),
        session_fourier_k=int(session_cfg["session_fourier_k"]),
        session_delta_n_layers=int(session_cfg["session_delta_n_layers"]),
        session_delta_hidden_size=int(session_cfg["session_delta_hidden_size"]),
        session_max_index_by_subject_index=list(meta["session_max_index_by_subject_index"]))

    init_path = Path(source_cfg["model_dir"]) / "outputs" / "initialization" / "before_training" / "params.json"
    param_sets = {
        "trained": (_load_params(ft_outputs / "params.json"), 1.0),
        "untrained": (expand_local_multisubject_params(_load_params(init_path), n_new_subjects=len(appended),
                                                       init="mean"), 0.0),
    }
    ev = bundle.eval_set.get_all()
    eval_sessions = {str(s) for s in meta["eval_session_ids"]}
    xs_full = np.asarray(bundle.extras["dataset"].get_all()["xs"])
    summary = {"n_heldout_mice": len(heldout_ids), "n_eval_sessions": len(eval_sessions),
               "logged_eval_likelihood": logged_eval}
    for name, (params, lam) in param_sets.items():
        yhat_ev, _ = forward(lambda: make_network(lam), params, np.asarray(ev["xs"]))
        lik = float(rnn_utils.normalized_likelihood(np.asarray(ev["ys"]), yhat_ev[:, :, :3]))
        summary[f"{name}_eval_likelihood"] = lik
        print(name, "eval likelihood", lik, flush=True)
        if name == "trained" and abs(lik - logged_eval) > 1e-6:
            sys.exit(f"trained eval likelihood {lik} != fine-tune's {logged_eval}: the rebuilt inputs differ")
        yhat, states = forward(lambda: make_network(lam), params, xs_full)
        out = add_gru_model_results(bundle.raw.copy(), states, yhat, ignore_policy="include")
        out = out[out["ses_idx"].astype(str).isin(eval_sessions)]
        keep = ["subject_id", "ses_idx", "trial"] + [c for c in out.columns if c.startswith("latent_")]
        if name == "trained":
            keep += [c for c in out.columns if c.startswith("choice_")]
        out = out[keep].reset_index(drop=True)
        floats = [c for c in out.columns if c.startswith(("latent_", "choice_"))]
        out[floats] = out[floats].astype(np.float32)
        out.to_parquet(work / f"gru_{name}_latents.parquet")
        summary[f"{name}_rows"] = len(out)
    (work / "gru_latents_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
