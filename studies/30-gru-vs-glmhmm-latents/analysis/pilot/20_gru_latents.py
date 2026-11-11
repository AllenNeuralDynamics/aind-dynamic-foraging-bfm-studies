"""Stage 2: GRU hidden states and logits on the pilot mice's eval sessions.

Trained: study-02 H256/D614 seed 0 (nxd-3way-20260707-075401-80a6d102), params.json, which
is byte-identical to checkpoints/step_90000 (= final = best eval; the run early-stopped at
90000 and its eval likelihood rose at every checkpoint). Session-curriculum lambda = 1.0
(90000 >= 30000 pretrain + 20000 warmup; output_summary.session_curriculum.final_lambda).

Untrained: the same run's initialization/before_training/params.json (the wrapper's init),
run with lambda = 0.0, as the wrapper's before-training evaluation does at step 0.

Row t's latent is h_t = GRU(x_t, h_{t-1}), where x_t holds choice[t-1] and reward[t-1]
(checked in stage 1). hk.dynamic_unroll(return_all_states=True) returns h_t, and the logits
at row t are readout(h_t): the latent at row t is exactly the state that predicts trial t.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
GRU_DIR = PILOT / "wandb_dl/gru_D614_s0/outputs"
sys.path.insert(0, str(Path(os.environ["WRAPPER_ROOT"]) / "code"))

from disentangled_rnns.library import rnn_utils  # noqa: E402
from evaluation.common import _load_saved_params  # noqa: E402
from evaluation.gru_evaluation import add_gru_model_results  # noqa: E402
from models.gru_network import make_gru_network  # noqa: E402

CHUNK = 128


def forward(make_net, params, xs):
    ys_, ss_ = [], []
    for start in range(0, xs.shape[1], CHUNK):
        y, s = rnn_utils.eval_network(make_net, params, xs[:, start:start + CHUNK, :])
        ys_.append(np.asarray(y, dtype=np.float32))
        ss_.append(np.asarray(s, dtype=np.float32))
    return np.concatenate(ys_, axis=1), np.concatenate(ss_, axis=1)


def main():
    cfg = json.loads((GRU_DIR / "gru_config.json").read_text())
    arch = cfg["architecture"]
    meta = json.loads((GRU_DIR / "multisubject_metadata.json").read_text())
    make_network = make_gru_network(
        hidden_size=int(arch["hidden_size"]), output_size=int(cfg["output_size"]),
        multisubject=True, max_n_subjects=int(meta["num_subjects"]),
        subject_embedding_size=int(arch["subject_embedding_size"]),
        subject_embedding_init=str(arch["subject_embedding_init"]),
        session_encoding_type=str(arch["session_encoding_type"]),
        session_integration_type=str(arch["session_integration_type"]),
        session_fourier_k=int(arch["session_fourier_k"]),
        session_delta_n_layers=int(arch["session_delta_n_layers"]),
        session_delta_hidden_size=int(arch["session_delta_hidden_size"]),
        session_max_index_by_subject_index=[int(v) for v in meta["session_max_index_by_subject_index"]],
    )
    d = np.load(OUT / "pilot_eval_xs_ys.npz")
    xs, ys = d["xs"].astype(np.float64), d["ys"]
    ev = pd.read_parquet(OUT / "pilot_eval_raw.parquet")
    pm = json.loads((OUT / "pilot_meta.json").read_text())
    order = pm["eval_session_order"]
    assert list(dict.fromkeys(ev["ses_idx"].tolist())) == order

    results = {}
    for name, path, lam in [("trained", GRU_DIR / "params.json", 1.0),
                            ("untrained", GRU_DIR / "initialization/before_training/params.json", 0.0)]:
        params = _load_saved_params(path)
        yhat, states = forward(lambda: make_network(lam), params, xs)
        lik = float(rnn_utils.normalized_likelihood(ys, yhat[:, :, :3]))
        # Per-session likelihoods for a few sessions (geometric mean of p(observed choice)).
        lp = yhat[:, :, :3] - np.log(np.exp(yhat[:, :, :3] - yhat[:, :, :3].max(-1, keepdims=True)).sum(-1, keepdims=True)) - yhat[:, :, :3].max(-1, keepdims=True)
        per_session = []
        for j in range(min(5, ys.shape[1])):
            m = ys[:, j, 0] >= 0
            per_session.append(float(np.exp(np.mean(lp[m, j, ys[m, j, 0]]))))
        print(name, "normalized likelihood", lik, "first sessions", per_session, flush=True)
        out = add_gru_model_results(ev[["subject_id", "ses_idx", "trial"]].copy(), states, yhat,
                                    ignore_policy="include")
        assert len(out) == len(ev)
        assert (out[["ses_idx", "trial"]].to_numpy() == ev[["ses_idx", "trial"]].to_numpy()).all()
        if name == "untrained":
            out = out[["subject_id", "ses_idx", "trial"] + [c for c in out.columns if c.startswith("latent_")]]
        float_cols = [c for c in out.columns if c.startswith(("latent_", "choice_"))]
        out[float_cols] = out[float_cols].astype(np.float32)
        out.to_parquet(OUT / f"gru_{name}_outputs.parquet")
        results[name] = {"params_path": str(path), "session_curriculum_lambda": lam,
                         "normalized_likelihood": lik, "first_session_likelihoods": per_session}
    (OUT / "gru_forward_summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
