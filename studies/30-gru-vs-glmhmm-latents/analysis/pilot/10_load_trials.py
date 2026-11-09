"""Stage 1: load raw trials for the pilot training mice and one held-out mouse.

Scratch pilot (study 30, not scaffolded). Writes only under the pilot's out/.

* Picks ~50 training mice from the GRU run's 614, stratified by the run's own
  ``subject_curricula`` labels, seed 30.
* Loads their trials through the wrapper's loader with the same data config both runs used
  (mice_snapshot_scaling, ignore_policy=include, seed 0), but with ``subject_ids`` set.
  ``subject_ids`` bypasses only the subject *selection* (curricula affect ranking, not which
  sessions of a selected mouse are fetched), so each mouse's session list should be identical
  to the full-pool load. This is checked exactly against the GRU run's session_context_map.
* Builds the GRU eval inputs exactly as GruTrainer does: merged eval split with the subject
  index (remapped to the GRU run's 614-mouse index) and the 1-based session index prepended.
* Loads one held-out mouse for the Step-1 filtered-convention check.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
WRAPPER = Path(os.environ.get("WRAPPER_ROOT", "/home/han.hou/code/wt/wrapper-study30-pilot"))
DISPATCHER = Path(os.environ.get("DISPATCHER_ROOT", "/home/han.hou/code/wt/dispatcher-study30-pilot"))
GRU_DIR = PILOT / "wandb_dl/gru_D614_s0/outputs"
GLM_DIR = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb/run-20261027_213114-u4j7ukti/files/outputs/glmhmm")
sys.path.insert(0, str(WRAPPER / "code"))

DATA_OVERRIDES = [
    "data=mice_snapshot_scaling",
    "data.ignore_policy=include",
    "seed=0",
    "+data.cols_to_retain=[trial,subject_id,ses_idx,animal_response,earned_reward,"
    "current_stage_actual,reward_probabilityL,reward_probabilityR]",
    "model=glmhmm",
]
# Per-curriculum label (the GRU run's subject_curricula) -> number of pilot mice.
QUOTA = {"Uncoupled Baiting": 14, "Uncoupled Without Baiting": 12, "Mixed": 12,
         "None": 8, "Coupled Baiting": 4}
PICK_SEED = 30


def load(subject_ids):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=str(DISPATCHER / "code" / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=DATA_OVERRIDES)
    return instantiate(cfg.data, subject_ids=list(subject_ids), multisubject=True).load()


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((GRU_DIR / "multisubject_metadata.json").read_text())
    gru_index = {str(k): int(v) for k, v in meta["subject_id_to_index"].items()}
    ctx = {str(r["subject_id"]): r for r in json.loads((GRU_DIR / "session_context_map.json").read_text())["per_subject"]}
    curricula = {str(k): v for k, v in meta["subject_curricula"].items()}
    glm_fit = json.loads((GLM_DIR / "fit.json").read_text())
    heldout_mice = set(glm_fit["subjects"])  # GLM-HMM's held-out cohort; must not overlap
    assert not heldout_mice & set(gru_index), "held-out and training cohorts overlap"

    rng = np.random.default_rng(PICK_SEED)
    pilot = []
    for label, n in QUOTA.items():
        pool = sorted(s for s, c in curricula.items() if c == label)
        pilot += sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    print(f"pilot mice: {len(pilot)}", flush=True)

    bundle = load(pilot)
    md = bundle.metadata
    got = {str(r["subject_id"]): r for r in md["session_context"]["per_subject"]}
    assert set(got) == set(pilot), (set(pilot) - set(got), set(got) - set(pilot))
    for s in pilot:  # exact session lists = same session-index inputs as in training
        assert [str(x) for x in got[s]["ordered_session_ids"]] == [str(x) for x in ctx[s]["ordered_session_ids"]], s
        assert int(md["session_max_index_by_subject_index"][int(md["subject_id_to_index"][s])]) == \
            int(meta["session_max_index_by_subject_index"][gru_index[s]]), s
    gru_eval = [str(x) for x in meta["eval_session_ids"]]
    pilot_set = set(pilot)
    gru_eval_pilot = [x for x in gru_eval if x.split("__")[0] in pilot_set]
    my_eval = [str(x) for x in md["eval_session_ids"]]
    assert sorted(my_eval) == sorted(gru_eval_pilot), "eval session sets differ from the GRU run"
    for s in pilot:  # eval = positions 2, 4, 6, ... (1-based) of each mouse's list
        ids = [str(x) for x in got[s]["ordered_session_ids"]]
        assert [x for x in my_eval if x.split("__")[0] == s] == ids[1::2], s
    print("session lists and eval split match the GRU run exactly", flush=True)

    # GRU eval inputs, as GruTrainer builds them.
    from utils.multisubject import (prepend_session_index_to_multisubject_dataset,
                                    session_indices_for_split)
    eval_ds = prepend_session_index_to_multisubject_dataset(
        bundle.eval_set, session_indices=session_indices_for_split(md, split_name="eval"))
    xs = np.array(eval_ds.get_all()["xs"], dtype=np.float64)
    ys = np.array(eval_ds.get_all()["ys"])
    print("xs", xs.shape, "x_names", list(eval_ds.x_names), flush=True)
    fresh_to_sid = {int(k): str(v) for k, v in md["index_to_subject_id"].items()}
    pad = xs[..., 0] < 0
    remapped = np.vectorize(lambda i: gru_index[fresh_to_sid[int(i)]] if i >= 0 else -1)(xs[..., 0])
    xs[..., 0] = np.where(pad, -1, remapped)

    raw = bundle.raw
    raw = raw.assign(subject_id=raw["subject_id"].astype(str), ses_idx=raw["ses_idx"].astype(str))
    ev = raw[raw["ses_idx"].isin(set(my_eval))].copy()
    ev["ses_idx"] = pd.Categorical(ev["ses_idx"], categories=my_eval, ordered=True)
    ev = ev.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
    ev["ses_idx"] = ev["ses_idx"].astype(str)
    # Alignment checks: ys row t = choice t; prev-choice input row t = choice t-1.
    for j, sid in enumerate(my_eval):
        g = ev[ev["ses_idx"] == sid]
        n = len(g)
        assert (ys[:n, j, 0] == g["animal_response"].to_numpy()).all(), sid
        # Padding: ys = -1 and observations = -1 after the last trial. (Row n keeps the
        # subject/session index; it follows every real trial, so it cannot change them.)
        if not ((ys[n:, j, 0] < 0).all() and (xs[n:, j, 2:] < 0).all()):
            print("pad check", sid, n, ys.shape[0], "ys pad uniq", np.unique(ys[n:, j, 0])[:5],
                  "xs pad uniq per feature", [np.unique(xs[n:, j, f])[:5].tolist() for f in range(xs.shape[2])], flush=True)
            raise AssertionError(sid)
        assert (xs[1:n, j, 2] == g["animal_response"].to_numpy()[:-1]).all(), sid
        assert int(xs[0, j, 1]) == [str(x) for x in got[g["subject_id"].iloc[0]]["ordered_session_ids"]].index(sid) + 1
    print("ys = choice[t], prev-choice input = choice[t-1] on every eval session", flush=True)

    ev.to_parquet(OUT / "pilot_eval_raw.parquet")
    np.savez_compressed(OUT / "pilot_eval_xs_ys.npz", xs=xs.astype(np.float32), ys=ys.astype(np.int32))
    (OUT / "pilot_meta.json").write_text(json.dumps({
        "pilot_mice": pilot, "quota": QUOTA, "pick_seed": PICK_SEED,
        "curricula": {s: curricula[s] for s in pilot}, "eval_session_order": my_eval,
        "x_names": [str(x) for x in eval_ds.x_names], "data_overrides": DATA_OVERRIDES,
        "n_eval_sessions": len(my_eval), "n_eval_trials": int(len(ev)),
        "n_train_sessions": len(md["train_session_ids"]),
    }, indent=1))
    print("pilot eval:", len(my_eval), "sessions", len(ev), "trials", flush=True)

    # One held-out mouse for Step 1 (subject_ids path; checked against the trainer's parquet).
    pq = pd.read_parquet(GLM_DIR / "heldout_trials.parquet",
                         columns=["subject_id", "ses_idx", "trial", "split", "choice"])
    pq["subject_id"] = pq["subject_id"].astype(str)
    counts = pq[pq["split"] == "eval"].groupby("subject_id").size()
    ho = str(counts[(counts > 3000) & (counts < 8000)].index.sort_values()[0])
    hb = load([ho])
    hraw = hb.raw.assign(subject_id=hb.raw["subject_id"].astype(str), ses_idx=hb.raw["ses_idx"].astype(str))
    hraw.to_parquet(OUT / "step1_heldout_mouse_raw.parquet")
    (OUT / "step1_heldout_mouse_meta.json").write_text(json.dumps({
        "subject_id": ho, "train_session_ids": [str(x) for x in hb.metadata["train_session_ids"]],
        "eval_session_ids": [str(x) for x in hb.metadata["eval_session_ids"]]}, indent=1))
    print("held-out mouse", ho, len(hraw), "trials", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
