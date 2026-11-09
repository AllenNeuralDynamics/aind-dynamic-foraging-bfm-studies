"""Shuffle nulls for the K-scan decoding: same decoder, folds and features; only the target pairing is broken.

Usage: 99_shuffle_null.py <feature_set> <null> <K> <seed[,seed...]>
  feature_set: latents | history
  null:
    swap  each session's target is replaced by the target of a different session from a different mouse in the
          same fold (keeps both signals' within-session structure, breaks only the pairing); partner sessions at
          least as long are cut to length; if none is that long, the longest is tiled
    circ  each session's target is rotated in time by a random 20-80 % of its length (same session and mouse)
Decoder trained AND scored on the shuffled pairs, with the K-scan probe (kscan_probes.fit_soft_logistic, L2 1e-4)
and folds (fold_of_mouse). Writes out/shuffle/parts/<fset>_<null>_K<K>_s<seed>.json with the K-scan scores.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT, KOUT = PILOT / "out", PILOT / "out" / "kscan"
PARTS = OUT / "shuffle" / "parts"
spec = importlib.util.spec_from_file_location("kp", PILOT / "scripts/kscan_probes.py")
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)


def shuffled_target(P, sessions_idx, ses_mouse, ses_fold, null, rng):
    Q = np.empty_like(P)
    names = list(sessions_idx)
    lengths = {s: len(sessions_idx[s]) for s in names}
    for s in names:
        idx = sessions_idx[s]
        n = len(idx)
        if null == "circ":
            off = int(rng.integers(int(0.2 * n), int(0.8 * n) + 1))
            Q[idx] = np.roll(P[idx], off, axis=0)
            continue
        pool = [q for q in names if ses_fold[q] == ses_fold[s] and ses_mouse[q] != ses_mouse[s]]
        long_enough = [q for q in pool if lengths[q] >= n]
        if long_enough:
            src = P[sessions_idx[long_enough[rng.integers(len(long_enough))]]][:n]
        else:
            q = max(pool, key=lambda x: lengths[x])
            src = np.resize(P[sessions_idx[q]], (n, P.shape[1]))
        Q[idx] = src
    return Q


def main():
    fset, null, K, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(s) for s in sys.argv[4].split(",")]
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (PARTS / f"{fset}_{null}_K{K}_s{s}.json").exists()]
    if not todo:
        return
    j = pd.read_parquet(OUT / "joined_trials.parquet")
    keys = j[["ses_idx", "trial"]].to_numpy()
    if fset == "latents":
        dense, tis = j[[c for c in j.columns if c.startswith("latent_")]].to_numpy(np.float64), None
    else:
        lags, emas = kp.pilot_probes.history_features(j)
        dense, tis = np.hstack([lags, emas]), j["trial_in_session"].to_numpy(np.float64)[:, None]
    groups, sessions = j["subject_id"].astype(str).to_numpy(), j["ses_idx"].astype(str).to_numpy()
    del j
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    fold = np.array([fom[g] for g in groups])
    folds = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(5)]
    sessions_idx = pd.Series(np.arange(len(sessions))).groupby(sessions, sort=False).apply(np.asarray).to_dict()
    ses_mouse = {s: groups[i[0]] for s, i in sessions_idx.items()}
    ses_fold = {s: fold[i[0]] for s, i in sessions_idx.items()}
    Xs = []
    for tr, te in folds:
        sc = StandardScaler().fit(dense[tr])
        a, b = [sc.transform(dense[tr])], [sc.transform(dense[te])]
        if tis is not None:
            sp = SplineTransformer(n_knots=8, degree=3, knots="quantile").fit(tis[tr])
            s2 = StandardScaler().fit(sp.transform(tis[tr]))
            a.append(s2.transform(sp.transform(tis[tr])))
            b.append(s2.transform(sp.transform(tis[te])))
        Xs.append((np.hstack(a), np.hstack(b)))
    g = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet")
    assert (g[["ses_idx", "trial"]].to_numpy() == keys).all()
    P = g[[f"pred_{k}" for k in range(K)]].to_numpy()
    for seed in todo:
        t0 = time.time()
        rng = np.random.default_rng(1000 * K + seed + (0 if null == "swap" else 500))
        Pn = shuffled_target(P, sessions_idx, ses_mouse, ses_fold, null, rng)
        Q, C, iters = np.zeros_like(Pn), np.zeros_like(Pn), []
        for (tr, te), (Xtr, Xte) in zip(folds, Xs):
            predict, nit = kp.fit_soft_logistic(Xtr, Pn[tr])
            Q[te], C[te] = predict(Xte), Pn[tr].mean(0)
            iters.append(nit)
        s = kp.scores(Pn, Q, C, sessions)
        s.update({"feature_set": fset, "null": null, "seed": seed, "logistic_iters": iters,
                  "seconds": round(time.time() - t0, 1)})
        (PARTS / f"{fset}_{null}_K{K}_s{seed}.json").write_text(json.dumps(s, indent=1))
        print(fset, null, K, seed, {k: round(s[k], 3) for k in ("pooled_r2", "pooled_r2_within", "info_recovered")},
              iters, s["seconds"], flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
