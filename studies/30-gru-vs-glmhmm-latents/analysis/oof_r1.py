"""r1 out-of-fold predictions for every probe in r1's tables, so the freeze can bootstrap over mice.

The pilot saved per-trial out-of-fold (OOF) predictions only for latents and history (K = 3, and
K = 2, 4, 6, 10). This refits the same probes, on the same features and the same mouse folds
(``fold_of_mouse`` in out/probe_results.json), with the same fit (``kscan_probes.fit_soft_logistic``:
soft-label multinomial logistic, L2 1e-4, L-BFGS-B from zero), and saves each unit's OOF array.
Each unit asserts that its scores reproduce the pilot's stored value to 1e-3; the refit adds no
new result, only the per-trial predictions.

Units (hurdle emission, the K-scan targets out/kscan/glmhmm_hurdle_K<K>.parquet):
  latents | logits | history | untrained   at K = 2..10 (pilot K scan, kscan_probes.py)
  latents+history | trial_index           at K = 3 (pilot probe, 50_probes.py feature sets 4 and 6)

Writes out/oof_r1/hurdle_K<K>_<set>.npy (N x K float32, rows in joined_trials order). Compute node:

    run.sh analysis/oof_r1.py <set> <K[,K...]>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT, KOUT = PILOT / "out", PILOT / "out" / "kscan"
DST = OUT / "oof_r1"
TOL = 1e-3

spec = importlib.util.spec_from_file_location("kp", HERE / "pilot" / "kscan_probes.py")
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)
K3_ONLY = {"latents+history": "4_latents+history", "trial_index": "6_trial_index"}   # 50_probes set names


def features(j, fset):
    """(dense features, trial-index column or None) exactly as the pilot built them."""
    tis = j["trial_in_session"].to_numpy(np.float64)[:, None]
    lat = lambda df: df[[c for c in df.columns if c.startswith("latent_")]].to_numpy(np.float64)  # noqa: E731
    if fset == "latents":
        return lat(j), None
    if fset == "logits":
        return j[[f"choice_logit_{i}" for i in range(3)]].to_numpy(np.float64), None
    if fset == "untrained":
        u = pd.read_parquet(OUT / "untrained_latents.parquet")
        assert (u[["ses_idx", "trial"]].to_numpy() == j[["ses_idx", "trial"]].to_numpy()).all()
        return lat(u), None
    lags, emas = kp.pilot_probes.history_features(j)
    if fset == "history":
        return np.hstack([lags, emas]), tis
    if fset == "latents+history":
        return np.hstack([lat(j), lags, emas]), tis
    if fset == "trial_index":
        return np.zeros((len(j), 0)), tis
    raise ValueError(fset)


def fold_features(dense, tis, folds):
    """Training-fold scaler (and trial-index spline) per fold, as in the pilot."""
    Xs = []
    for tr, te in folds:
        a, b = [], []
        if dense.shape[1]:
            sc = StandardScaler().fit(dense[tr])
            a.append(sc.transform(dense[tr]))
            b.append(sc.transform(dense[te]))
        if tis is not None:
            sp = SplineTransformer(n_knots=8, degree=3, knots="quantile").fit(tis[tr])
            s2 = StandardScaler().fit(sp.transform(tis[tr]))
            a.append(s2.transform(sp.transform(tis[tr])))
            b.append(s2.transform(sp.transform(tis[te])))
        Xs.append((np.hstack(a), np.hstack(b)))
    return Xs


def reference(fset, K):
    """The pilot's stored scores for this unit: (name, value) pairs to reproduce."""
    if fset in K3_ONLY:
        ref = json.loads((OUT / "probe_parts" / f"pred__{K3_ONLY[fset]}.json").read_text())["logistic"]
        return [(f"r2_state{k}", v) for k, v in enumerate(ref["r2"])]
    ref = json.loads((KOUT / "parts" / f"hurdle__K{K}__{fset}.json").read_text())
    return [("pooled_r2", ref["pooled_r2"]), ("pooled_r2_within", ref["pooled_r2_within"])]


def main():
    fset, Ks = sys.argv[1], [int(k) for k in sys.argv[2].split(",")]
    assert fset not in K3_ONLY or Ks == [3], f"{fset} was fitted at K = 3 only"
    DST.mkdir(exist_ok=True)
    todo = [K for K in Ks if not (DST / f"hurdle_K{K}_{fset}.npy").exists()]
    if not todo:
        return
    j = pd.read_parquet(OUT / "joined_trials.parquet")
    keys = j[["ses_idx", "trial"]].to_numpy()
    dense, tis = features(j, fset)
    groups, sessions = j["subject_id"].astype(str).to_numpy(), j["ses_idx"].astype(str).to_numpy()
    j3 = j[[f"pred_{k}" for k in range(3)]].to_numpy()
    del j
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    fold = np.array([fom[g] for g in groups])
    folds = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(5)]
    Xs = fold_features(dense, tis, folds)
    for K in todo:
        g = pd.read_parquet(KOUT / f"glmhmm_hurdle_K{K}.parquet")
        assert (g[["ses_idx", "trial"]].to_numpy() == keys).all()
        P = g[[f"pred_{k}" for k in range(K)]].to_numpy()
        if K == 3:   # the K-scan K = 3 target is the pilot probe's target
            assert np.abs(P - j3).max() < 1e-6, np.abs(P - j3).max()
        Q, C = np.zeros_like(P), np.zeros_like(P)
        for (tr, te), (Xtr, Xte) in zip(folds, Xs):
            predict, _ = kp.fit_soft_logistic(Xtr, P[tr])
            Q[te], C[te] = predict(Xte), P[tr].mean(0)
        s = kp.scores(P, Q, C, sessions)
        got = dict(s, **{f"r2_state{k}": v for k, v in enumerate(s["state_r2"])})
        for name, want in reference(fset, K):
            print(f"hurdle K{K} {fset} {name}: refit {got[name]:.5f}  pilot {want:.5f}", flush=True)
            assert abs(got[name] - want) < TOL, (fset, K, name, got[name], want)
        np.save(DST / f"hurdle_K{K}_{fset}.npy", Q.astype(np.float32))


if __name__ == "__main__":
    main()
