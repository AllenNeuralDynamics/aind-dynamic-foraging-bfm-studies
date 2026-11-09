"""K scan, saved out-of-fold predictions: the same probe as kscan_probes.py, for chosen K, keeping Q.

Usage: 90_kscan_oof.py <feature_set> <emission> <K[,K...]>   (feature_set: latents | history)
Writes out/kscan/oof/<emission>_K<K>_<feature_set>.npy (N x K, rows in joined_trials order) and
checks that the pooled R^2 reproduces the K-scan part file.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT, KOUT = PILOT / "out", PILOT / "out" / "kscan"
spec = importlib.util.spec_from_file_location("kp", PILOT / "scripts/kscan_probes.py")
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)


def main():
    fset, e, Ks = sys.argv[1], sys.argv[2], [int(k) for k in sys.argv[3].split(",")]
    (KOUT / "oof").mkdir(exist_ok=True)
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
    for K in Ks:
        dst = KOUT / "oof" / f"{e}_K{K}_{fset}.npy"
        if dst.exists():
            continue
        g = pd.read_parquet(KOUT / f"glmhmm_{e}_K{K}.parquet")
        assert (g[["ses_idx", "trial"]].to_numpy() == keys).all()
        P = g[[f"pred_{k}" for k in range(K)]].to_numpy()
        Q, C = np.zeros_like(P), np.zeros_like(P)
        for (tr, te), (Xtr, Xte) in zip(folds, Xs):
            predict, _ = kp.fit_soft_logistic(Xtr, P[tr])
            Q[te], C[te] = predict(Xte), P[tr].mean(0)
        s = kp.scores(P, Q, C, sessions)
        ref = json.loads((KOUT / "parts" / f"{e}__K{K}__{fset}.json").read_text())
        print(e, K, fset, round(s["pooled_r2"], 5), "part:", round(ref["pooled_r2"], 5), flush=True)
        assert abs(s["pooled_r2"] - ref["pooled_r2"]) < 1e-3
        np.save(dst, Q.astype(np.float32))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
