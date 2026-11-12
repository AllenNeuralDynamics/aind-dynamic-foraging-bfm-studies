"""K scan, probe side: decode the K-state predictive posterior from fixed feature sets.

Usage: kscan_probes.py <feature_set> <emission[,emission]> <K[,K...]>
  feature_set: latents | logits | history | untrained

Same mice, sessions, folds (fold_of_mouse from out/probe_results.json, GroupKFold seed 0) and
probe as the pilot's primary probe: soft-label multinomial logistic, L2 = 1e-4 on standardised
features (scaler fitted on the training fold), L-BFGS-B maxiter 1000, gtol 1e-7. History
features = the pilot's set 3 (design lags + causal EMAs + fold-fitted trial-index spline).
The pilot's secondary ridge-ALR probe is not run here.

Scores per (emission, K, set), pooled out-of-fold:
  pooled R^2 = 1 - sum_k SSE_k / sum_k SST_k (all trials; and on per-session-demeaned target and
  prediction); information recovered = 1 - KL_probe / KL_const, KL_const from the training-fold
  mean posterior; also mean per-state R^2, CE, H, balanced accuracy of the argmax.
Writes one JSON per job unit to out/kscan/parts/ (skips units already done).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
KOUT = OUT / "kscan"
PARTS = KOUT / "parts"
L2 = 1e-4

spec = importlib.util.spec_from_file_location("pilot_probes", PILOT / "scripts/50_probes.py")
pilot_probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot_probes)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_soft_logistic(X, P):
    n, d = X.shape
    K = P.shape[1]
    Xb = np.hstack([X, np.ones((n, 1))])

    def f(w):
        W = w.reshape(d + 1, K)
        Z = Xb @ W
        Z = Z - Z.max(1, keepdims=True)
        logq = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        loss = -(P * logq).sum() / n + 0.5 * L2 * (W[:-1] ** 2).sum()
        G = Xb.T @ (np.exp(logq) - P) / n
        G[:-1] += L2 * W[:-1]
        return loss, G.ravel()

    res = minimize(f, np.zeros((d + 1) * K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "gtol": 1e-7})
    W = res.x.reshape(d + 1, K)
    return lambda Xt: softmax(np.hstack([Xt, np.ones((len(Xt), 1))]) @ W), int(res.nit)


def scores(P, Q, C, sessions):
    K = P.shape[1]
    lg = lambda a: np.log(np.clip(a, 1e-12, 1))  # noqa: E731
    H = float(-(P * lg(P)).sum(1).mean())
    ce = float(-(P * lg(Q)).sum(1).mean())
    ce_c = float(-(P * lg(C)).sum(1).mean())
    sse = ((P - Q) ** 2).sum(0)
    sst = ((P - P.mean(0)) ** 2).sum(0)
    sm_p = pd.DataFrame(P).groupby(sessions).transform("mean").to_numpy()
    sm_q = pd.DataFrame(Q).groupby(sessions).transform("mean").to_numpy()
    dp, dq = P - sm_p, Q - sm_q
    sse_w = ((dp - dq) ** 2).sum(0)
    sst_w = (dp ** 2).sum(0)
    return {"pooled_r2": float(1 - sse.sum() / sst.sum()),
            "pooled_r2_within": float(1 - sse_w.sum() / sst_w.sum()),
            "info_recovered": float(1 - (ce - H) / (ce_c - H)),
            "mean_state_r2": float(np.mean(1 - sse / sst)), "state_r2": (1 - sse / sst).tolist(),
            "mean_state_r2_within": float(np.mean(1 - sse_w / sst_w)),
            "ce": ce, "target_entropy": H, "kl": ce - H, "kl_const": ce_c - H,
            "balanced_acc_argmax": float(balanced_accuracy_score(P.argmax(1), Q.argmax(1))), "K": K}


def main():
    fset, emissions, Ks = sys.argv[1], sys.argv[2].split(","), [int(k) for k in sys.argv[3].split(",")]
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [(e, K) for e in emissions for K in Ks if not (PARTS / f"{e}__K{K}__{fset}.json").exists()]
    if not todo:
        return
    j = pd.read_parquet(OUT / "joined_trials.parquet")
    keys = j[["subject_id", "ses_idx", "trial"]]
    tis = None
    spline = fset == "history"
    if fset == "latents":
        dense = j[[c for c in j.columns if c.startswith("latent_")]].to_numpy(np.float64)
    elif fset == "logits":
        dense = j[[f"choice_logit_{i}" for i in range(3)]].to_numpy(np.float64)
    elif fset == "history":
        lags, emas = pilot_probes.history_features(j)
        dense = np.hstack([lags, emas])
        tis = j["trial_in_session"].to_numpy(np.float64)[:, None]
    elif fset == "untrained":
        u = pd.read_parquet(OUT / "untrained_latents.parquet")
        assert (u[["ses_idx", "trial"]].to_numpy() == j[["ses_idx", "trial"]].to_numpy()).all()
        dense = u[[c for c in u.columns if c.startswith("latent_")]].to_numpy(np.float64)
    else:
        raise ValueError(fset)
    groups = j["subject_id"].astype(str).to_numpy()
    sessions = j["ses_idx"].astype(str).to_numpy()
    del j
    fold_of_mouse = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    fold = np.array([fold_of_mouse[g] for g in groups])
    folds = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(5)]
    # Fold-wise features are the same for every K: build them once.
    Xs = []
    for tr, te in folds:
        sc = StandardScaler().fit(dense[tr])
        a, b = [sc.transform(dense[tr])], [sc.transform(dense[te])]
        if spline:
            sp = SplineTransformer(n_knots=8, degree=3, knots="quantile").fit(tis[tr])
            s2 = StandardScaler().fit(sp.transform(tis[tr]))
            a.append(s2.transform(sp.transform(tis[tr])))
            b.append(s2.transform(sp.transform(tis[te])))
        Xs.append((np.hstack(a), np.hstack(b)))
    for e, K in todo:
        t0 = time.time()
        g = pd.read_parquet(KOUT / f"glmhmm_{e}_K{K}.parquet")
        assert (g[["ses_idx", "trial"]].to_numpy() == keys[["ses_idx", "trial"]].to_numpy()).all()
        P = g[[f"pred_{k}" for k in range(K)]].to_numpy()
        Q, C, iters = np.zeros_like(P), np.zeros_like(P), []
        for (tr, te), (Xtr, Xte) in zip(folds, Xs):
            predict, nit = fit_soft_logistic(Xtr, P[tr])
            Q[te] = predict(Xte)
            C[te] = P[tr].mean(0)
            iters.append(nit)
        s = scores(P, Q, C, sessions)
        s.update({"emission": e, "feature_set": fset, "n_features": int(Xs[0][0].shape[1]),
                  "logistic_iters": iters, "seconds": round(time.time() - t0, 1)})
        (PARTS / f"{e}__K{K}__{fset}.json").write_text(json.dumps(s, indent=1))
        print(e, K, fset, {k: round(v, 3) for k, v in s.items() if k in ("pooled_r2", "pooled_r2_within", "info_recovered")},
              iters, s["seconds"], flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
