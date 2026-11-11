"""Stage 5: linear probes for the GLM-HMM state posterior from GRU latents and baselines.

Target (primary): the one-step predictive state posterior pred_k = p(z_t | y_1..t-1) under the
GLOBAL hurdle K = 3 fit. Secondary: filtered (incl. trial t) and smoothed, feature sets 1 and 3.

Probes (both fitted per outer fold on the same folds):
* soft-label multinomial logistic regression (primary; decided before seeing results): minimises
  the cross-entropy to the 3-vector target, L2 = 1e-4 on standardised features (N >> p).
* ridge on the additive log-ratio log(p_k / p_2), k = 0, 1 (targets clipped to [1e-6, 1 - 1e-6]),
  alpha by RidgeCV over an inner GroupKFold(3) by mouse; predictions mapped back with softmax.

CV: GroupKFold(5, shuffle=True, random_state=0) by mouse. Scaler (and the trial-index spline)
are fitted on the training fold only.

Metrics on pooled out-of-fold predictions: per-state R^2 on probabilities (all trials, and on
per-session-demeaned target and prediction), cross-entropy, target entropy, KL = CE - H,
balanced accuracy of the argmax.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
K = 3
SEED = 0
L2 = 1e-4
TAUS = (10, 50, 200)
EPS = 1e-6


def history_features(df):
    """GLM-HMM design (L = 10 lags x RewC, UnrC, Ignored, Rewarded; no bias/mask) + causal EMAs."""
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs
    blocks, emas = [], []
    for _, g in df.groupby("ses_idx", sort=False):
        c = g["choice"].to_numpy().astype(int)
        r = g["reward"].to_numpy().astype(float)
        x = build_inputs(c, r, 10)
        blocks.append(x[:, :40])  # drop bias and mask columns
        ign = (c == 2).astype(float)
        rew = np.where(c == 2, 0.0, r)
        cols = []
        for v in (ign, rew):
            for tau in TAUS:
                a = 1.0 / tau
                m = np.zeros(len(c))
                acc, wacc = 0.0, 0.0
                for t in range(len(c)):  # value at t uses trials < t only
                    m[t] = acc / wacc if wacc > 0 else 0.0
                    acc = (1 - a) * acc + a * v[t]
                    wacc = (1 - a) * wacc + a
                cols.append(m)
        emas.append(np.column_stack(cols))
    return np.vstack(blocks), np.vstack(emas)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_soft_logistic(X, P):
    n, d = X.shape
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


def alr(P):
    P = np.clip(P, EPS, 1 - EPS)
    P = P / P.sum(1, keepdims=True)
    return np.log(P[:, :2] / P[:, 2:3])


def metrics(P, Q, sessions):
    Qc = np.clip(Q, 1e-12, 1)
    Pc = np.clip(P, 1e-12, 1)
    ce = float(-(P * np.log(Qc)).sum(1).mean())
    h = float(-(P * np.log(Pc)).sum(1).mean())
    r2 = [float(1 - ((P[:, k] - Q[:, k]) ** 2).sum() / ((P[:, k] - P[:, k].mean()) ** 2).sum()) for k in range(K)]
    sm_p = pd.DataFrame(P).groupby(sessions).transform("mean").to_numpy()
    sm_q = pd.DataFrame(Q).groupby(sessions).transform("mean").to_numpy()
    dp, dq = P - sm_p, Q - sm_q
    r2w = [float(1 - ((dp[:, k] - dq[:, k]) ** 2).sum() / (dp[:, k] ** 2).sum()) for k in range(K)]
    bacc = float(balanced_accuracy_score(P.argmax(1), Q.argmax(1)))
    return {"r2": r2, "r2_within": r2w, "ce": ce, "target_entropy": h, "kl": ce - h,
            "balanced_acc_argmax": bacc}


def main():
    j = pd.read_parquet(OUT / "joined_trials.parquet")
    u = pd.read_parquet(OUT / "untrained_latents.parquet")
    assert (u[["ses_idx", "trial"]].to_numpy() == j[["ses_idx", "trial"]].to_numpy()).all()
    H = sum(c.startswith("latent_") for c in j.columns)
    lat = j[[f"latent_{i}" for i in range(H)]].to_numpy(np.float64)
    ulat = u[[f"latent_{i}" for i in range(H)]].to_numpy(np.float64)
    logits = j[[f"choice_logit_{i}" for i in range(3)]].to_numpy(np.float64)
    lags, emas = history_features(j)
    tis = j["trial_in_session"].to_numpy(np.float64)[:, None]
    # Feature sets: dense part + whether to append the (fold-fitted) trial-index spline basis.
    sets = {
        "1_gru_latents": (lat, False),
        "2_gru_logits": (logits, False),
        "3_history": (np.hstack([lags, emas]), True),
        "4_latents+history": (np.hstack([lat, lags, emas]), True),
        "5_untrained_latents": (ulat, False),
        "6_trial_index": (np.zeros((len(j), 0)), True),
    }
    targets = {"pred": j[[f"pred_{k}" for k in range(K)]].to_numpy(),
               "filtered": j[[f"filtered_{k}" for k in range(K)]].to_numpy(),
               "smoothed": j[[f"smoothed_{k}" for k in range(K)]].to_numpy()}
    groups = j["subject_id"].astype(str).to_numpy()
    sessions = j["ses_idx"].astype(str).to_numpy()
    folds = list(GroupKFold(5, shuffle=True, random_state=SEED).split(lat, groups=groups))
    fold_of_mouse = {str(m): int(f) for f, (_, te) in enumerate(folds) for m in np.unique(groups[te])}

    # Variance split of each target: between-session (session means) vs within-session.
    var_split = {}
    for tname, P in targets.items():
        sm = pd.DataFrame(P).groupby(sessions).transform("mean").to_numpy()
        tot = ((P - P.mean(0)) ** 2).sum(0)
        within = ((P - sm) ** 2).sum(0)
        var_split[tname] = {"within_frac": (within / tot).tolist(), "between_frac": (1 - within / tot).tolist(),
                            "total_var": (tot / len(P)).tolist()}

    jobs = [(s, "pred") for s in sets] + [(s, t) for t in ("filtered", "smoothed") for s in ("1_gru_latents", "3_history")]
    parts = OUT / "probe_parts"
    parts.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    results, oof_store = {}, {}
    if mode == "merge":
        for sname, tname in jobs:
            results[f"{tname}|{sname}"] = json.loads((parts / f"{tname}__{sname}.json").read_text())
        for sname in ("1_gru_latents", "3_history"):
            oof_store[sname] = np.load(parts / f"oof_pred__{sname}.npy")
        jobs = []
    elif mode != "all":
        jobs = [jobs[int(mode)]]
    for sname, tname in jobs:
        dense, spline = sets[sname]
        P = targets[tname]
        Q_log = np.zeros_like(P)
        Q_rdg = np.zeros_like(P)
        info = []
        t0 = time.time()
        for f, (tr, te) in enumerate(folds):
            parts_tr, parts_te = [], []
            if dense.shape[1]:
                sc = StandardScaler().fit(dense[tr])
                parts_tr.append(sc.transform(dense[tr]))
                parts_te.append(sc.transform(dense[te]))
            if spline:
                sp = SplineTransformer(n_knots=8, degree=3, knots="quantile").fit(tis[tr])
                b_tr, b_te = sp.transform(tis[tr]), sp.transform(tis[te])
                sc2 = StandardScaler().fit(b_tr)
                parts_tr.append(sc2.transform(b_tr))
                parts_te.append(sc2.transform(b_te))
            Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
            predict, nit = fit_soft_logistic(Xtr, P[tr])
            Q_log[te] = predict(Xte)
            inner = list(GroupKFold(3).split(Xtr, groups=groups[tr]))
            rcv = RidgeCV(alphas=np.logspace(-2, 5, 8), cv=inner).fit(Xtr, alr(P[tr]))
            Z = rcv.predict(Xte)
            Q_rdg[te] = softmax(np.hstack([Z, np.zeros((len(Z), 1))]))
            info.append({"fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "logistic_iters": nit,
                         "ridge_alpha": float(rcv.alpha_), "n_features": int(Xtr.shape[1])})
        const = np.zeros_like(P)
        for tr, te in folds:
            const[te] = P[tr].mean(0)
        results[f"{tname}|{sname}"] = {
            "target": tname, "feature_set": sname, "n_features": info[0]["n_features"],
            "logistic": metrics(P, Q_log, sessions), "ridge_alr": metrics(P, Q_rdg, sessions),
            "constant_baseline_ce": float(-(P * np.log(np.clip(const, 1e-12, 1))).sum(1).mean()),
            "folds": info, "seconds": round(time.time() - t0, 1)}
        print(tname, sname, json.dumps({m: results[f"{tname}|{sname}"][m] for m in ("logistic", "ridge_alr")}), flush=True)
        (parts / f"{tname}__{sname}.json").write_text(json.dumps(results[f"{tname}|{sname}"], indent=1))
        if tname == "pred" and sname in ("1_gru_latents", "3_history"):
            oof_store[sname] = Q_log
            np.save(parts / f"oof_pred__{sname}.npy", Q_log)
    if mode not in ("all", "merge"):
        return

    pd.DataFrame({"subject_id": j["subject_id"], "ses_idx": j["ses_idx"], "trial": j["trial"],
                  **{f"probe_latents_{k}": oof_store["1_gru_latents"][:, k] for k in range(K)},
                  **{f"probe_history_{k}": oof_store["3_history"][:, k] for k in range(K)}}
                 ).to_parquet(OUT / "probe_oof_predictions.parquet")
    (OUT / "probe_results.json").write_text(json.dumps({
        "variance_split": var_split, "results": results, "fold_seed": SEED,
        "fold_of_mouse": fold_of_mouse, "l2_logistic": L2, "ema_taus": TAUS,
        "trial_index_basis": "SplineTransformer(n_knots=8, degree=3, knots='quantile'), fitted per fold",
    }, indent=1))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
