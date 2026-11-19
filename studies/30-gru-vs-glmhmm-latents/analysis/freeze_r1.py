"""Freeze r1: the training-mice pilot's numbers and example-session traces.

The pilot (code in analysis/pilot/, run 2026-11-07/08 on HPC compute nodes) wrote its outputs to
/allen/aind/scratch/han.hou/pilots/study30-probe/out/. This freeze reads them, records each file's sha256 and
the W&B runs behind them as sources, and writes two committed files that r1_pilot_decoding.py reads offline:

  frozen_r1.json          every score, confusion matrix, K-scan and shuffle result, TDR summary, provenance, and
                          `bootstrap`: 95% intervals over mice from the out-of-fold refit (analysis/oof_r1.py)
  frozen_r1_sessions.npz  per-trial traces of five example sessions (one showcase + four picked for clear
                          transitions, all in test fold 3), latents as float16

Needs the /allen mount, so it runs on an HPC compute node:
    make -C studies/30-gru-vs-glmhmm-latents freeze-r1
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
import upstream_sources  # noqa: E402  (analysis/upstream_sources.py)

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
SRC = PILOT / "out"
OUT_JSON, OUT_NPZ = HERE / "frozen_r1.json", HERE / "frozen_r1_sessions.npz"
# Study 14's GLM-HMM groups. The GRU run (study 02, group nxd-3way@20260707-075401, which has no launch record
# in study 02) is read through its W&B artifact and recorded as a `wandb_run` source and in `upstream`.
WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
GRU_PROJECT, GLM_PROJECT = "AIND-disRNN/mice_ignore_scaling", "AIND-disRNN/mice_glmhmm"
KS_SCAN, KS_SHOW, TEST_FOLD = list(range(2, 11)), [2, 3, 4, 6, 10], 2
CONF = 0.6
_sources: list[dict] = []


def src(path: Path) -> Path:
    """Register a scratch file as a freeze source (sha256) and return it."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    _sources.append({"type": "file", "id": str(path), "sha256": digest.hexdigest()})
    return path


def jload(path: Path):
    return json.loads(src(path).read_text())


def state_order(P, choice):
    """States sorted by p(ignore), high to low (state 0 = most ignore-prone)."""
    return np.argsort(-((P * (choice == 2)[:, None]).sum(0) / P.sum(0)))


def confusion(P, Q, conf):
    K = P.shape[1]
    t, d = P[conf].argmax(1), Q[conf].argmax(1)
    cm = np.zeros((K, K))
    np.add.at(cm, (t, d), 1)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    bacc = float(np.mean([row[k, k] for k in range(K) if cm[k].sum() > 0]))
    return {"counts": cm.astype(int).tolist(), "row": np.round(row, 4).tolist(), "balanced_acc": bacc}


def main():
    pilot = jload(SRC / "results.json")
    probe3 = jload(SRC / "probe_results.json")
    conf3 = jload(SRC / "confusion_results.json")
    tdr = jload(SRC / "tdr_results.json")
    selected = jload(SRC / "selected_sessions.json")
    heat_meta = jload(SRC / "fig_unit_heatmaps.json")
    kscan_glm = {e: jload(SRC / "kscan" / f"glmhmm_{e}_summary.json") for e in ("hurdle", "reference")}
    kscan = {p.stem: json.loads(src(p).read_text()) for p in sorted((SRC / "kscan" / "parts").glob("*.json"))}
    shuffle = {p.stem: json.loads(src(p).read_text()) for p in sorted((SRC / "shuffle" / "parts").glob("*.json"))}

    # ---- per-trial inputs (large; read, never committed)
    lat_cols = [c for c in pq.ParquetFile(SRC / "joined_trials.parquet").schema.names if c.startswith("latent_")]
    j = pd.read_parquet(src(SRC / "joined_trials.parquet"),
                        columns=["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward", "stage",
                                 "reward_probabilityL", "reward_probabilityR", "choice_prob_0", "choice_prob_1"]
                        + lat_cols)
    j["subject_id"], j["ses_idx"] = j["subject_id"].astype(str), j["ses_idx"].astype(str)
    oof3 = pd.read_parquet(src(SRC / "probe_oof_predictions.parquet"))
    oof3["ses_idx"] = oof3["ses_idx"].astype(str)
    assert (oof3[["ses_idx", "trial"]].to_numpy() == j[["ses_idx", "trial"]].to_numpy()).all()
    choice = j["choice"].to_numpy()
    ses = j["ses_idx"].to_numpy()
    fom = probe3["fold_of_mouse"]
    fold = j["subject_id"].map(fom).to_numpy()

    target, dec_lat, dec_his = {}, {}, {}
    for K in KS_SHOW:
        g = pd.read_parquet(src(SRC / "kscan" / f"glmhmm_hurdle_K{K}.parquet"))
        assert (g[["ses_idx", "trial"]].astype(str).to_numpy() == j[["ses_idx", "trial"]].astype(str).to_numpy()).all()
        P = g[[f"pred_{k}" for k in range(K)]].to_numpy()
        # K = 3 keeps the pilot's own order and names (S0 ignore-prone, S1 engaged strong win-stay, S2 engaged
        # weak win-stay); every other K is sorted by p(ignore), high to low
        o = np.arange(3) if K == 3 else state_order(P, choice)
        target[K] = P[:, o]
        if K == 3:   # the pilot's own K = 3 probe (same posterior as the K-scan run, to 1e-15)
            dec_lat[K] = oof3[[f"probe_latents_{k}" for k in range(3)]].to_numpy()[:, o]
            dec_his[K] = oof3[[f"probe_history_{k}" for k in range(3)]].to_numpy()[:, o]
        else:
            dec_lat[K] = np.load(src(SRC / "kscan" / "oof" / f"hurdle_K{K}_latents.npy"))[:, o]
            dec_his[K] = np.load(src(SRC / "kscan" / "oof" / f"hurdle_K{K}_history.npy"))[:, o]

    # confusion matrices (out of fold, max pi >= CONF) and scoring scatters (2-D histograms) at K = 3
    confusions = {str(K): {f: confusion(target[K], q[K], target[K].max(1) >= CONF)
                           for f, q in (("latents", dec_lat), ("history", dec_his))} for K in KS_SHOW}
    for K in KS_SHOW:
        confusions[str(K)]["frac_confident"] = float((target[K].max(1) >= CONF).mean())
    dm = lambda v: v - pd.Series(v).groupby(ses).transform("mean").to_numpy()  # noqa: E731
    p1 = target[3][:, 1]
    hist2d = {}
    for name, x, y, lim in (("latents_all", p1, dec_lat[3][:, 1], (0, 1)),
                            ("latents_within", dm(p1), dm(dec_lat[3][:, 1]), (-0.8, 0.8)),
                            ("history_all", p1, dec_his[3][:, 1], (0, 1))):
        H, _, _ = np.histogram2d(x, y, bins=40, range=[lim, lim])
        r2 = 1 - ((x - y) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        hist2d[name] = {"counts": H.astype(int).tolist(), "lim": list(lim), "r2": float(r2)}

    # ---- example sessions (all in test fold 3: probe, Rastermap and TDR never saw these mice)
    showcase = heat_meta["fig_unit_heatmap_0.svg"]["session"]
    sessions = [showcase] + selected["sessions"]
    tags = ["showcase: the session with the most posterior movement"] + selected["tags"]
    sc = np.load(src(SRC / "kscan" / "weights" / f"hurdle_K2_fold{TEST_FOLD}.npz"))
    Z = (j[lat_cols].to_numpy() - sc["mean"]) / sc["scale"]
    weights = {K: np.load(src(SRC / "kscan" / "weights" / f"hurdle_K{K}_fold{TEST_FOLD}.npz"))
               for K in (2, 4, 6, 10)}
    rm = np.load(src(SRC / "rastermap_order.npz"))
    from scipy.stats import spearmanr
    rm_stability = {n: float(abs(spearmanr(rm[f"{n}_s0_embedding"], rm[f"{n}_s1_embedding"])[0]))
                    for n in ("raw", "smooth")}
    train = np.flatnonzero(fold != TEST_FOLD)
    samp = np.sort(np.random.default_rng(0).choice(train, 100_000, replace=False))
    Zs = Z[samp]

    def unit_r(P):   # Pearson r of each unit with each state, training-fold mice only
        Ps = P[samp]
        return ((Zs - Zs.mean(0)).T @ (Ps - Ps.mean(0))) / (len(samp) * Zs.std(0)[:, None] * Ps.std(0)[None, :])

    r3, r4 = unit_r(target[3]), unit_r(target[4])
    # Top principal components (fitted on the same 100k-trial training sample as analysis/pilot/93_unit_views.py)
    # and their slow share: the variance a centred 20-trial moving average keeps, computed on COMPLETE
    # training-fold sessions (the pilot averaged over the random sample, i.e. over non-consecutive trials).
    from sklearn.decomposition import PCA
    pca = PCA(6).fit(Zs)
    pcs = pd.DataFrame(pca.transform(Z[train]))
    pcs_smooth = pcs.groupby(ses[train]).transform(lambda v: v.rolling(20, center=True, min_periods=1).mean())
    pc_stats = {"explained_var": pca.explained_variance_ratio_.round(4).tolist(),
                "slow_fraction": (pcs_smooth.var() / pcs.var()).round(3).tolist(),
                "window_trials": 20, "fit_sample": "100,000 training-fold trials, numpy default_rng(0)",
                "slow_fraction_on": "all training-fold trials, complete sessions"}
    tdr_ax = np.load(src(SRC / "tdr_axes.npz"))
    pilot_tdr = pilot_tdr_regressors(j, target[3])
    npz = {"rastermap_raw_isort": rm["raw_s0_isort"], "rastermap_smooth_isort": rm["smooth_s0_isort"],
           "unit_r_K3": r3.astype(np.float32), "unit_r_K4": r4.astype(np.float32)}
    sess_info = []
    for i, s in enumerate(sessions):
        rows = np.flatnonzero(ses == s)
        assert len(rows) and (fold[rows] == TEST_FOLD).all(), s
        g = j.iloc[rows]
        pre = f"s{i}_"
        npz[pre + "trial_in_session"] = g["trial_in_session"].to_numpy().astype(np.int32)
        npz[pre + "choice"] = g["choice"].to_numpy().astype(np.int8)
        npz[pre + "reward"] = g["reward"].to_numpy().astype(np.int8)
        npz[pre + "pL"] = g["reward_probabilityL"].to_numpy().astype(np.float32)
        npz[pre + "pR"] = g["reward_probabilityR"].to_numpy().astype(np.float32)
        npz[pre + "gru_p_right_given_respond"] = (g["choice_prob_1"] / (g["choice_prob_0"] + g["choice_prob_1"])
                                                  ).to_numpy().astype(np.float32)
        npz[pre + "latents_z"] = Z[rows].astype(np.float16)
        for K in KS_SHOW:
            npz[pre + f"target_K{K}"] = target[K][rows].astype(np.float32)
            npz[pre + f"decoded_latents_K{K}"] = dec_lat[K][rows].astype(np.float32)
            npz[pre + f"decoded_history_K{K}"] = dec_his[K][rows].astype(np.float32)
        for K, w in weights.items():
            P = pd.read_parquet(SRC / "kscan" / f"glmhmm_hurdle_K{K}.parquet",
                                columns=[f"pred_{k}" for k in range(K)]).to_numpy()
            o = state_order(P, choice)
            L = Z[rows] @ w["W"][:-1][:, o] + w["W"][-1][o]
            npz[pre + f"state_logit_K{K}"] = (L - L.mean(1, keepdims=True)).astype(np.float32)
        proj = (Z[rows] - tdr_ax["z_mean_train"]) @ tdr_ax["axes_n"]
        npz[pre + "tdr_projection_z"] = ((proj - tdr_ax["proj_mean_train"]) / tdr_ax["proj_sd_train"]).astype(np.float32)
        npz[pre + "tdr_variable_z"] = ((pilot_tdr[rows] - tdr_ax["x_mean"]) / tdr_ax["x_sd"]).astype(np.float32)
        sess_info.append({"key": pre, "session": s, "mouse": str(g["subject_id"].iloc[0]),
                          "stage": str(g["stage"].iloc[0]), "n_trials": int(len(rows)), "tag": tags[i]})
    np.savez_compressed(OUT_NPZ, **npz)

    boot = bootstrap(j, fold, ses, j["subject_id"].to_numpy(), pilot_tdr, Z)
    # every point estimate (the full sample) must be the pilot's stored score: the refit adds intervals only
    diffs = [abs(boot["kscan"][str(K)][f][m]["point"] - kscan[f"hurdle__K{K}__{f}"][m])
             for K in KS_SCAN for f in FSETS_SCAN for m in ("pooled_r2", "pooled_r2_within", "info_recovered")]
    diffs += [abs(a - b) for f, s in (("latents+history", "4_latents+history"), ("trial_index", "6_trial_index"))
              for key in ("r2", "r2_within")
              for a, b in zip(boot["k3"][f][key]["point"], probe3["results"][f"pred|{s}"]["logistic"][key])]
    diffs += [abs(boot["tdr"]["r2_full_heldout"]["point"] - tdr["r2_full_heldout"]),
              abs(boot["tdr"]["cos_prev_side_prev_ign"]["point"] - tdr["cosine"][3][4])]
    boot["max_abs_point_vs_pilot"] = float(max(diffs))
    assert boot["max_abs_point_vs_pilot"] < 1e-3, boot["max_abs_point_vs_pilot"]

    groups_runs = [{"type": "wandb_run", "id": f"{GRU_PROJECT}/{pilot['gru']['run_id']}"}]
    groups_runs += [{"type": "wandb_run", "id": f"{GLM_PROJECT}/{v['run_id']}"}
                    for e in ("hurdle", "reference") for v in kscan_glm[e]["per_K"].values()]
    meta = upstream_sources.apply(build_meta(
        "analysis/freeze_r1.py", WANDB_GROUPS, study_root=HERE.parent, kind="freeze",
        code=[__file__, HERE / "upstream_sources.py", HERE / "pilot" / "95_tdr.py"],
        inputs=upstream_sources.RECORDS, sources=groups_runs + _sources))
    write_artifact(OUT_JSON, {
        "_meta": meta,
        "pilot": {"gru": pilot["gru"], "glmhmm": pilot["glmhmm"], "pilot": pilot["pilot"],
                  "verification": pilot["verification"], "step1_filtered_convention": pilot["step1_filtered_convention"],
                  "code": pilot["_meta"]},
        "k3": {"results": probe3["results"], "variance_split": probe3["variance_split"],
               "fold_of_mouse": fom, "l2_logistic": probe3["l2_logistic"], "ema_taus": probe3["ema_taus"]},
        "k3_engaged": conf3,
        "kscan": {"glmhmm": {e: kscan_glm[e]["per_K"] for e in kscan_glm}, "parts": kscan},
        "confusions": confusions,
        "hist2d_k3_state1": hist2d,
        "shuffle": shuffle,
        "tdr": tdr,
        "population": {"pc": pc_stats, "rastermap_seed_stability_spearman": rm_stability,
                       "rastermap_params": {"n_clusters": 40, "n_PCs": 64, "locality": 0.0, "version": "1.0"}},
        "sessions": {"selection_criterion": selected["criterion"], "items": sess_info,
                     "npz": "analysis/frozen_r1_sessions.npz"},
        "bootstrap": boot,
    }, outputs=[OUT_NPZ])


# ---------------------------------------------------------------- bootstrap over mice
# Percentile 95% intervals from resampling mice with replacement. Every score is a ratio of sums over
# trials, so each is computed from per-mouse sums: one weight matrix W (row 0 = the full sample, the point
# estimate; rows 1.. = resamples) times per-mouse sums gives every resample exactly, denominators
# re-centred on the resampled mice.
N_BOOT, BOOT_SEED = 1000, 0
FSETS_K3 = ["latents", "logits", "history", "latents+history", "untrained", "trial_index"]
FSETS_SCAN = ["latents", "untrained", "logits", "history"]
ENGAGED, CONFIDENT = 0.2, 0.8   # result 3's definitions (analysis/pilot/85_confusion.py)


def boot_weights(n):
    rng = np.random.default_rng(BOOT_SEED)
    W = np.ones((N_BOOT + 1, n))
    for b in range(N_BOOT):
        W[b + 1] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def per_mouse(v, m, n):
    """Sum the rows of v (trials x ...) by mouse index m: (n mice x ...)."""
    out = np.zeros((n,) + v.shape[1:])
    np.add.at(out, m, v)
    return out


def ci(rows):
    """{point, lo, hi} from bootstrap rows (row 0 = full sample); vectors stay vectors."""
    rows = np.asarray(rows)
    lo, hi = np.percentile(rows[1:], [2.5, 97.5], axis=0)
    f = lambda x: x.tolist() if np.ndim(x) else float(x)  # noqa: E731
    return {"point": f(rows[0]), "lo": f(lo), "hi": f(hi)}


def decoding_boot(P, Q, C, ses, m, W):
    """Per-state and pooled R² (all trials, within session), information recovered and KL, by resample."""
    n = W.shape[1]
    lg = lambda a: np.log(np.clip(a, 1e-12, 1))  # noqa: E731
    sm = lambda A: pd.DataFrame(A).groupby(ses).transform("mean").to_numpy()  # noqa: E731
    dp, dq = P - sm(P), Q - sm(Q)
    S = {k: W @ per_mouse(v, m, n) for k, v in {
        "n": np.ones((len(P), 1)), "p": P, "p2": P ** 2, "sse": (P - Q) ** 2, "ssew": (dp - dq) ** 2,
        "sstw": dp ** 2, "ce": -(P * lg(Q)).sum(1, keepdims=True), "cec": -(P * lg(C)).sum(1, keepdims=True),
        "h": -(P * lg(P)).sum(1, keepdims=True)}.items()}
    sst = S["p2"] - S["p"] ** 2 / S["n"]
    kl, klc = (S["ce"] - S["h"])[:, 0], (S["cec"] - S["h"])[:, 0]
    rows = {"pooled_r2": 1 - S["sse"].sum(1) / sst.sum(1), "pooled_r2_within": 1 - S["ssew"].sum(1) / S["sstw"].sum(1)}
    return {"r2": ci(1 - S["sse"] / sst), "r2_within": ci(1 - S["ssew"] / S["sstw"]),
            "pooled_r2": ci(rows["pooled_r2"]), "pooled_r2_within": ci(rows["pooled_r2_within"]),
            "info_recovered": ci(1 - kl / klc), "kl": ci(kl / S["n"][:, 0]), "_rows": rows}


def engaged_boot(P, Q, ses, m, W):
    """Result 3: confusion on confident trials (balanced accuracy, S2 → S2) and S1-vs-S2 R² on engaged trials."""
    n = W.shape[1]
    conf = P.max(1) >= CONFIDENT
    cm = np.zeros((len(P), 3, 3))
    cm[np.flatnonzero(conf), P[conf].argmax(1), Q[conf].argmax(1)] = 1
    CM = np.einsum("bm,mij->bij", W, per_mouse(cm, m, n))
    row = CM / CM.sum(2, keepdims=True)
    diag = np.stack([row[:, k, k] for k in range(3)], 1)
    eng = P[:, 0] < ENGAGED
    c, ch, s = P[eng, 1] / P[eng, 1:].sum(1), Q[eng, 1] / Q[eng, 1:].sum(1), ses[eng]
    dm = lambda v: v - pd.Series(v).groupby(s).transform("mean").to_numpy()  # noqa: E731
    me = m[eng]
    S = {k: W @ per_mouse(v, me, n) for k, v in {
        "n": np.ones(len(c)), "c": c, "c2": c ** 2, "sse": (c - ch) ** 2,
        "ssew": (dm(c) - dm(ch)) ** 2, "sstw": dm(c) ** 2}.items()}
    return {"balanced_acc_confident": ci(diag.mean(1)), "s2_to_s2_confident": ci(diag[:, 2]),
            "engaged_r2_S1_vs_S2": ci(1 - S["sse"] / (S["c2"] - S["c"] ** 2 / S["n"])),
            "engaged_r2_S1_vs_S2_within": ci(1 - S["ssew"] / S["sstw"]),
            "n_confident_trials": int(conf.sum()), "n_engaged_trials": int(eng.sum())}


def tdr_boot(Z, X, fold, mouse):
    """TDR as in analysis/pilot/95_tdr.py: held-out and unique R² resampled over the 10 test-fold mice
    (axes fitted once on the 40 training mice), and the cosine between the choice t−1 and ignore t−1 axes
    resampled over the 40 training mice (regression and PC denoising refitted on each resample)."""
    tr, te = np.flatnonzero(fold != TEST_FOLD), np.flatnonzero(fold == TEST_FOLD)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xs = (X - mu) / sd
    names = ["pi_ign", "pi_weak", "side_ema", "prev_side", "prev_ign", "prev_rew", "prev_win", "rew_ema", "tis"]
    allc = list(range(len(names)))
    ztr_mean = Z[tr].mean(0)
    _, mte = np.unique(mouse[te], return_inverse=True)
    Wte = boot_weights(mte.max() + 1)
    Zte = Z[te] - ztr_mean
    s1, s2 = per_mouse(Zte, mte, Wte.shape[1]), per_mouse(Zte ** 2, mte, Wte.shape[1])
    nte = per_mouse(np.ones(len(te)), mte, Wte.shape[1])
    sst = ((Wte @ s2) - (Wte @ s1) ** 2 / (Wte @ nte)[:, None]).sum(1)

    def r2_rows(cols):
        A = np.column_stack([Xs[tr][:, cols], np.ones(len(tr))])
        B, *_ = np.linalg.lstsq(A, Z[tr] - ztr_mean, rcond=None)
        sse = ((Zte - (Xs[te][:, cols] @ B[:-1] + B[-1])) ** 2).sum(1)
        return 1 - (Wte @ per_mouse(sse, mte, Wte.shape[1])) / sst

    full = r2_rows(allc)
    unique = {names[k]: ci(full - r2_rows([i for i in allc if i != k])) for k in allc}

    # cosine: per-training-mouse normal equations and PC scatter (the pilot's every-4th-row subsample)
    _, mtr = np.unique(mouse[tr], return_inverse=True)
    ntr = mtr.max() + 1
    Wtr = boot_weights(ntr)
    A = np.column_stack([Xs[tr], np.ones(len(tr))])
    AtA = per_mouse(A[:, :, None] * A[:, None, :], mtr, ntr)
    AtZ = np.stack([A[mtr == i].T @ Z[tr][mtr == i] for i in range(ntr)])
    s_all, n_all = per_mouse(Z[tr], mtr, ntr), per_mouse(np.ones(len(tr)), mtr, ntr)
    sub = np.arange(0, len(tr), 4)
    Zsub, msub = Z[tr][sub], mtr[sub]
    zz = np.stack([Zsub[msub == i].T @ Zsub[msub == i] for i in range(ntr)])
    s_sub, n_sub = per_mouse(Zsub, msub, ntr), per_mouse(np.ones(len(sub)), msub, ntr)
    cos = []
    for w in Wtr:
        B = np.linalg.solve(np.tensordot(w, AtA, 1), np.tensordot(w, AtZ, 1))[:-1]
        c = (w @ s_all) / (w @ n_all)
        ws = w @ s_sub
        M = np.tensordot(w, zz, 1) - np.outer(c, ws) - np.outer(ws, c) + (w @ n_sub) * np.outer(c, c)
        V = np.linalg.eigh(M)[1][:, ::-1][:, :12]          # top 12 PCs (N_PC in 95_tdr.py)
        ax = (B @ V @ V.T).T
        ax /= np.linalg.norm(ax, axis=0, keepdims=True)
        cos.append(ax[:, names.index("prev_side")] @ ax[:, names.index("prev_ign")])
    return {"r2_full_heldout": ci(full), "unique_r2": unique, "cos_prev_side_prev_ign": ci(cos),
            "n_test_mice": int(Wte.shape[1]), "n_train_mice": int(ntr)}


def bootstrap(j, fold, ses, mouse_ids, X_tdr, Z):
    m = np.unique(mouse_ids, return_inverse=True)[1]
    W = boot_weights(m.max() + 1)
    oof = lambda K, f: np.load(src(SRC / "oof_r1" / f"hurdle_K{K}_{f}.npy")).astype(np.float64)  # noqa: E731
    folds = [np.flatnonzero(fold == f) for f in range(5)]
    out = {"n_mice": int(W.shape[1]), "n_boot": N_BOOT, "seed": BOOT_SEED, "interval": "percentile 95%, mice resampled",
           "k3": {}, "k3_engaged": {}, "kscan": {}}
    rows = {}
    for K in KS_SCAN:
        P = pd.read_parquet(SRC / "kscan" / f"glmhmm_hurdle_K{K}.parquet",
                            columns=[f"pred_{k}" for k in range(K)]).to_numpy()
        C = np.zeros_like(P)
        for te in folds:             # the constant guess: the training-fold mean posterior
            C[te] = np.delete(P, te, 0).mean(0)
        fsets = FSETS_K3 if K == 3 else FSETS_SCAN
        out["kscan"][str(K)] = {}
        for f in fsets:
            r = decoding_boot(P, oof(K, f), C, ses, m, W)
            rows[(K, f)] = r.pop("_rows")
            out["kscan"][str(K)][f] = {k: r[k] for k in ("pooled_r2", "pooled_r2_within", "info_recovered")}
            if K == 3:   # K = 3 keeps the pilot's state order (S0, S1, S2)
                out["k3"][f] = r
                if f in ("latents", "logits", "history"):
                    out["k3_engaged"][f] = engaged_boot(P, oof(K, f), ses, m, W)
    # paired differences (the same resampled mice on both sides)
    out["paired"] = {
        f"latents_K3_minus_K10_{s}": ci(rows[(3, "latents")][s] - rows[(10, "latents")][s])
        for s in ("pooled_r2", "pooled_r2_within")}
    out["paired"].update({f"latents_minus_history_K{K}_{s}": ci(rows[(K, "latents")][s] - rows[(K, "history")][s])
                          for K in KS_SCAN for s in ("pooled_r2", "pooled_r2_within")})
    out["tdr"] = tdr_boot(Z, X_tdr, fold, mouse_ids)
    return out


def pilot_tdr_regressors(j, P3):
    """The TDR regressors of analysis/pilot/95_tdr.py, for the K = 3 posterior in p(ignore) order."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tdr", HERE / "pilot" / "95_tdr.py")
    tdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdr)
    c = j["choice"].to_numpy()
    ses = j["ses_idx"].to_numpy()
    first = np.r_[True, ses[1:] != ses[:-1]]
    pc, pr = np.r_[-1, c[:-1]], np.r_[0, j["reward"].to_numpy()[:-1]]
    m = (~first) & (c != 2) & (pc != 2) & (pr == 1)
    ws = (P3[m] * (c[m] == pc[m])[:, None]).sum(0) / P3[m].sum(0)
    if ws[1] > ws[2]:
        P3 = P3[:, [0, 2, 1]]
    return tdr.regressors(j, P3)


if __name__ == "__main__":
    main()
