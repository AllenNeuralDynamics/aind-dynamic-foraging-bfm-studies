"""Freeze r1: the training-mice pilot's numbers and example-session traces.

The pilot (code in analysis/pilot/, run 2026-11-07/08 on HPC compute nodes) wrote its outputs to
/allen/aind/scratch/han.hou/pilots/study30-probe/out/. This freeze reads them, records each file's sha256 and
the W&B runs behind them as sources, and writes two committed files that r1_pilot_decoding.py reads offline:

  frozen_r1.json          every score, confusion matrix, K-scan and shuffle result, TDR summary, provenance
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
    }, outputs=[OUT_NPZ])


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
