"""Stage 8b: is the decoded state only engagement? Confusion matrices and engaged-only decoding.

Uses the pilot's out-of-fold probe predictions (latents, history) and refits the 3-dim GRU-logits
probe on the same folds (fast). Hurdle K = 3, predictive posterior target.

a-c  row-normalised confusion matrices, argmax(target) vs argmax(decoded), on trials where the
     target is confident (max pi >= 0.8), for latents / logits / history
d    engaged trials only (pi_S0 < 0.2): conditional c = pi_S1 / (pi_S1 + pi_S2) vs the decoded
     conditional from the same 3-class probe; R2 all and within session, per feature set
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
sys.path.insert(0, "/home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/"
                   "bridge-cse_01XAC3mEv4rcjCW6gNDnZbH4/studies/util")
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location("probes", Path(__file__).with_name("50_probes.py"))
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)

NAMES = ["S0 ignore-prone", "S1 engaged,\nstrong history", "S2 engaged,\nweak history"]
SHORT = ["S0", "S1", "S2"]
CONF, ENGAGED = 0.8, 0.2


def r2(p, q):
    return float(1 - ((p - q) ** 2).sum() / ((p - p.mean()) ** 2).sum())


def main():
    fs.apply_paper_style()
    cols = ["subject_id", "ses_idx", "trial", "pred_0", "pred_1", "pred_2",
            "choice_logit_0", "choice_logit_1", "choice_logit_2"]
    j = pd.read_parquet(OUT / "joined_trials.parquet", columns=cols)
    oof = pd.read_parquet(OUT / "probe_oof_predictions.parquet")
    for d in (j, oof):
        d["subject_id"] = d["subject_id"].astype(str)
        d["ses_idx"] = d["ses_idx"].astype(str)
    m = j.merge(oof, on=["subject_id", "ses_idx", "trial"], validate="one_to_one")
    P = m[[f"pred_{k}" for k in range(3)]].to_numpy()
    groups = m["subject_id"].to_numpy()
    folds = list(GroupKFold(5, shuffle=True, random_state=0).split(P, groups=groups))
    fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
    check = {str(mm): f for f, (_, te) in enumerate(folds) for mm in np.unique(groups[te])}
    assert check == fom, "folds differ from the pilot's"

    X = m[[f"choice_logit_{k}" for k in range(3)]].to_numpy(np.float64)
    Q_logit = np.zeros_like(P)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        predict, _ = probes.fit_soft_logistic(sc.transform(X[tr]), P[tr])
        Q_logit[te] = predict(sc.transform(X[te]))
    Q = {"GRU latents": m[[f"probe_latents_{k}" for k in range(3)]].to_numpy(),
         "GRU logits": Q_logit,
         "History": m[[f"probe_history_{k}" for k in range(3)]].to_numpy()}

    ses = m["ses_idx"].to_numpy()
    conf = P.max(1) >= CONF
    eng = P[:, 0] < ENGAGED
    c_true = P[:, 1] / (P[:, 1] + P[:, 2])
    out = {"n_trials": int(len(P)), "n_confident": int(conf.sum()), "n_engaged": int(eng.sum()),
           "engaged_share_S1_argmax": float((P[eng, 1] > P[eng, 2]).mean()), "sets": {}}
    for name, Qs in Q.items():
        t, d = P[conf].argmax(1), Qs[conf].argmax(1)
        cm = np.zeros((3, 3))
        np.add.at(cm, (t, d), 1)
        cm_row = cm / cm.sum(1, keepdims=True)
        c_hat = Qs[:, 1] / (Qs[:, 1] + Qs[:, 2])
        ce, ch, se = c_true[eng], c_hat[eng], ses[eng]
        dm = lambda v: v - pd.Series(v).groupby(se).transform("mean").to_numpy()  # noqa: E731
        t_all, d_all = P.argmax(1), Qs.argmax(1)
        out["sets"][name] = {
            "confusion_confident_counts": cm.astype(int).tolist(), "confusion_confident_row": cm_row.tolist(),
            "balanced_acc_confident": float(np.diag(cm_row).mean()),
            "balanced_acc_all": float(np.mean([(d_all[t_all == k] == k).mean() for k in range(3)])),
            "engaged_r2_S1_vs_S2": r2(ce, ch), "engaged_r2_S1_vs_S2_within": r2(dm(ce), dm(ch)),
            "engaged_acc_S1_vs_S2_confident": float(((ch > 0.5) == (ce > 0.5))[np.abs(ce - 0.5) >= 0.3].mean()),
            "r2_p_ignore_state": r2(P[:, 0], Qs[:, 0])}
    (OUT / "confusion_results.json").write_text(json.dumps(out, indent=1))

    W, H = 165, 58
    fig = plt.figure(figsize=fs.mm(W, H))
    ax_at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731  (mm)
    for i, name in enumerate(Q):
        ax = ax_at(16 + i * 34, 16, 27, 27)
        cmr = np.array(out["sets"][name]["confusion_confident_row"])
        ax.imshow(cmr, cmap="Greys", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, fs.num(cmr[a, b], ".2f"), ha="center", va="center",
                        color="white" if cmr[a, b] > 0.55 else "black")
        ax.set_xticks(range(3), SHORT)
        ax.set_yticks(range(3), SHORT)
        ax.set_xlabel("decoded (argmax)")
        if i == 0:
            ax.set_ylabel("GLM-HMM (argmax)")
        ax.set_title(f"{name}\nbalanced acc. {fs.num(out['sets'][name]['balanced_acc_confident'], '.2f')}",
                     fontsize=6)
        fs.panel_label(ax, "abc"[i], dx_pt=-30 if i == 0 else -14)
    ax = ax_at(126, 16, 36, 27)
    x = np.arange(3)
    for k, (key, lab, col) in enumerate([("engaged_r2_S1_vs_S2", "all engaged trials", "#555555"),
                                         ("engaged_r2_S1_vs_S2_within", "within session", "#bbbbbb")]):
        vals = [out["sets"][n][key] for n in Q]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, list(Q), rotation=25, ha="right")
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1)
    ax.set_ylabel("R², p(S1 | S1 or S2)")
    ax.set_title("engaged trials (p(S0) < 0.2):\nS1 vs S2 only", fontsize=6)
    ax.legend(frameon=False, fontsize=5, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    fs.panel_label(ax, "d", dx_pt=-26)
    fig.text(0.01, 0.015, f"a–c: out-of-fold, trials where the GLM-HMM is confident (max π ≥ {CONF}; "
             f"{out['n_confident']:,} of {out['n_trials']:,}); rows sum to 1.\nd: {out['n_engaged']:,} engaged trials; "
             "the same 3-class probes, renormalised over S1 and S2. Hurdle K = 3, 50 training mice. Scratch pilot.",
             fontsize=5)
    fs.save_figure(fig, OUT / "fig_confusion.svg")
    print(json.dumps({n: {k: v for k, v in s.items() if "counts" not in k and "row" not in k}
                      for n, s in out["sets"].items()}, indent=1))
    print({k: v for k, v in out.items() if k != "sets"})


if __name__ == "__main__":
    main()
