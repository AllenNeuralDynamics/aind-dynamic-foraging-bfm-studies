"""I9e: update-rule "trace" models suggested by v's kernel (I9d), fitted to the mice's choices.

    python analysis/i9e_trace_models.py     # GPU (JAX)

I9d described v as a fast choice trace, a reward boost and a slow residue of unrewarded choices.
Here each description becomes a generative model, fitted per mouse to the animals' choices
(maximum likelihood, Adam, all mice in parallel) and scored on held-out sessions (2-fold: each
mouse's sessions alternate between folds in session order; every trial is scored by the fold
that did not see it). c = +1 right / −1 left, r ∈ {0, 1}; traces start at 0 in every session.
  traces      C ← λ_c C + c              (every choice)
              W ← λ_w W + c r            (rewarded choices)
              U ← λ_u U + c (1 − r)      (unrewarded choices)
              logit P(right) = Σ β_x X + bias  [+ w_ws c(t−1) r(t−1) + w_ls c(t−1)(1 − r(t−1))]
  "choice + reward" = C + W;  "two outcomes" = W + U;  "three traces" = C + W + U.
  "+ last" adds the two last-trial weights (the behavioural counterpart of I9c's offsets).
Baselines refitted the same way: Q-learning in Hattori's form (α_rew, α_unrew, forgetting, β,
bias) and Q-learning + a choice kernel (K ← K + α_K (c − K), weight w_K; Bari-like), with and
without "+ last". Also scored on the same trials: the GRU and the wrapper's Hattori / Bari / CTT
fits (trained on other sessions — a reference, not a like-for-like comparison).
Writes ``i9e_trace_models.json``, ``_cache/i9e_*_<key>.{npz,parquet}`` and ``fig_I9e_trace_models.png``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from c4e_signatures import signature_curves  # noqa: E402
from h1_model_ladder import session_arrays  # noqa: E402

N_STEPS = 2500
LR = 0.03
NAMES = ["a_rew", "a_unrew", "forget", "beta", "a_K", "w_K", "lam_c", "b_c", "lam_w", "b_w", "lam_u", "b_u", "bias", "w_ws", "w_ls"]
# Components per model: q (Q-learning), K (choice kernel), C / W / U (traces), L (last trial).
MODELS = {
    "Q (Hattori form)": "q", "Q + kernel": "qK", "Q + kernel + last": "qKL",
    "choice + reward": "CW", "choice + reward + last": "CWL",
    "two outcomes": "WU", "two outcomes + last": "WUL",
    "three traces": "CWU", "three traces + last": "CWUL",
}
logit_ = lambda p: float(np.log(p / (1 - p)))  # noqa: E731
BASE = {"a_rew": -1.0, "a_unrew": -2.0, "forget": -0.5, "beta": 1.5, "a_K": -1.0, "w_K": 0.0, "lam_c": logit_(0.65), "b_c": 0.3,
        "lam_w": logit_(0.7), "b_w": 0.3, "lam_u": logit_(0.9), "b_u": 0.05, "bias": 0.0, "w_ws": 0.0, "w_ls": 0.0}
# Second start for the trace models: slower reward trace, unrewarded trace near-permanent.
ALT = {**BASE, "lam_c": logit_(0.5), "lam_w": logit_(0.8), "lam_u": logit_(0.99), "b_u": 0.02}
AXIS_NAMES = ["determinism / stickiness", "side bias", "slow learning / long choice memory", "dynamics-only"]
REF_COLORS = {"GRU": "#222222", "CTT": "#C44E52", "Hattori": "#55A868", "Bari": "#4C72B0"}


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    s12.apply_presentation_style()
    print("devices", jax.devices(), flush=True)
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "p_model1_right", "subject_session_index"])
    d = session_arrays(t, t["p_model1_right"].to_numpy())
    n_mice, n_s = len(d["mice"]), d["shape"][0]
    choice, reward, mask = (jnp.asarray(d[k]) for k in ("choice", "reward", "mask"))
    mouse = jnp.asarray(d["mouse"])
    # Fold of each session: alternate within mouse, in session order.
    ses_first = t.drop_duplicates("ses_idx")
    order = ses_first.sort_values(["subject_id", "subject_session_index"])
    fold_of = pd.Series(order.groupby("subject_id").cumcount().to_numpy() % 2, index=order["ses_idx"].to_numpy())
    ses_codes = t["ses_idx"].to_numpy()[np.unique(d["col"], return_index=True)[1]]  # session code per column
    fold = jnp.asarray(fold_of.reindex(ses_codes).to_numpy())
    a_ = t["action"].to_numpy()

    def unpack(theta):
        sig, sp = jax.nn.sigmoid, jax.nn.softplus
        th = dict(zip(NAMES, theta.T))
        out = {k: th[k] for k in NAMES}
        for k in ("a_rew", "a_unrew", "forget", "a_K", "lam_c", "lam_w", "lam_u"):
            out[k] = sig(th[k])
        out["beta"] = sp(th["beta"])
        return out

    def policy(theta, spec: str):
        prm = {k: v[mouse] for k, v in unpack(theta).items()}

        def step(carry, inputs):
            ql, qr, k_, tc, tw, tu, pc, pr, started = carry
            c, r = inputs
            s, sp_ = 2 * c - 1, jnp.where(started, 2 * pc - 1, 0.0)
            z = prm["bias"]
            if "q" in spec:
                z = z + prm["beta"] * (qr - ql)
            if "K" in spec:
                z = z + prm["w_K"] * k_
            z = z + ("C" in spec) * prm["b_c"] * tc + ("W" in spec) * prm["b_w"] * tw + ("U" in spec) * prm["b_u"] * tu
            if "L" in spec:
                z = z + sp_ * jnp.where(pr > 0, prm["w_ws"], prm["w_ls"])
            a = jnp.where(r > 0, prm["a_rew"], prm["a_unrew"])
            ql = jnp.where(c == 0, ql + a * (r - ql), (1 - prm["forget"]) * ql)
            qr = jnp.where(c == 1, qr + a * (r - qr), (1 - prm["forget"]) * qr)
            k_ = k_ + prm["a_K"] * (s - k_)
            tc = prm["lam_c"] * tc + s
            tw = prm["lam_w"] * tw + s * r
            tu = prm["lam_u"] * tu + s * (1 - r)
            return (ql, qr, k_, tc, tw, tu, c, r, jnp.ones_like(started)), z

        zeros = jnp.zeros(n_s)
        carry = (zeros,) * 8 + (jnp.zeros(n_s, bool),)
        _, z = jax.lax.scan(step, carry, (choice.T, reward.T))
        return z.T

    def per_session_ll(theta, spec):
        z = policy(theta, spec)
        ll = choice * jax.nn.log_sigmoid(z) + (1 - choice) * jax.nn.log_sigmoid(-z)
        return jnp.sum(ll * mask, axis=1), z

    def fit(spec, init, train_fold):
        w = (fold == train_fold).astype(jnp.float32)
        theta = jnp.tile(jnp.asarray([init[k] for k in NAMES]), (n_mice, 1))
        opt = optax.adam(LR)
        state = opt.init(theta)
        n_tr = jnp.sum(mask.sum(1) * w)

        @jax.jit
        def update(th, st):
            def loss(th_):
                return -jnp.sum(per_session_ll(th_, spec)[0] * w) / n_tr
            val, g = jax.value_and_grad(loss)(th)
            upd, st = opt.update(g, st)
            return optax.apply_updates(th, upd), st, val

        trace = []
        t0 = time.time()
        for i in range(N_STEPS):
            theta, state, val = update(theta, state)
            if i == 0:
                val.block_until_ready()
                t1 = time.time()
            if i % 250 == 0 or i == N_STEPS - 1:
                trace.append(float(val))
        print(f"  {spec} fold {train_fold}: compile+first step {t1 - t0:.1f}s, {(time.time() - t1) / max(N_STEPS - 1, 1):.3f}s/step", flush=True)
        ll_s, z = (np.array(x) for x in jax.jit(lambda th: per_session_ll(th, spec))(theta))
        train_ll = pd.Series(ll_s * np.asarray(w)).groupby(np.asarray(mouse)).sum().reindex(range(n_mice)).to_numpy()
        return theta, train_ll, z, trace

    results, p_held, params, conv = {}, {}, {}, {}
    for name, spec in MODELS.items():
        z_held = np.zeros(d["shape"], np.float32)
        th_folds = []
        for f in (0, 1):
            best = None
            for init in ([BASE, ALT] if set(spec) & set("CWU") else [BASE]):
                theta, train_ll, z, trace = fit(spec, init, f)
                if best is None:
                    best = [np.array(theta), train_ll, np.array(z), trace]
                else:  # keep the better start per mouse (by training likelihood)
                    better = train_ll > best[1]
                    best[0][better] = np.asarray(theta)[better]
                    best[1] = np.where(better, train_ll, best[1])
                    sess_better = better[np.asarray(mouse)]
                    best[2][sess_better] = z[sess_better]
                    best[3] = best[3] + trace
            test_cols = np.asarray(fold) != f
            z_held[test_cols] = best[2][test_cols]
            th_folds.append(best[0])
            conv[f"{name} | fold {f}"] = best[3]
            print(name, "fold", f, "loss trace", [round(x, 5) for x in best[3]], flush=True)
        p = 1 / (1 + np.exp(-z_held[d["col"], d["pos"]].astype(np.float64)))
        p_held[name] = p
        params[name] = {k: np.mean([np.asarray(unpack(jnp.asarray(th))[k]) for th in th_folds], axis=0) for k in NAMES}

    subj = t["subject_id"].to_numpy()
    lp = {name: np.log(np.clip(np.where(a_ == 1, p, 1 - p), 1e-10, 1)) for name, p in p_held.items()}
    ref = {"GRU": t["p_model1_right"].to_numpy(np.float64)}
    for m, lab in (("hattori", "Hattori"), ("bari", "Bari"), ("ctt", "CTT")):
        ref[lab] = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
    lp_ref = {k: np.log(np.clip(np.where(a_ == 1, v, 1 - v), 1e-10, 1)) for k, v in ref.items()}
    mouse_ll = pd.DataFrame({**{n: pd.Series(v).groupby(subj).mean() for n, v in lp.items()},
                             **{f"ref {n}": pd.Series(v).groupby(subj).mean() for n, v in lp_ref.items()}})
    q0 = "Q (Hattori form)"
    for name in MODELS:
        dll = mouse_ll[name] - mouse_ll[q0]
        results[name] = {"heldout_likelihood": float(np.exp(lp[name].mean())),
                         "median_mouse_dll_vs_Q": float(dll.median()),
                         "share_mice_beat_Q": float((dll > 0).mean()),
                         "share_mice_beat_GRU": float((mouse_ll[name] > mouse_ll["ref GRU"]).mean()),
                         "share_mice_beat_Q_kernel_last": float((mouse_ll[name] > mouse_ll["Q + kernel + last"]).mean()),
                         "params_median": {k: float(np.median(v)) for k, v in params[name].items()},
                         "params_iqr": {k: np.percentile(v, [25, 75]).tolist() for k, v in params[name].items()}}
    ref_lik = {k: float(np.exp(v.mean())) for k, v in lp_ref.items()}
    gap = lambda n: (results[n]["heldout_likelihood"] - results[q0]["heldout_likelihood"]) / (ref_lik["GRU"] - results[q0]["heldout_likelihood"])  # noqa: E731
    for n in MODELS:
        results[n]["gap_closed_Q_to_GRU"] = gap(n)
    pairs = [("three traces + last", "two outcomes + last"), ("three traces + last", "choice + reward + last"),
             ("three traces + last", "Q + kernel + last"), ("three traces", "three traces + last"),
             ("two outcomes + last", "choice + reward + last")]
    paired = {}
    for a, b in pairs:
        diff = mouse_ll[a] - mouse_ll[b]
        paired[f"{a} vs {b}"] = {"median_dll_per_trial": float(diff.median()), "share_a_better": float((diff > 0).mean()),
                                 "wilcoxon_p": float(wilcoxon(diff).pvalue)}
    for k, v in results.items():
        print(f"{k:28s} lik {v['heldout_likelihood']:.5f} gap {v['gap_closed_Q_to_GRU']:+.3f} beatQ {v['share_mice_beat_Q']:.2f} "
              f"beatGRU {v['share_mice_beat_GRU']:.2f}", flush=True)
    print("reference", ref_lik)
    print(paired, flush=True)

    # Three-trace parameters: sign and timescales; agreement with v's own kernel (I9d) and the named axes.
    p3 = pd.DataFrame(params["three traces + last"], index=d["mice"])
    v9 = pd.read_parquet(s12.CACHE / f"i9d_mouse_long_{s12.MODEL_KEY}.parquet").query("target == 'GRU v'").set_index("subject_id")
    vmap = {"b_w": "K20 | three traces | p0", "lam_w": "K20 | three traces | p1", "b_u": "K20 | three traces | p2",
            "lam_u": "K20 | three traces | p3", "b_c": "K20 | three traces | p4", "lam_c": "K20 | three traces | p5"}
    common = p3.index.intersection(v9.index)
    vs_v = {k: float(spearmanr(p3.loc[common, k], v9.loc[common, c])[0]) for k, c in vmap.items()}
    axes_df = pd.read_parquet(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.parquet")
    axes_df.columns = AXIS_NAMES
    ca = p3.index.intersection(axes_df.index)
    keys = ["b_c", "lam_c", "b_w", "lam_w", "b_u", "lam_u", "w_ws", "w_ls"]
    corr = pd.DataFrame({ax: [spearmanr(p3.loc[ca, k], axes_df.loc[ca, ax])[0] for k in keys] for ax in AXIS_NAMES}, index=keys)
    shape = {"share_b_u_pos": float((p3["b_u"] > 0).mean()), "share_lam_u_gt_lam_w": float((p3["lam_u"] > p3["lam_w"]).mean()),
             "share_lam_u_gt_0.95": float((p3["lam_u"] > 0.95).mean()), "share_b_w_pos": float((p3["b_w"] > 0).mean()),
             "share_b_c_pos": float((p3["b_c"] > 0).mean())}
    print("three-trace shape", shape, "\nvs v-fit (Spearman)", vs_v, "\n", corr.round(2).to_string(), flush=True)

    # C4e signatures on held-out predictions.
    show = ["Q (Hattori form)", "Q + kernel + last", "two outcomes + last", "three traces + last"]
    sig_p = {"GRU": ref["GRU"], **{n: p_held[n] for n in show}}
    row1, row2 = signature_curves(t["ses_idx"].to_numpy(), a_.astype(int), t["reward"].to_numpy() > 0, sig_p)
    sig_curves = {**{f"return, {k}": v for k, v in row1.items()}, **{f"stay, {k}": v for k, v in row2.items()}}
    sig_err = {lab: {m: float(np.average(np.abs(dd[m] - dd["animal"]), weights=dd["n"])) for m in sig_p} for lab, dd in sig_curves.items()}

    np.savez_compressed(s12.CACHE / f"i9e_p_right_{s12.MODEL_KEY}.npz", **{f"m{i}": p_held[n] for i, n in enumerate(MODELS)})
    pd.DataFrame([{"model": n, "subject_id": m, **{k: float(v[i]) for k, v in params[n].items()}}
                  for n in MODELS for i, m in enumerate(d["mice"])]).to_parquet(s12.CACHE / f"i9e_params_{s12.MODEL_KEY}.parquet")
    mouse_ll.to_parquet(s12.CACHE / f"i9e_mouse_ll_{s12.MODEL_KEY}.parquet")

    # Figure.
    names = list(MODELS)
    colors = {n: ("#999999" if n.startswith("Q") else "#CC79A7" if n.startswith("choice") else "#000000" if n.startswith("two")
                  else "#009E73") for n in names}
    fig, axs = plt.subplots(2, 4, figsize=(34, 14))
    ax = axs[0, 0]
    y = np.arange(len(names))
    lik = [results[n]["heldout_likelihood"] for n in names]
    ax.barh(y, lik, color=[colors[n] for n in names])
    for k, c in REF_COLORS.items():
        ax.axvline(ref_lik[k], color=c, ls="--" if k == "GRU" else ":", lw=1.8, label=f"{k} (wrapper fit) {ref_lik[k]:.4f}")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    lo = min(lik + list(ref_lik.values()))
    ax.set_xlim(lo - 0.002, max(lik + list(ref_lik.values())) + 0.002)
    for i, v in enumerate(lik):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=10)
    ax.set_xlabel("held-out likelihood on the animals' choices (geometric mean per trial)")
    ax.set_title("Which update rule predicts the mice?", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax = axs[0, 1]
    for i, n in enumerate(names):
        dll = (mouse_ll[n] - mouse_ll[q0]).to_numpy()
        ax.boxplot(dll, positions=[i], widths=0.6, orientation="horizontal", showfliers=False, patch_artist=True,
                   boxprops={"facecolor": colors[n], "alpha": 0.5})
    gdll = (mouse_ll["ref GRU"] - mouse_ll[q0]).to_numpy()
    ax.axvline(np.median(gdll), color="#222222", ls="--", lw=1.5, label=f"GRU, median {np.median(gdll):+.4f}")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("per-mouse Δ log-likelihood per trial vs Q (Hattori form, refit)")
    ax.set_title("Per mouse, against refitted Q-learning", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[0, 2]
    alld = np.concatenate([(mouse_ll[a] - mouse_ll[b]).to_numpy() for a, b in pairs[:3]])
    lo_hi = np.percentile(alld, [0.5, 99.5])  # one mouse (1.5k trials) has a diverged two-outcome fit at +0.26
    for (a, b), c in zip(pairs[:3], ("#000000", "#CC79A7", "#999999")):
        diff = (mouse_ll[a] - mouse_ll[b]).to_numpy()
        ax.hist(np.clip(diff, *lo_hi), bins=np.linspace(*lo_hi, 70), histtype="step", lw=2, color=c,
                label=f"− {b}: median {np.median(diff):+.4f}, better in {100 * np.mean(diff > 0):.0f}%")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Δ log-likelihood per trial, three traces + last minus … (clipped to the 0.5–99.5th percentile)")
    ax.set_ylabel("mice")
    ax.set_title("Does the third trace earn its place?", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax = axs[0, 3]
    kmax = 20
    lags = np.arange(1, kmax + 1)
    m3 = results["three traces + last"]["params_median"]
    rew = m3["b_c"] * m3["lam_c"] ** (lags - 1) + m3["b_w"] * m3["lam_w"] ** (lags - 1)
    unr = m3["b_c"] * m3["lam_c"] ** (lags - 1) + m3["b_u"] * m3["lam_u"] ** (lags - 1)
    rew[0] += m3["w_ws"]
    unr[0] += m3["w_ls"]
    ax.plot(lags, rew, color="#009E73", marker="o", lw=2.4, label="rewarded choice k back")
    ax.plot(lags, unr, color="#009E73", marker="s", ls="--", lw=2.4, label="unrewarded choice k back")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("trials back")
    ax.set_ylabel("weight on logit P(that side) (median parameters)")
    ax.set_title("Three traces + last: the implied choice kernels (behaviour)", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 0]
    ax.scatter(p3["lam_w"], p3["lam_u"], s=10, alpha=0.5, color="#009E73",
               label=f"λ_u > λ_w in {100 * shape['share_lam_u_gt_lam_w']:.0f}% of mice")
    ax.scatter(p3["lam_w"], p3["lam_c"], s=10, alpha=0.5, color="#E69F00", label="λ_c (choice trace)")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":")
    ax.set_xlabel("λ_w (reward trace)")
    ax.set_ylabel("λ_u (unrewarded trace), λ_c (choice trace)")
    ax.set_title("Per-mouse timescales from behaviour", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 1]
    for k, c in (("b_c", "#E69F00"), ("b_w", "#0072B2"), ("b_u", "#009E73")):
        ax.hist(p3[k], bins=60, histtype="step", lw=2, color=c, label=f"{k}: median {p3[k].median():+.3f}, > 0 in {100 * (p3[k] > 0).mean():.0f}%")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("trace weight β")
    ax.set_ylabel("mice")
    ax.set_title("Per-mouse trace weights (does the unrewarded trace push towards the side?)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1, 2]
    im = ax.imshow(corr.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(AXIS_NAMES)), AXIS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys)), [f"{k}  (vs v-fit ρ {vs_v[k]:+.2f})" if k in vs_v else k for k in keys], fontsize=10)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Behavioural three-trace parameters × named embedding axes", fontsize=12)
    ax = axs[1, 3]
    labs = list(sig_err)
    xx = np.arange(len(labs))
    for j, m in enumerate(sig_p):
        ax.bar(xx + (j - 2) * 0.16, [sig_err[l_][m] for l_ in labs], 0.16, label=m,
               color="#222222" if m == "GRU" else colors[m])
    ax.set_xticks(xx, labs, rotation=15, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("mean |model − animals| on the C4e curve")
    ax.set_title("C4e signatures (held-out predictions)", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I9e_trace_models.png"))
    plt.close(fig)

    s12.write_json("i9e_trace_models.json", {
        "models": MODELS, "results": results, "reference_likelihood_wrapper_fits": ref_lik, "paired_per_mouse": paired,
        "three_traces_last_shape": shape, "three_traces_last_vs_v_fit_spearman": vs_v,
        "three_traces_last_vs_named_axes_spearman": corr.to_dict(), "signature_errors": sig_err,
        "fit": {"target": "animals' choices", "cv": "2-fold by alternating sessions within mouse", "optimizer": "adam",
                "lr": LR, "steps": N_STEPS, "starts": "trace models: best of 2 starts per mouse by training likelihood",
                "init": BASE, "alt_init": ALT}, "loss_traces": conv, "n_mice": n_mice,
    }, "analysis/i9e_trace_models.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
