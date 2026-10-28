"""E2 (stage A): do side and engagement live in separable subspaces? Mean-replacement ablations.

    python analysis/e2_ablations.py

Re-runs C3 (wrapper ``gru_step``, each session's own context) on N_SESSIONS random eval
sessions with the state's component along a subspace replaced by the all-trials mean at
every step. Subspaces: the side direction (w_R − w_L), the engage direction
((w_L + w_R)/2 − w_ignore), both (2-d), and N_RANDOM random 1-d directions as controls. Two
modes: ``full`` (the ablated state is read out and carried) and ``memory`` (read out from the
intact state, only the carried state ablated — does the subspace carry information forward?).
Scores on those sessions: L/R likelihood on engaged trials, engage-vs-ignore likelihood on
every step, and the 3-way likelihood. A dissociation — engage ablation costing engagement
but not side, and vice versa — means the two are carried separately.
Writes ``e2_ablations.json`` and ``fig_E2_ablations.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

N_SESSIONS = 1500
N_RANDOM = 5
SEED = 0
PARITY_TOLERANCE_MEDIAN = 5e-3


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s13.apply_presentation_style()
    split = load_gru_split(s13.CACHE / "models" / s13.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    ctx = np.load(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz")
    if list(ctx["session_ids"]) != split.session_ids:
        raise AssertionError("Cached contexts are not in the split's session order (rerun d_dynamics.py)")
    rng = np.random.default_rng(SEED)
    cols = np.sort(rng.choice(len(split.session_ids), size=N_SESSIONS, replace=False))
    obs_s = jnp.asarray(obs[:, cols], dtype=jnp.float32)
    c_s = jnp.broadcast_to(jnp.asarray(ctx["contexts"][cols], dtype=jnp.float32), obs_s.shape[:2] + ctx["contexts"].shape[1:])
    n_valid = split.n_trials[cols]
    T = obs.shape[0]
    valid = np.arange(T)[:, None] < n_valid[None, :]
    # Targets: row τ predicts trial τ, whose response is row τ+1's "previous choice" input; the
    # last trial's response is not in xs, so it is not scored.
    choice = np.full((T, N_SESSIONS), -1.0)
    choice[:-1] = np.asarray(obs[1:, cols, 0])
    scored = valid & (np.arange(T)[:, None] + 1 < n_valid[None, :])

    w, b = jnp.asarray(step.readout_w), jnp.asarray(step.readout_b)
    mean = jnp.asarray(s13.load_pca()["mean"], dtype=jnp.float32)
    dirs = s13.readout_directions()
    unit = {k: v / np.linalg.norm(v) for k, v in dirs.items()}
    subspaces = {"side": unit["side"][:, None], "engage": unit["engage"][:, None],
                 "side + engage": np.linalg.qr(np.stack([unit["side"], unit["engage"]], axis=1))[0]}
    for i in range(N_RANDOM):
        r = rng.normal(size=s13.HIDDEN)
        subspaces[f"random {i + 1}"] = (r / np.linalg.norm(r))[:, None]

    def run(basis, memory_only):
        basis = jnp.asarray(basis, dtype=jnp.float32)

        def ablate(h):
            return h - ((h - mean) @ basis) @ basis.T

        def body(h, inputs):
            h_new = step.step(h, *inputs)
            h_abl = ablate(h_new) if basis.shape[1] else h_new
            logits = (h_new if memory_only else h_abl) @ w + b
            return h_abl, logits

        h0 = jnp.zeros((N_SESSIONS, step.hidden_size), dtype=jnp.float32)
        return np.asarray(jax.jit(lambda o, c: jax.lax.scan(body, h0, (o, c))[1])(obs_s, c_s), dtype=np.float64)

    def score(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        ch = choice.astype(int)
        engaged = scored & (ch >= 0) & (ch < 2)
        ignored = scored & (ch == 2)
        lr = np.where(ch == 1, logp[..., 1], logp[..., 0]) - np.logaddexp(logp[..., 0], logp[..., 1])
        p_eng = np.logaddexp(logp[..., 0], logp[..., 1])
        eng = np.where(ignored, logp[..., 2], p_eng)
        three = np.take_along_axis(logp, np.clip(ch, 0, 2)[..., None], axis=-1)[..., 0]
        m = engaged | ignored
        return {"LR_engaged": float(np.exp(lr[engaged].mean())), "engage": float(np.exp(eng[m].mean())),
                "three_way": float(np.exp(three[m].mean())), "n_engaged": int(engaged.sum()), "n_steps": int(m.sum())}

    intact = run(np.zeros((s13.HIDDEN, 0)), False)
    # Parity: intact logits vs A1 on these sessions.
    trials = s13.load_trials(["ses_idx", "trial_idx", "logit_diff"], sessions=[split.session_ids[c] for c in cols])
    col_of = {split.session_ids[c]: i for i, c in enumerate(cols)}
    ii = trials["ses_idx"].map(col_of).to_numpy()
    tt = trials["trial_idx"].to_numpy() - 1
    err = np.abs((intact[..., 1] - intact[..., 0])[tt, ii] - trials["logit_diff"].to_numpy())
    if np.median(err) > PARITY_TOLERANCE_MEDIAN:
        raise AssertionError(f"Intact replay parity failed: median {np.median(err)}")

    results = {"intact": score(intact)}
    for name, basis in subspaces.items():
        for mode in ("full", "memory"):
            results[f"{name} | {mode}"] = score(run(basis, mode == "memory"))
        print(name, {m: {k: round(v, 4) for k, v in results[f'{name} | {m}'].items() if not k.startswith('n_')} for m in ("full", "memory")})

    def cost(key, metric):
        return results["intact"][metric] - results[key][metric]

    rand_cost = {mode: {metric: [cost(f"random {i + 1} | {mode}", metric) for i in range(N_RANDOM)]
                        for metric in ("LR_engaged", "engage")} for mode in ("full", "memory")}
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=False)
    for ax, mode in zip(axes, ("full", "memory")):
        names = ["side", "engage", "side + engage"]
        x = np.arange(len(names))
        for k, (metric, color) in enumerate((("LR_engaged", s13.MODEL_COLOR), ("engage", s13.ENGAGE_COLOR))):
            ax.bar(x + (k - 0.5) * 0.38, [cost(f"{n} | {mode}", metric) for n in names], 0.38, color=color,
                   label={"LR_engaged": "loss in L/R likelihood (engaged)", "engage": "loss in engage likelihood"}[metric])
            r = rand_cost[mode][metric]
            ax.errorbar(len(names) + (k - 0.5) * 0.38, np.mean(r), yerr=[[np.mean(r) - min(r)], [max(r) - np.mean(r)]],
                        fmt="o", color=color, capsize=4)
        ax.set_xticks(list(x) + [len(names)], names + [f"random 1-d\n(mean, range of {N_RANDOM})"])
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set(ylabel="likelihood lost vs intact", title={
            "full": "ablated state read out and carried", "memory": "only the carried state ablated (memory)"}[mode])
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=2)
    fig.suptitle(f"Mean-replacement ablations of C3 ({N_SESSIONS:,} eval sessions; intact L/R "
                 f"{results['intact']['LR_engaged']:.4f}, engage {results['intact']['engage']:.4f})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(s13.figure_path("fig_E2_ablations.png"))
    plt.close(fig)

    s13.write_json("e2_ablations.json", {
        "n_sessions": N_SESSIONS, "seed": SEED, "n_random": N_RANDOM,
        "replacement": "component along the subspace set to that of the all-trials mean state, every step",
        "parity_median_abs_err": float(np.median(err)), "results": results,
        "cost_vs_intact": {k: {m: results["intact"][m] - v[m] for m in ("LR_engaged", "engage", "three_way")}
                           for k, v in results.items() if k != "intact"},
        "cosine_side_engage": float(unit["side"] @ unit["engage"]),
    }, "analysis/e2_ablations.py")


if __name__ == "__main__":
    main()
