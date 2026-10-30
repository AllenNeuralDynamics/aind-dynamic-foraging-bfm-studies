"""I10: how the GRU computes v — retention along v's axis from the gates.

    PYTHONPATH=<wrapper>/code python analysis/i10_gates.py

The Haiku GRU update is h' = (1 − z) ⊙ h + z ⊙ a, with update gate z = σ(·), reset gate r and
candidate a = tanh(W_a x + U_a (r ⊙ h) + b_a). Along v's unit axis e (I2 value axis), a small
change δ·e of the state changes the next state by J e, and
    J e = (1 − z) ⊙ e  +  (a − h) ⊙ (∂z·e)  +  z ⊙ (∂a·e)
          leak            gate change          candidate (recurrent drive, via r and U_a)
so v's one-step retention eᵀ J e splits into three terms. Each is computed exactly (JAX JVPs)
on N real transitions per group (trial t's input × stay / switch, as I3), at each state's own
session context, and compared with I3/I4's fitted slope λ of v(t+1) on v(t). Also: which units
carry v (loadings e_i², participation ratio), and their update-gate values.
Writes ``i10_gates.json`` and ``fig_I10_gates.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
import _transitions as tr  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

N_PER_GROUP = 20_000
SEED = 18
GROUPS = [g for g, n in enumerate(tr.GROUP_NAMES) if not n.endswith("(first)")]
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
N_BINS = 10


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s12.apply_presentation_style()
    params = {k: {n: np.asarray(a, np.float64) for n, a in v.items()} for k, v in s12.best_params().items() if isinstance(v, dict)}
    step = GruStep.from_params(params)
    n = step.hidden_size
    w_h_z, w_h_a = np.split(np.asarray(step.w_h), [2 * n], axis=1)
    b_z, b_a = np.split(np.asarray(step.b), [2 * n])
    w_i = np.asarray(step.w_i)

    def parts(h, obs, c):
        x = jnp.concatenate([obs, c], axis=-1)
        zr_x, a_x = jnp.split(x @ w_i, [2 * n], axis=-1)
        z, r = jnp.split(jax.nn.sigmoid(zr_x + h @ w_h_z + b_z), 2, axis=-1)
        a = jnp.tanh(a_x + (r * h) @ w_h_a + b_a)
        return z, r, a

    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    e = tdr["basis"][:, [str(x) for x in tdr["terms"]].index("value")].astype(np.float64)
    e_j = jnp.asarray(e)

    @jax.jit
    def analyse(h, obs, c):
        (z, r, a), (dz, dr, da) = jax.jvp(lambda hh: parts(hh, obs, c), (h,), (jnp.broadcast_to(e_j, h.shape),))
        h_next = (1 - z) * h + z * a
        leak = ((1 - z) * e_j) @ e_j
        gate = ((a - h) * dz) @ e_j
        cand = (z * da) @ e_j
        return h_next, z, r, leak, gate, cand

    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward"])
    lay = tr.session_layout(t)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    ctx_of = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"])
    contexts = ctx["contexts"].astype(np.float64)
    v_all = value_coordinate()
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(np.float32)
    pca_mean = s12.load_pca()["mean"]
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(lay["has_next"] & ~lay["first"])
    res, unit_z = {}, {}
    parity = []
    for g in GROUPS:
        rows = idx[lay["group"][idx] == g]
        rows = np.sort(rng.choice(rows, size=min(N_PER_GROUP, len(rows)), replace=False))
        h = jnp.asarray(states[rows], jnp.float64)
        obs = jnp.asarray(np.column_stack([t["action"].to_numpy()[rows], (t["reward"].to_numpy()[rows] > 0)]).astype(np.float64))
        c = jnp.asarray(contexts[ctx_of.reindex(t["ses_idx"].to_numpy()[rows]).to_numpy()])
        h_next, z, r, leak, gate, cand = (np.asarray(x) for x in analyse(h, obs, c))
        parity.append(float(np.median(np.abs(h_next - states[rows + 1]))))
        v0, v1 = v_all[rows], v_all[rows + 1]
        slope = float(np.polyfit(v0, v1, 1)[0])
        total = leak + gate + cand
        edges = np.percentile(v0, np.linspace(0, 100, N_BINS + 1))
        b = np.clip(np.digitize(v0, edges) - 1, 0, N_BINS - 1)
        res[tr.GROUP_NAMES[g]] = {
            "n": int(len(rows)), "fitted_slope": slope, "eJe_mean": float(total.mean()), "eJe_median": float(np.median(total)),
            "leak": float(leak.mean()), "gate_change": float(gate.mean()), "candidate": float(cand.mean()),
            "z_along_v": float(((z * e ** 2).sum(1)).mean()), "z_mean_all_units": float(z.mean()),
            "eJe_by_v_decile": {"v": pd.Series(v0).groupby(b).mean().tolist(), "eJe": pd.Series(total).groupby(b).mean().tolist()},
        }
        unit_z[tr.GROUP_NAMES[g]] = z.mean(0)
        print(tr.GROUP_NAMES[g], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[tr.GROUP_NAMES[g]].items() if k != "eJe_by_v_decile"}, flush=True)
    print("one-step parity vs stored states (median |Δh|)", np.round(parity, 5), flush=True)

    load = e ** 2
    pr = float(1 / np.sum(load ** 2))
    order = np.argsort(-load)
    top = order[:10]
    names = [tr.GROUP_NAMES[g] for g in GROUPS]
    fig, axes = plt.subplots(1, 4, figsize=(32, 7.5), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    w = 0.16
    for j, (key, color, lab) in enumerate((("fitted_slope", "#222222", "fitted slope λ (v(t+1) on v(t))"),
                                           ("eJe_mean", "#4C72B0", "eᵀJe (exact one-step retention)"),
                                           ("leak", "#55A868", "leak: (1 − z) along v"),
                                           ("gate_change", "#C44E52", "gate change: (a − h)·∂z"),
                                           ("candidate", "#E69F00", "candidate: z·∂a (recurrent drive)"))):
        ax.barh(y + (j - 2) * w, [res[nm][key] for nm in names], w, color=color, label=lab)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y, names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("retention of a perturbation along v (one step)")
    ax.set_title("What sets v's retention? eᵀJe = leak + gate change + candidate", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax = axes[1]
    for nm in names:
        d = res[nm]["eJe_by_v_decile"]
        base = nm.split(" (")[0]
        ax.plot(d["v"], d["eJe"], color=COLORS[base], ls="-" if "(stay)" in nm else "--", marker="o", ms=4, lw=2, label=nm)
    ax.set_xlabel("v at trial t (decile means)")
    ax.set_ylabel("eᵀJe (one-step retention along v)")
    ax.set_title("Is the retention state-dependent?", fontsize=13)
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    ax.bar(range(30), load[order[:30]], color="#4C72B0")
    ax.set_xlabel("units, sorted by loading on v's axis")
    ax.set_ylabel("e_i² (share of the axis)")
    ax.set_title(f"Which units carry v? participation ratio {pr:.1f} of {n}; top 10 hold {load[top].sum():.0%}", fontsize=12)
    ax = axes[3]
    for nm in names:
        base = nm.split(" (")[0]
        ax.scatter(load, unit_z[nm], s=10, alpha=0.5, color=COLORS[base], marker="o" if "(stay)" in nm else "x")
    for nm, ls in (("R rewarded (stay)", "-"), ("R rewarded (switch)", "--")):
        ax.axhline(res[nm]["z_along_v"], color="0.3", ls=ls, lw=1.2, label=f"{nm}: z̄ along v {res[nm]['z_along_v']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("unit loading on v's axis e_i² (log)")
    ax.set_ylabel("mean update gate z of the unit")
    ax.set_title("Update gates of v's units (o stay, x switch)", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I10_gates.png"))
    plt.close(fig)

    s12.write_json("i10_gates.json", {
        "groups": res, "participation_ratio_of_v_axis": pr, "top10_units": top.tolist(), "top10_share": float(load[top].sum()),
        "one_step_parity_median": parity, "n_per_group": N_PER_GROUP, "seed": SEED,
        "decomposition": "e^T J e = e^T[(1−z)⊙e] + e^T[(a−h)⊙(∂z e)] + e^T[z⊙(∂a e)], exact JVPs, float64",
    }, "analysis/i10_gates.py")


if __name__ == "__main__":
    main()
