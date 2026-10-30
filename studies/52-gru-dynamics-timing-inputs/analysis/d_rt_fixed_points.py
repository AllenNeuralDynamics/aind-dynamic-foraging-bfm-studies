"""D-RT (stage A): each input's attractor as a function of the RT input.

    JAX_PLATFORMS=cpu PYTHONPATH=<wrapper>/code python analysis/d_rt_fixed_points.py R-s0

Study 50 (D, I3) found one stable fixed point per (choice, reward) input on the chosen
side. Here the input is (choice, reward, RT = v) held constant, for v in RT_VALUES (input
units: sd of log RT), at the model's mean session context, from N_STARTS real states. The
wrapper's ``find_fixed_points`` (forward iteration + damped Newton, float64) finds the
fixed points; stable ones (spectral radius < 1) are kept. For each: its switch-logit (logit
toward the side NOT in the input; very negative = a deep "stay" attractor) and the leading
Jacobian eigenvalue moduli, with τ = −1/ln|λ₁| the slowest relaxation time in trials.
A constant input is an idealisation (the same outcome and RT every trial); the attractor
is where the state heads under that input, not a state the animal visits.

Writes ``d_rt_fixed_points_<key>.json``; ``fig_d_rt_fixed_points.py`` draws both seeds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import CACHE, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/d_rt_fixed_points.py"
RT_VALUES = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
INPUTS = {"R rewarded": (1, 1), "R unrewarded": (1, 0), "L rewarded": (0, 1), "L unrewarded": (0, 0)}
N_STARTS, N_FORWARD, DEDUP = 300, 600, 1e-3
SEED = 4
HIDDEN = 128


def best_params(key: str) -> dict:
    outputs = CACHE / "models" / key / "run" / "outputs"
    index = json.loads(_pk03._read_cache(outputs / "checkpoints" / "index.json").read_text())
    best = max(index["checkpoints"], key=lambda c: c["eval_likelihood"])
    return json.loads(_pk03._read_cache(outputs / "checkpoints" / f"step_{best['step']}" / "params.json").read_text())


def main(key: str, values=None) -> None:
    _pk03.use_keys(key)
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep, find_fixed_points  # noqa: PLC0415

    step = GruStep.from_params({k: {n: np.asarray(a, dtype=np.float64) for n, a in v.items()}
                                for k, v in best_params(key).items() if isinstance(v, dict)})
    c = jnp.asarray(np.load(_pk03._read_cache(CACHE / f"contexts_{key}.npz"))["contexts"].mean(axis=0), dtype=jnp.float64)
    sessions = pd.read_parquet(_pk03._read_cache(CACHE / f"trial_states_{key}.parquet"), columns=["ses_idx"])["ses_idx"].unique()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(sessions, 60, replace=False).tolist()
    states = pq.read_table(_pk03._read_cache(CACHE / f"trial_states_{key}.parquet"), columns=[f"rnn_state_{i}" for i in range(HIDDEN)],
                           filters=[("ses_idx", "in", pick)]).to_pandas().to_numpy(dtype=np.float64)
    starts = states[rng.choice(len(states), N_STARTS, replace=False)]
    out = {}
    for name, (ch, rw) in INPUTS.items():
        away = -1.0 if ch == 1 else 1.0
        for v in (values or RT_VALUES):
            t0 = time.time()
            u = jnp.asarray([float(ch), float(rw), float(v)])
            found, q = find_fixed_points(step, starts, u, c, n_forward=N_FORWARD, dedup=DEDUP)
            points = []
            for f in found:
                lam = np.sort(np.abs(np.linalg.eigvals(np.asarray(step.jacobian(jnp.asarray(f), u, c)))))[::-1]
                if lam[0] < 1:
                    points.append({"switch_logit": float(step.logit_diff(jnp.asarray(f))) * away,
                                   "eig_moduli_top3": lam[:3].tolist(), "tau_trials": float(-1 / np.log(lam[0])),
                                   "state": f.tolist()})
            out[f"{name} | {v:+.1f}"] = {"input": name, "rt": v, "n_found": len(found), "stable": points,
                                         "frac_starts_converged": float(np.mean(q < 1e-12))}
            print(name, v, len(found), [round(p["switch_logit"], 3) for p in points],
                  [round(p["tau_trials"], 2) for p in points], f"{time.time() - t0:.0f}s", flush=True)
    write_json(f"d_rt_fixed_points_{key}.json", {"key": key, "rt_values": values or RT_VALUES, "n_starts": N_STARTS,
                                                  "fixed_points": out}, PRODUCER)


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]] or None)
