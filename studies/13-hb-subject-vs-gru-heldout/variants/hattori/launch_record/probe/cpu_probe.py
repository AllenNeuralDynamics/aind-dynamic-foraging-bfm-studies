"""Study 13 diagnostic: one subject per sampler on CPU, 4 chains on 4 cores."""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import json, sys, time
import numpy as np, jax
from numpyro.infer import MCMC, NUTS
sys.path[:0] = ["/home/han.hou/code/wt/hpc13/models-src/src", "/home/han.hou/code/wt/hpc13/models-src"]
from tests.test_hb_heldout_batched import _simulate_cohort
from aind_dynamic_foraging_models.hierarchical_bayes.heldout import FIXED_HYPERPRIOR, adapt_subjects_batched

c, r = _simulate_cohort(3, 5, 500, seed=1)
out = []
for s in range(3):
    for method in ("parallel", "vectorized"):
        mcmc = MCMC(NUTS(adapt_subjects_batched), num_warmup=200, num_samples=100, num_chains=4,
                    chain_method=method, progress_bar=False)
        t = time.time()
        mcmc.run(jax.random.PRNGKey(s), c[s:s + 1], r[s:s + 1], FIXED_HYPERPRIOR, extra_fields=("num_steps", "diverging"))
        sec = time.time() - t
        steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
        row = dict(subject=s, chain_method=method, seconds=round(sec, 1), s_per_iter=round(sec / 300, 3),
                   median_num_steps=float(np.median(steps)), max_num_steps=int(steps.max()),
                   divergences=int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"]))))
        print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open("cpu_probe.json", "w"), indent=1)
