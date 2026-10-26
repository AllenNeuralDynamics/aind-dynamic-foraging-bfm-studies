"""Study 13 diagnostic: why is the held-out two-level fit slow? Prior x chains, 20 subjects."""
import json, sys, time
import numpy as np, jax
from numpyro.infer import MCMC, NUTS
sys.path[:0] = ["/home/han.hou/code/wt/hpc13/models-src/src", "/home/han.hou/code/wt/hpc13/models-src"]
from tests.test_hb_heldout_batched import _simulate_cohort, POPULATION
from aind_dynamic_foraging_models.hierarchical_bayes.heldout import FIXED_HYPERPRIOR, adapt_subjects_batched

c, r = _simulate_cohort(20, 5, 500, seed=1)
out = []
for prior_name, prior in (("fixed_wide", FIXED_HYPERPRIOR), ("population_tight", POPULATION)):
    for chains in (1, 4):
        mcmc = MCMC(NUTS(adapt_subjects_batched), num_warmup=50, num_samples=20, num_chains=chains,
                    chain_method="vectorized", progress_bar=False)
        t = time.time()
        mcmc.run(jax.random.PRNGKey(0), c, r, prior, extra_fields=("num_steps", "diverging", "adapt_state.step_size"))
        sec = time.time() - t
        ex = mcmc.get_extra_fields()
        steps = np.asarray(ex["num_steps"])
        row = dict(prior=prior_name, chains=chains, seconds=round(sec, 1), s_per_iter=round(sec / 70, 2),
                   median_num_steps=float(np.median(steps)), max_num_steps=int(steps.max()),
                   frac_saturated=float(np.mean(steps >= 1023)),
                   step_size=float(np.median(np.asarray(ex["adapt_state.step_size"]))),
                   divergences=int(np.sum(np.asarray(ex["diverging"]))))
        print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open("/home/han.hou/code/wt/hpc13/probe/geometry_probe.json", "w"), indent=1)
