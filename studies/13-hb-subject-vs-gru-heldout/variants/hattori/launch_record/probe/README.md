# Route probe, 2026-10-25 (diagnostic, not a launch)

Why the batched HPC smoke (`hattori@20261025-210635`) ran 17–50 s per warmup iteration, and
which route production should take. Standalone SLURM jobs, nothing logged to W&B; the code and
raw results are beside this file. foraging-models `ef1d31a`.

| job | what | result (`*_results.jsonl`) |
|---|---|---|
| 27500429 (A100) | 20 simulated subjects in ONE sampler, 50 warmup + 20 draws: {fixed wide hyperprior, tight population-style prior} × {1, 4 chains} | wide prior 4.55 s/iter, median 255 leapfrog steps; tight prior 1.28 s/iter, 31 steps; 4 vectorized chains cost ~5× per step |
| 27500455 (CPU, 4 cores) | 3 simulated subjects, ONE subject per sampler, 4 chains, 200 warmup + 100 draws | vectorized chains 0.23–0.35 s/iter, median 31–63 steps |

**Decision.** One shared sampler pays the hardest subject's step size for every subject; one
subject per sampler on CPU is ~100× faster per iteration. Production runs one held-out
subject per SLURM CPU array task (`sweep_hpc_per_subject.yaml`), pooled via
`heldout/eval_log_lik_sum` (wrapper#109).

**Limits.** Simulated data at real-cohort shape (5 context sessions × 500 trials), not the
snapshot. The tight prior is foraging-models' test `POPULATION`, a stand-in for a fitted
population, not study 08's values. `chain_method="parallel"` CPU timings (~4 s per 300
iterations) returned before sampling finished and are discarded. Warmup was short, so step
sizes had not settled; timings include compilation.
