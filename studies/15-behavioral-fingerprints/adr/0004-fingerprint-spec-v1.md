# Fingerprint spec v1: what is binned, how bins are named, and what is stored

The spec has a version, `fingerprints-v1`, written into every output. Changing a key, a cap or a
statistic makes a new version; tallies made under an old version keep its label.

**Bins.** A past trial is coded relative to the last choice: same side or other side, rewarded or
not (for example `S+ O− S−` for the last three trials). Run lengths and loss counts are capped at 1
to 5, then 6 or more. This is MEvolve's form. The wrapper's "abstract" code names a pattern by its
oldest trial instead; both identify a pattern with its mirror image, so they give the same 32 bins
at 3 trials back, and the golden test maps one set of labels onto the other.

**Statistics in v1.**

| family | statistics |
|---|---|
| local switching | p(switch) by the last 1 to 3 trials; by consecutive losses on the current side; by last reward × current run length (the hazard); after a switch, by reward and the run before it (study 01 r9's headline) |
| sequence shape | run-length distributions, tallied as counts of completed runs per length (1 to 5, 6–10, 11–20, 21–40, 41 or more; a session's last run is incomplete and not counted) (rollout); replay tests them through the hazard, which determines them |

Each is also split by stage group × task variant. Only engaged trials count; in replay, p(left)
and p(right) are renormalised over the engaged choices. Ignore statistics come in v2, with the
3-way models.

**Warm-up sessions are their own group.** A session is a warm-up when, on every trial, one side's
reward probability is 0, or when its probabilities are only 0 and 1 (older Bpod sessions). One side
always pays and the other never does, so there is nothing to forage and the switch statistics
measure task learning. Warm-ups get their own group in every split and are left out of the
headline grid. They are mostly a subject's first sessions: 86 of the test subjects' score
sessions, 58 of them among the first three (r1). Decided with Han, 2026-11-03.

**No regression kernels.** A fitted kernel has no bins and cannot be rebuilt from tallies, and a
GLM with the same design matches it by construction. If longer history (4 to 10 trials back) is
wanted, a later version adds it as a binned statistic, for example p(switch) by the rewards in the
last 10 trials (decided with Han, 2026-11-03).

**No block-adaptation family.** Reversal curves, matching (choice ratio against reward ratio) and
foraging efficiency are left out (decided with Han, 2026-11-03).

**Stored.** For each model, regime, subject and session partition, the library writes each bin's
tally (trials, summed prediction, summed outcome), not only summaries. Any aggregation, scatter or
later statistic can then be recomputed offline. Study 01 r9 shows why: its RL scatter could not be
redrawn from the summaries that had been committed. The spec also fixes which bins a subject keeps,
so the gap and its noise floor (ADR 0003) are computed over the same bins.
