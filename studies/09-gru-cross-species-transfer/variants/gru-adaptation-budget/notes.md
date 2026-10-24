# GRU adaptation-budget transfer (dispatcher #166)

How does transfer change when the new-subject embedding sees only K target adaptation
sessions? Same frozen H=128 E4 GRU core as the matched-half headline cell (Study 01
`D=614`, seeds 0--2, `../../source_runs.json`), same 500-step embedding fine-tune, same
immutable even-position test sessions. Only the adaptation budget changes: the first K
sessions of each subject's chronological odd-position adaptation sequence
(`external_transfer.gru_transfer --adapt-sessions-per-subject K`, wrapper #105). K=0 is
zero-shot (no gradient steps); `half` is the full adaptation half, rerun here so every K
shares one code version.

| sweep | dataset | K | tasks |
|---|---|---|---|
| `sweep_k_grossman.yaml` | Grossman (mouse) | 0, 1, 2, 4, 8 | 15 |
| `sweep_k_chen.yaml` | Chen (mouse) | 0, 1, 2 | 9 |
| `sweep_half.yaml` | both | half | 6 |

Every Chen subject has exactly 4 adaptation sessions, so Chen has no K=8 and its K=4 is
its half. The cohorts are frozen before any result in `../../analysis/adaptation_budget_cohort.json`
(primary: >= 4 adaptation sessions, K in {0, 1, 2, 4}; k8: Grossman's >= 8). Each task runs
every subject; the cohort is applied when the results are frozen, which is equivalent
because each subject's embedding is fitted independently.

GPU only: one-GPU Beaker tasks on `ai1/octo-hub-aws-h200`, mounting the same canonical
dataset (`01M1RC0MMA9SX0FF87S1V3YPTQ`) and pyarrow bundle as the matched-half variants.
Comparator: `../q-adaptation-budget/`.
