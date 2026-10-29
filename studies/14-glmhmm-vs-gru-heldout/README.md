---
schema_version: 1
id: 14-glmhmm-vs-gru-heldout
question: Where does a GLM-HMM over left/right/ignore, refit on each held-out mouse's
  context sessions, land against the 3-way GRU on the same held-out trials, and how
  many latent states does it need?
status: planned
wandb:
  entity: AIND-disRNN
  project: mice_glmhmm
motivated_by:
- study: 02-gru-scaling-law-ignore
  report: r1
  note: the 3-way GRU predicts left/right on engaged trials as well as the 2-way models, but
    nothing states which latent structure carries that; a GLM-HMM is the classical model with
    explicit discrete strategy states, scored on the same keys
- study: 02-gru-scaling-law-ignore
  report: r2
  note: ignore detection is real and scales with D, so engagement is a modelled outcome; a
    GLM-HMM can give disengagement its own latent state and test it against the GRU's ignore head
- issue: 48
  note: the design settled for this study (emissions, protocol, priors, K selection, data
    scope), recorded there before scaffolding
tags:
  model:
  - glmhmm
  question:
  - comparison
  - interpretation
issues:
- 48
---
# Study: 14-glmhmm-vs-gru-heldout

W&B project [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm).

**Question.** Where does a GLM-HMM over left/right/ignore, refit on each held-out mouse's
context sessions, land against the 3-way GRU on the same held-out trials, and how many latent
states does it need?

**Prior work.** Faeze fit a pooled GLM-HMM with `ssm` on about 400 mice:
- Ignore was the reference class.
- Only final/graduated sessions were used.
- There were no held-out mice and no detailed look at performance.
- K was chosen by CV and AIC/BIC, without a clear winner.

This study builds on that work. It adds dynamax, two emission parameterisations, the GRU's
exact split, and the unified `run_hpc` stack with the GRU's W&B keys. A diagnostic variant will
reproduce her fit (`PLAN.md`, Variants).

## Design

Full design: [`PLAN.md`](PLAN.md); decisions: issue #48.

- **Model.** K latent states with stationary transitions, restarted every session. Each state
  has a GLM over {left, right, ignore} driven by trial history: RewC and UnrC lags (signed)
  and Ignored and Rewarded lags (unsigned). Two emissions:
  - `hurdle`: per state, an engagement GLM times a side GLM;
  - `reference`: per state, one multinomial GLM with ignore as the zero logit.

  Code: `aind-dynamic-foraging-models` `glm_hmm` (AllenNeuralDynamics/aind-dynamic-foraging-models#81).
- **Data and split.** As study 02's 3-way GRU: `mice_snapshot_scaling` (snapshot
  `20260603`), `ignore_policy=include`, all training stages, and the 149 reserved held-out mice.
  Each mouse's even sessions are context and its odd sessions are scored.
- **Protocol.** Wrapper `GLMHMMTrainer` (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#110):
  1. A global fit on the training mice's train sessions.
  2. Zero-shot scores on the held-out mice's eval sessions (`heldout/zero_shot/*`; not
     `heldout/test_likelihood`, which for the GRU covers every held-out session).
  3. A MAP refit per held-out mouse on its context sessions, scored on its eval sessions
     (`heldout/eval_likelihood`, `heldout/final/*`, the keys study 02 reads).

  Every score is the forward filter's one-step predictive.
- **K selection.** The same per-mouse protocol on the training mice (`glmhmm/select/*`).
  Neither the held-out mice nor the global fit's own likelihood is used.

| variant | what varies | answers |
|---|---|---|
| `lag-scan` | K = 1, history length L ∈ {3, 5, 7, 10}, both emissions | which L the K grid uses |
| `hurdle` | K = 1 … 10, hurdle emission, chosen L | held-out score vs K; states |
| `reference` | K = 1 … 10, reference emission, chosen L | the same, for Faeze's 3-class `ssm` parameterisation (ignore as the zero-logit class) |

## Verdict

None yet: no variant has been launched. Launches wait for the code PRs to merge and the models
package to be released.
