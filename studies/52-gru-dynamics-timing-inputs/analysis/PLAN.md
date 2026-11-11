# Study 52 — analysis plan

Model order: **R** (RT only) first, then L (licks), T (both), with S (shuffled) and the
other seeds as controls. Each phase runs on R before moving to L. Simple analyses come
before complex ones (Study 50 convention). Model comparisons use side-by-side panels and
PCs 1–4, and every figure separates what is true by construction from what is a finding.

## What changes relative to Study 50

Study 50's analyses rely on C having **four discrete inputs** (prev choice × prev reward).
B3's four clusters, B4's update maps, C1's impulse responses and D's fixed point per input
all depend on that. R adds a **continuous** fifth channel, so each of those analyses
becomes a family indexed by the RT value. The default discretization is **RT quintiles
within each choice × reward cell**, plus a synthetic sweep over the input range
(−2.5…+2.5 sd) where the model is driven directly. Comparisons with C always use the same
trials. The eval split should match C's exactly (same `ignore_policy`, `eval_every_n`,
snapshot and cohort); A2 verifies this rather than assuming it.

The headline question for R: **does RT enter as**
1. an additive push on the existing value coordinate v (I-phase),
2. a gain on the outcome update (RT × reward interaction — e.g. a slow, hesitant choice
   learns less from its outcome),
3. a separate state dimension (e.g. engagement or vigilance) that reaches the readout
   independently of v, or
4. a quantity mostly absorbed by between-mouse differences (the subject embedding), with
   little trial-level effect?

Study 07's probe found that RT predicts switching independently of reward. That
points to (1) or (3) rather than (2), but it was measured with a logistic probe, not on
the GRU.

## A. Extraction and alignment
- **A1** trial table for R-s0 (`extract_trial_states.py`): reproduces the logged eval
  likelihood, readout reconstructs logits, `prev_log_rt_input` = what the GRU saw.
- **A2** align with C-s0 on the same trials (`a2_align_with_c.py`). **Done for R-s0:** the
  sessions and trials are identical; R − C = +0.00087, all of it in RT-present sessions.

## B. Descriptive (straightforward first)
- **B0 where does R beat C?** Δ log-likelihood on shared trials, broken down by:
  prev choice × reward; prev-RT quintile; trials since a switch; task; mouse. Is the
  +0.0009 spread evenly or concentrated (e.g. slow RT after a loss)? Include the raw
  behaviour as well: P(switch | prev RT quintile, prev reward) for animals, R and C.
- **B1** example sessions with RT shown alongside choices.
- **B3** PCA of R's states (PCs 1–4). Do C's four last-input clusters persist? Within each
  cluster, colour by `prev_log_rt_input`: do the clusters smear along an RT direction, and
  is that direction shared across clusters? Compare participation ratio with C (4.3).
- **B4** update maps: Δlogit and Δv per trial as a function of prev-RT quintile within each
  choice × reward cell, R vs C on the same trials.

## C. Behaviour and baselines
- **C4-RT** the RT signatures in the animals (switch probability, stay after a slow loss,
  etc.) and whether R reproduces them where C cannot. Calibration plots per RT bin.
- RL baselines with an RT term (Q-learning with RT-modulated learning rate or choice
  bias) as an interpretable reference for (1) vs (2) above, if B0 and B4 show a clear form.

## I/D. Where the input enters the dynamics
- **I-RT1** (done, r2; sweep instead of Jacobian) input direction: ∂h_{t+1}/∂(RT input) at each visited state (via `gru_step`),
  projected onto the readout, R's own v axis (recomputed via I2 TDR), the reward axis and
  the slow modes. Is the RT direction state-dependent (that would mean a gain), or roughly
  constant (an additive push)?
- **I-RT2** (done, r2) counterfactual clamp: replay each session with the RT input held at 0 (the
  population mean) or at the mouse's own mean. Δlikelihood measures how much is trial-level
  RT and how much is the mouse's typical RT. The persistence of the logit difference over
  later trials gives the RT memory timescale.
- **C1-RT** (done, r2) impulse response to a single RT pulse (±1 sd) at matched states, compared with
  the choice/reward kernels.
- **D-RT** (done, r4: no consistent effect) fixed points and slow modes of the autonomous map at fixed RT values (quintile
  centres): do the attractors move smoothly with RT, and does attractor depth change?

## F. Subject embedding
- RT differs strongly between mice (within-mouse medians 0.12–0.86 s). Compare R's and C's
  embeddings: does C's embedding encode mouse-level mean RT (regress embedding on per-mouse
  RT statistics), and does that component shrink in R? Repeat F3's bias vs gain/persistence
  axes.

## G. Robustness and controls
- R-s1, R-s2 for every headline claim.
- S (shuffled) answers which effects come from having extra inputs regardless of their
  trial alignment. It carries RT and licks, so it serves mainly L and T. For R alone, the
  clamp in I-RT2 is the within-model control.

## By construction (state it on every relevant figure)
- 24% of eval trials have no RT. They come as whole sessions (1,659; 108 mice entirely),
  concentrated in Coupled Baiting. There the RT input is the constant 0. A2: R's gain is
  entirely in RT-present sessions (+0.00123 vs −0.00015), so restrict RT analyses to
  those sessions and report the RT-absent ones separately as a within-model control
  (R should behave like C there).
- RT on row t is the previous *engaged* trial's RT; the first row of a session is −1.
- Global standardization: the RT input differs across mice by construction. A mouse-level
  RT effect is not the same thing as a trial-level one (separate them with I-RT2).
- The loss only asks for choice, so any RT structure in the state exists because RT
  predicts choice.

---

# Licks model (L) — plan (added 2026-10-26)

L adds two inputs: the previous trial's left and right lick counts in [go cue, go cue + 2 s),
standardized as (count − 3.139) / 4.159 and filled with 0 when missing. Study 07: L carries
about 98% of the response-feature gain (held-out +0.0077 over C, ~10× R).

## What changes relative to R

- **Two channels, strongly tied to the inputs C already has.**
  - The chosen-side count largely restates the choice (R² ≈ 0.6).
  - On rewarded trials it includes consummatory licks, which partly restates the reward.
- **Coordinates:** every analysis re-expresses the inputs relative to the choice:
  - **chosen-side licks** (vigour/consumption);
  - **other-side licks** (licking the unchosen port: checking/exploring).

  Analyses are split by the previous choice × reward, as for R. Study 07's probe suggested
  licks predict switching mainly after unrewarded trials, which points at other-side licks
  after losses.
- **By construction:** after a reward the chosen-side count is high by definition; zero
  licks on both sides at an engaged trial should be rare (check the A1 summary); missing
  lick data become 0, which is not distinguishable from a real zero count.

## Steps (mirroring R)

- **A1/A2:** extract L-s0 (and L-s1 as the seed baseline); align with C; check whether the
  gain depends on sessions with unusual lick data.
- **B0-L:**
  - gain over C by lick pattern within each choice × reward cell (chosen and other-side
    deciles);
  - P(switch) vs other-side and chosen-side licks for mice, L and C, side by side;
  - a conditional regression with lags 1–3;
  - how concentrated the gain is, vs seed noise;
  - per mouse.
- **B3-L:** PCA coloured by the lick inputs; decoding chosen/other counts at lags 1–5
  (raw, beyond history, unique); the lick directions vs the readout, the reward axis and
  R's RT axis (if comparable).
- **I-L (mechanism):**
  - a 2-D update rule: sweep (chosen, other) counts at real states, per outcome;
  - push vs gain;
  - impulse responses of +k other-side or chosen-side licks vs a reward flip;
  - clamps:
    - replace licks with their mean given choice × reward (removes only what is beyond
      choice/reward);
    - shuffle within session;
    - clamp one channel at a time.
- **F/D:** embedding (mouse-level lick vigour) and, if warranted, fixed points.
- **Combined model (T, RT + licks)** after L: is RT's effect preserved, absorbed or
  redundant when licks are also available (Study 07: mildly sub-additive)?
