---
schema_version: 1
id: r6
slug: integration-axis
status: draft
authors: [pck, claude]
wandb_groups:
  - e4-e8-d614-source@20260906-195409
  - rl-baseline-bari@20260713-010244
  - rl-baseline-ctt@20260713-010305
  - rl-baseline-hattori@20260713-010255
inputs:
  scripts:
    - analysis/i1_cluster_axes.py
    - analysis/i2_tdr.py
    - analysis/i3_tdr_dynamics.py
    - analysis/i4_distilled_model.py
    - analysis/i5_mouse_dynamics.py
    - analysis/i6_rl_transitions.py
    - analysis/i7_second_variable.py
    - analysis/i8_v_tests.py
    - analysis/i9_v_kernel.py
    - analysis/i10_gates.py
    - analysis/i11_value_across_lobes.py
    - analysis/i9b_v_kernel_per_mouse.py
    - analysis/i9c_v_kernel_variants.py
    - analysis/report_r6.py
  data:
    - analysis/i1_cluster_axes.json
    - analysis/i2_tdr.json
    - analysis/i3_tdr_dynamics.json
    - analysis/i4_distilled_model.json
    - analysis/i5_mouse_dynamics.json
    - analysis/i6_rl_transitions.json
    - analysis/i7_second_variable.json
    - analysis/i8_v_tests.json
    - analysis/i9_v_kernel.json
    - analysis/i10_gates.json
    - analysis/i11_value_across_lobes.json
    - analysis/i9b_v_kernel_per_mouse.json
    - analysis/i9c_v_kernel_variants.json
reproduce: make -C studies/50-gru-dynamics-embedding r6
---

# Result 6 — is there a shared integration axis? (PLAN.md phase I)

**Question.** The state falls into four clusters by the last input (previous choice ×
outcome, r1 B3c), each spanning a range of P(right). Is the decision variable within the
clusters carried along one shared integration axis, with inputs moving the state between
clusters — or does each cluster integrate along its own direction?

**Answer: one shared integration *coordinate*, carried along two mirror-tilted directions.**

## Summary

1. **Two axes, not one and not four.** Within each cluster, the direction the state moves as
   the decision variable changes (the encoding axis, u_c ∝ Σ_c r — not the readout r itself,
   which would be circular) is the same for the two clusters that share a previous choice
   (cos 0.92 for the model's logit, 0.93–0.94 for Hattori ΔQ) but only partly shared across
   previous choices (0.35–0.52 logit, 0.49–0.71 ΔQ). The two lobe axes share a large PC1/PC4
   component and differ in the sign of their PC2 component — a "V".
2. **The model's own value and the RL value use the same within-cluster direction** (logit
   axis · ΔQ axis 0.94–0.96 in every cluster); the true reward-probability difference does
   not (0.12–0.26) and explains only 1–4% of within-cluster variance, against 23–32% for the
   logit axis. The integrated quantity is a value estimate, not the hidden task state.
3. **TDR shows the organisation directly:** the two previous-choice lobes separate along a
   choice axis and each spans the full range of a common value axis; the outcome separates
   rewarded from unrewarded within each lobe along a reward axis. Five task axes hold 74% of
   the state variance (top five PCs: 82%).
4. **The value coordinate is carried through every jump between clusters** — v(t+1) on v(t)
   has slope 0.78–0.80 when the choice repeats and 0.66–0.74 when it switches side, R²
   0.93–0.98 throughout — while the input moves the state along the choice axis (≈ 2 units on
   a switch) and the reward axis (≈ 0.8–1.7). Each input's map crosses the diagonal near that
   input's attractor (closely for rewarded stays; the fitted affine v* is smaller than the
   attractor for unrewarded stays, e.g. +1.71 vs +2.59, because the maps are curved). The value
   coordinate relaxes toward an input-specific target,
   v(t+1) ≈ v* + λ(v(t) − v*), λ ≈ 0.7–0.8 — the form of a delta-rule update, with an implied
   update fraction (0.20–0.34) in the range of the fitted RL learning rates (Bari median
   0.19, CTT 0.29). (I3's attractor search used forward iteration from 300 starts and
   found one "left rewarded" attractor; D's fuller search finds a second, deeper one, −4.98.)
5. **This reconciles r3 and B4.** r3 found input-driven jumps and attractors whose slowest
   modes barely touch the readout; B4 found strongly nonlinear, reset-like updates of the
   logit. In the value coordinate the update is close to linear, so most of B4's nonlinearity
   is likely in how the readout combines the value coordinate with the input-driven axes and
   the curved remainder of the state, not in how value is carried — a hypothesis, not yet a
   test. (I4 below gives a partial test: a readout a·v + b[last input] reproduces the logit
   with R² 0.94.)
6. **A 1-variable model distilled from the GRU tracks its value coordinate but does not reach
   its likelihood (I4).** Eight affine update rules (input × stay/switch) plus the readout
   a·v + b[last input], simulated from the animals' choices and rewards with no GRU, follow
   the GRU's value coordinate closely (corr 0.985 with per-mouse parameters). It scores
   0.718 (pooled parameters: 0.707), against 0.734 for the GRU and 0.716–0.721 for the RL
   fits; it beats CTT in only 21% of mice. *Corrected in I7:* the first version of this item
   said the readout, not the dynamics, limits it. That compared a pooled readout on the
   GRU's own v (0.719) with per-mouse simulations. With per-mouse readouts on both sides,
   the GRU's own v reaches 0.724. So of the 0.016 gap, ~0.010 is the readout (the GRU
   reads more than v and the last input) and ~0.006 is the simulated dynamics.
7. **Each behavioural axis maps onto a different parameter of the distilled model (I5).**
   - Bias is the asymmetry of the stay targets (ρ +0.91) and the readout offset (+0.78).
   - Reward gain and persistence are mostly *where a switch sends v*: the switch target,
     signed toward the new side, gives ρ +0.93 and +0.92. Faster forgetting matters too
     (λ_switch ρ −0.79 / −0.74; λ_stay −0.74 / −0.60), and the rewarded − unrewarded target
     gap less so (+0.61).
   - The "choice push" (how far an unrewarded stay pulls toward the chosen side, median
     1.8) is a separate, mostly independent parameter.
   - All of them are decodable from the 4-d embedding (rank R² 0.41–0.74; shuffle null
     ≈ 0).
8. **In policy space the GRU looks like Q-learning; in its own state it does not (I6).**
   - On the policy logit, all three models share the same structure. Stays retain 0.7–0.9;
     switches nearly reset (λ 0.13–0.32, R² 0.05–0.30). Normalised targets are similar:
     unrewarded stay 0.17–0.31 of rewarded stay, rewarded switch 0.3–0.4.
   - The difference is in the carried variable. The GRU's v is updated by an affine,
     Markov rule in *every* group, switches included: λ 0.66–0.80 and R² ≥ 0.93, with
     nearly the same retention for every input.
   - RL's ΔQ is not Markov on switches (R² 0.12–0.86, λ 0.18–0.46). After a switch the new
     ΔQ depends on the individual Q values, not only on their difference.
   - Hattori needs separate retentions for rewarded and unrewarded stays (per-mouse median
     0.49 vs 0.90). Bari uses one (0.79), which matches the GRU's input-independent λ
     (0.76) better.
   - The unrewarded-stay push lives in ΔQ for Hattori (target ±0.33 of ±1.01). For Bari it
     lives only in the choice kernel (ΔQ target ±0.07). For the GRU it lives in v itself
     (±1.7–2.0 of ±2.7–2.9).
   - The GRU therefore keeps one leaky-integrated value whose input-specific targets also
     encode choice history. The reset after a switch comes both from v's switch targets
     (near 0, against ±2–2.9 on stays) and from the readout offset b[last input].
9. **The second carried variable is a history of unrewarded choices, along I2's
   choice×reward axis (I7).**
   - The readout direction minus its v component sets the bound: with v, it reads out the
     logit exactly. It holds 1.8% of state variance, and its largest TDR component is the
     choice×reward axis (cos 0.56).
   - That axis alone is nearly as good. Teacher-forced (per mouse) it scores 0.731 against
     0.724 for v alone. Simulated, it scores 0.7234 and closes 34% of the gap, the same as
     the exact residual (33%) and above CTT (0.721).
   - The choice, reward, value×choice and PC5 axes add ≤ 0.001 teacher-forced and close
     ≤ 10% of the gap.
   - What the two coordinates encode (history 2–10 trials back, within each last-input
     cluster):
     - v is mostly choice persistence. It gives positive weight to past choices of a side,
       more if rewarded (lag 2: +0.64 rewarded, +0.41 unrewarded). The weights fall by about
       0.75 per trial (τ ≈ 4 trials, matching I4's λ). The reward bonus fades by lag 5, so
       older history counts by side alone.
     - The residual gives negative weight to unrewarded choices (lag 2: −0.09; lag 3:
       −0.11, decaying) and little to rewarded ones (+0.02 at lag 2, −0.05 at lag 3). It is a "lose → avoid" memory that
       lasts beyond one trial, and it plausibly underlies C1's negative late kernel weights.
   - The remaining two thirds of the gap are in the dynamics. The second coordinate's
     update is curved, not affine, in (v, u): update R² 0.92, against ≥ 0.99 for v. The
     switch groups bend most.
10. **Causal tests of v (I8).** v is a good description of the carried state, but not a
    single leaky mode. It is choice-dominated, like a choice kernel, but so, through
    forgetting, is Q-learning's ΔQ. What distinguishes the GRU is how outcome and time
    combine.
    - **Closure.** v(t) alone predicts v(t+1) with held-out R² 0.93–0.98. Adding the top 20
      PCs gives 0.996–0.998, and the full state 0.9995. So v is approximately
      self-contained, but 2–7% of its next value comes from other directions, most on
      rewarded switches.
    - **Perturbation.** A push along v, run through the GRU, decays as 0.69, 0.48, 0.36,
      0.29 … 0.15 at k = 10, against the one-variable model's Π λ of 0.78, 0.61, 0.47 …
      0.08. That is faster at first, then a slow tail at about 0.92 per trial. Half of the
      perturbation leaves v's axis after one step, and 87% after ten. Mechanistically, v is
      a projection of a multi-timescale system (a fast component plus a slow, off-axis one;
      cf. r3's slow modes). The one-variable rule describes v along the trajectories the
      animals produce (I4, corr 0.985); it is not an isolated mode.
    - **Flip kernels: what v integrates.** In v, flipping a past choice moves it far more
      than flipping its reward. At k = 2, an unrewarded choice gives 0.85 and a rewarded
      choice 1.35 log-odds units, against 0.26 for the reward alone.
      - The choice effect peaks at k = 2. The input reaches the choice and reward axes
        first and flows into v a trial later.
      - From k = 2 on, the unrewarded/rewarded ratio is stable at about 0.65 (0.50 at k = 1),
        matching I4's target ratio.
    - **The same kernels at the policy level, GRU vs RL.**
      - **Separable in the GRU.** Its logit's unrewarded/rewarded choice ratio stays at
        0.39–0.48 for all lags: one choice trace, scaled by outcome.
      - **Not separable in Hattori.** The ratio rises from 0.39 to 1.31: unrewarded
        choices outlast rewarded ones. A small α_unrew (median 0.06, against α_rew 0.50) barely moves the chosen Q, and
        forgetting of the other side does the work.
      - **Bari:** 0.57 at k = 1 (its one-step choice kernel), then 0.28 → 0.46.
      - **The GRU's reward memory is shorter than its choice memory.** The reward's
        share of a rewarded choice's effect falls from 0.31 at k = 2 to 0.06 at k = 10.
        In Hattori it rises (0.25 → 0.37) and in Bari it stays at about 0.4, because both
        store reward and choice information in the same Q values.
    - **Correction.** I wrote earlier that v is "closer to a choice kernel than to a
      reward-driven value difference". With fitted forgetting, Hattori's ΔQ is also
      choice-dominated (k = 1: unrewarded-choice effect 1.53 against reward effect 0.68).
      The GRU-specific properties are separability and a short-lived reward memory.

11. **v is a choice trace plus a faster-fading reward boost, and three parameters describe it
    almost fully (I9).**
    - **The fit.** v was regressed on the last 10 choices (±1) and rewarded choices, then
      scored by held-out R² (80/20 split by session).
    - **Free fit (21 parameters): R² 0.953.**
      - Rewarded-choice weights are 0.48, 0.64, 0.45, 0.29 … at lags 1–4; unrewarded-choice
        weights are 0.26, 0.41, 0.32, 0.23.
      - Both peak at lag 2, the one-trial delay seen in I8.
    - **"One choice trace scaled by outcome"** (v = A Σ λ^(k−1) c_k (1 + κ r_k), 4 parameters):
      R² 0.945, with λ = 0.80 and κ = 0.29. So a reward makes a choice count about 1.3×.
    - **Two-rate** (a separate decay for the reward part, 5 parameters): R² 0.947. The choice
      trace decays at λ = 0.82 and the reward trace faster, at μ = 0.62.
    - **The outcome is not a constant scale in the regression.** The unrewarded/rewarded
      weight ratio rises from 0.55 at lag 1 to above 1 by lag 6: the reward boost fades
      before the choice memory does.
      - I8's causal flips gave a flat ratio of choice effects (about 0.65) instead.
      - The two measure different things. The regression is marginal over correlated
        histories; the flips hold later history fixed and measure a choice × outcome
        interaction.
      - Both agree that the reward part is shorter-lived (I8: reward share 0.31 → 0.06).
    - **Comparison with the RL models.** ΔQ is described about as well by the same models
      (Hattori 0.940 / 0.942 / 0.943; Bari 0.914 / 0.915 / 0.915), so R² alone does not
      separate the GRU from Q-learning. The kernel shapes do:
      - Hattori's unrewarded/rewarded ratio falls (0.58 → 0.25);
      - Bari's ΔQ is almost purely reward-driven (ratio ≈ 0.1; its choice memory sits in
        the choice kernel, outside ΔQ);
      - v's ratio rises.
    - **Revised description.** A slowly decaying memory of which side was chosen (about 0.8
      per trial), with a reward boost (about 30%) that fades faster (about 0.6 per trial).
      This replaces "scaled by outcome" in the strict sense.

12. **How the GRU holds v: recurrent re-writing, not a closed gate (I10).**
    - **Method.** v's one-step retention eᵀJe (e = v's axis, J the step's Jacobian) was
      computed exactly at 20,000 real transitions per input group, at each state's own
      context. It splits into three terms:
      - **leak** (1 − z), the share of the old state the update gate lets through;
      - **gate change**, how the gate itself shifts;
      - **candidate** z ⊙ ∂a, the recurrent drive that re-writes v from the old state.
    - **The gate does not hold v.**
      - The update gate is mostly open along v (z̄ 0.65–0.69), so the leak alone keeps only
        0.31–0.35 per trial.
      - The candidate term contributes about as much again (0.32–0.40), giving a total
        eᵀJe of 0.66–0.72. That matches I8's perturbation decay at k = 1 (0.69).
      - The gate-change term is about 0.
      - So v is kept by the network re-computing it each trial through its recurrent
        weights, not by a gate that stays closed. The hypothesis λ ≈ 1 − z̄ is rejected:
        1 − z̄ is about 0.33, λ about 0.7.
    - **Stay vs switch.** The local retention is the same for stays and switches (0.66–0.72),
      and the gate along v barely differs (z̄ 0.65 vs 0.67).
      - The lower fitted slopes on switches (0.66–0.74, against 0.78–0.80 on stays) are not
        a more open gate.
      - On stays, the fitted slope exceeds the local retention. Other state components
        that covary with v feed into it, so the slope measures v plus those components,
        not v alone.
    - **State dependence.** Retention varies modestly with v (0.63–0.74). It drops when
      switching away from a strongly held side: a faster reset after leaving a committed
      lobe.
    - **Which units carry v.** It is distributed: participation ratio 31.8 of 128 units, with
      the top 10 units holding 48% of the axis. Their update gates range from near 0 to
      near 1.

13. **The kernel result holds within every mouse; it is not a pooling artefact (I9b).**
    - **Method.** I9's fits repeated per mouse (610 mice with ≥ 1,500 trials of full
      history), for v and for each mouse's own Hattori and Bari ΔQ. Odd sessions were held
      out, with split-half reliability.
    - **"Choice memory outlasts the reward boost" holds in essentially every mouse.**
      - **GRU v:** the reward trace decays faster than the choice trace (μ < λ) in **99%**
        of mice, and the unrewarded/rewarded ratio rises from lags 1–3 to lags 5–8 in
        **99.8%**.
      - **Hattori ΔQ:** the opposite in nearly every mouse. μ < λ in 2% and a rising ratio
        in 1%; its reward memory outlasts its choice memory.
      - **Bari ΔQ:** reward-only (early ratio 0.14), with a rising ratio in 4%.
    - **v's timescales are tight across mice and reliable.**
      - Choice-trace decay λ: median 0.79 (10th–90th percentile 0.75–0.83), reliability
        0.95.
      - Reward boost κ: 0.34 (0.18–0.53), reliability 0.79.
      - Reward-trace decay μ: 0.65 (0.58–0.72), reliability 0.65.
      - The late ratio is not reliable per mouse (0.40); the early ratio is (0.82).
    - **What the per-mouse parameters mean behaviourally:**
      - **λ (choice-memory timescale) is the slow-learning embedding axis** (Spearman
        +0.70). Mice whose v has a longer choice memory show longer animal choice memory
        (+0.74), less win-stay (−0.79) and more switching (+0.69). Each outcome moves them
        less, so they commit less after a win.
      - **κ (reward boost) goes the other way:** more win-stay (+0.56), less on the
        slow-learning axis (−0.59), somewhat more determinism (+0.31).
      - **Side bias has no relation to the kernel** (|ρ| ≤ 0.16), consistent with bias
        being an attractor shift (F2), not an integration change.
    - **This settles the I8 / I9 tension in favour of the regression's shape.** The rising
      ratio is a within-mouse property of v, not a mixture effect. I8's flat ratio measures
      a different contrast: choice flips at a fixed later history.

14. **v integrates rewarded choices strongly but briefly, unrewarded choices weakly but for
    longer (I9c).**
    - **Lag-1 control.** The last trial was moved into one offset per last-trial cluster,
      and kernels were fitted on lags 2–10, scored by within-cluster R².
      - v's older kernel is unchanged: lag 2 is 0.64 rewarded and 0.41 unrewarded, as
        before. So the construction's partialling of lag 1 does not distort what v
        integrates from older history.
      - That older history explains 86% of v's within-cluster variance (89% per mouse),
        against 67% for Hattori's ΔQ and 74% for Bari's.
      - **v is closer to a linear filter of past choices and outcomes than ΔQ is.**
        Q-learning updates depend multiplicatively on which side was chosen.
    - **Two separate outcome traces** (rewarded choices A_rew·λ_rew^k, unrewarded choices
      A_unr·λ_unr^k), with offsets:
      - pooled: A_rew 0.64, λ_rew 0.68; A_unr 0.36, λ_unr 0.83;
      - per mouse, λ_unr > λ_rew in **100%** of mice (Hattori 7%);
      - this family fits v slightly better than "choice + reward" in 89% of mice (the
        margins are small; both families describe v almost equally well).
    - **The two parameters map onto two named embedding axes:**
      - **rewarded-choice amplitude A_rew is the determinism axis** (Spearman +0.83);
      - **the timescales and the weight on unrewarded choices are the slow-learning axis**
        (λ_rew +0.65, A_unr +0.53).
    - **Description at this stage (superseded by item 15).** A sum of two leaky traces
      over past choices: rewarded choices about 1.8× the weight at lag 2 with a short
      memory (about 0.68 per trial); unrewarded choices less weight but a longer memory
      (about 0.83 per trial). I9d shows the 0.83 is a compromise between a fast and a
      near-permanent component.

15. **v is a fast choice trace with a reward boost, plus a small, slow residue of unrewarded
    choices (I9d).**
    - **Setup.** I9c's offsets fits with K = 20 lags instead of 10 (K = 10 refitted on the
      same rows), and a six-parameter "three traces" model: the two outcome traces plus a
      shared choice trace A_c·λ_c^k. It nests both four-parameter families. Scored by
      held-out within-cluster R² and per-mouse BIC.
    - **The longer history matters, and the free kernel shows why.**
      - With K = 20, history explains 89.5% of v's within-cluster variance (86.4% at K = 10
        on the same rows).
      - The rewarded kernel falls to ≈ 0 by lag 16. The unrewarded kernel does not: it
        plateaus at about 0.04 from lag 12 to 19. The last lag still rises (0.07), so some
        history older than 20 trials is still piling onto it.
    - **Three traces wins; the two four-parameter families do not.**
      - Pooled gap to the free fit: three traces 0.0006, two outcomes 0.011, choice + reward
        0.019.
      - Per mouse, three traces has the lowest BIC in 99.5% of mice (two outcomes 0%).
        For ΔQ the picture is mixed: Hattori 57% (choice + reward 41%), Bari 73%.
    - **What the three traces are** (pooled; per-mouse medians in brackets):
      - a choice trace on every choice: A_c 0.36, λ_c 0.67 (0.35, 0.65);
      - an extra boost for rewarded choices at a similar timescale: A_rew 0.28,
        λ_rew 0.71 (0.29, 0.72). At lag 2, rewarded choices weigh about 1.6× unrewarded
        ones (per-mouse median);
      - a small unrewarded residue that does not decay over 20 trials: A_unr 0.044,
        λ_unr at the 0.999 bound (A_unr 0.045, 10th–90th percentile 0.034–0.087; > 0 in
        100% of mice; λ_unr at the bound in 64%).
    - **Revised description of v.** Rewarded and unrewarded choices share one fast
      timescale (about 0.65–0.72 per trial), with rewarded choices weighted about 1.6–1.8×.
      On top of that, unrewarded choices leave a small, almost permanent push towards their
      side. I9c's "unrewarded choices have a longer memory" (λ_unr 0.83) was a compromise
      between these two components: the fast part and the slow residue.
    - **What still holds.** In the two-outcome refit at K = 20, λ_unr > λ_rew in 100% of
      mice (0.82 vs 0.69); in Hattori it is 6%. The robust contrast with ΔQ remains: in v,
      unrewarded choices keep pulling towards the chosen side, and the reward part fades
      first.
    - **Caveats.**
      - The residue is small (about 7% of the lag-2 unrewarded weight) and flat over the
        window, so it looks like a session-level variable more than a trace. Whether it is
        a slow side-preference signal that v carries, or a correlate of one, is untested.
      - λ_unr sits at its bound, so its exact value is not identified; only "slower than
        20 trials" is.

16. **The three-trace description, as an update rule, is the best simple model of the mice's
    choices we have, but it does not reproduce the GRU's behaviour around switches (I9e).**
    - **Setup.** Each kernel description became a generative model: traces
      C ← λ_c C + c, W ← λ_w W + c·r, U ← λ_u U + c·(1 − r), and logit = Σ β·trace + bias,
      optionally with last-trial win-stay / lose-stay weights.
      - Fitted per mouse to the animals' choices (Adam, two starts for trace models).
      - Scored on held-out sessions: 2-fold, sessions alternating within mouse.
      - Q-learning (Hattori's form) and Q + choice kernel were refitted the same way.
      - The GRU and the wrapper's RL fits are shown on the same trials. They were trained on
        other sessions, so they are a reference, not a like-for-like comparison. The
        wrapper's Hattori scores 0.7159 against 0.7145 for our refit, so fitting on half the
        sessions costs about 0.001.
    - **Held-out likelihood** (geometric mean per trial):

      | model | likelihood | share of the Q → GRU gap closed |
      |---|---|---|
      | Q (Hattori form), refit | 0.7145 | 0 |
      | Q + kernel + last | 0.7219 | 39% |
      | choice + reward + last | 0.7213 | 35% |
      | two outcomes + last | 0.7221 | 40% |
      | **three traces** (± last) | **0.7231** | **45%** |
      | GRU (wrapper) | 0.7339 | 100% |
      | CTT / Bari / Hattori (wrapper) | 0.7210 / 0.7181 / 0.7159 | |

      - Three traces + last beats Q + kernel + last in 68% of mice (median +0.0013 per
        trial), two outcomes + last in 75%, and choice + reward + last in 82% (Wilcoxon
        p < 10⁻²⁸ for each).
      - It beats the wrapper's CTT in 66% of mice and Bari in 84%, but the GRU in none.
      - Once all three traces are in, the last-trial weights add nothing.
    - **The mice show v's slow unrewarded residue.**
      - Per-mouse medians: a reward trace λ_w 0.64 (β 0.96), an unrewarded trace λ_u 0.98
        (β 0.019; 10th–90th percentile −0.07 to +0.05) and a choice trace λ_c 0.39 (β 0.73).
      - λ_u > λ_w in 99% of mice, and λ_u > 0.95 in 80%.
      - β_u > 0 in 85%: unrewarded choices build a small, lasting pull towards their side.
    - **Where behaviour and v differ.**
      - The behavioural choice trace is much faster than v's (λ_c 0.39 against 0.65).
      - Across mice, the reward-trace parameters agree with v's own fit (β_w Spearman
        +0.64, λ_w +0.42). The slow residue does not (β_u +0.05, λ_u +0.10). Mice differ in
        how much reward drives v, but the per-mouse size of the residue in v doesn't track
        the residue in behaviour.
      - Named axes: β_w is the determinism axis (+0.56); λ_w and the choice-trace timescale
        follow slow learning (+0.40, +0.33).
    - **What it misses.** On the C4e signatures (held-out predictions), three traces + last
      is worse than Q + kernel + last on returns after a failed switch (error 0.034 against
      0.027) and on staying right after switching (0.024 against 0.008). The GRU is an order
      of magnitude closer on both (0.005, 0.004). The trace model captures the likelihood
      gain from long, outcome-split history, but not the GRU's handling of switches (cf. r8
      H4's switch reset).
    - **Caveats.**
      - One mouse (687553, 1.5k trials) has a diverged two-outcome + last fit (−0.55 nats
        per trial held out). It does not change the medians.
      - The loss traces had converged to about 10⁻⁵ per 250 steps by the end.

16. **v carried across the lobes, seen in the (choice axis, v) plane (I11, 2026-10-27).**
    - **The picture.** In I2's choice axis c (which lobe) against v, the two lobes are two
      tilted columns.
      - In a real session, every switch is a near-horizontal jump to the other column: the
        state changes lobe and keeps its v.
      - The per-input flow (1M transitions, full-length arrows) is I3 in 2-d. From the
        opposite lobe the arrows run horizontally; inside the input's own lobe they turn
        vertical and run along v to the attractor.
    - **Switch-triggered averages** (50,000 switches ending runs of ≥ 5 same-side choices;
      signed toward the old side, in SDs):
      - c flips at once (1.00 → −0.84) and so does the GRU's logit (0.71 → −0.29).
      - v crosses over about three trials: 0.92 → 0.59 → 0.19 → 0.04 → −0.10.
      - So for one to two trials after a switch, v still favours the old side while the
        prediction already favours the new one. The lobe sets the immediate prediction; v is
        the carried, slowly updated part.
    - **The whole (c, v) plane (`fig_I11b_cv_plane`, 150,000 states).**
      - **Coloured by the t−1 input:** each lobe holds two parallel bands. The main band has v
        favouring the lobe's own side. A smaller band on the other side of v is the
        just-switched states, whose v still favours the old side.
      - **Coloured by P(right):** the lobe sets most of the prediction, and v modulates it
        within the lobe. Just-switched states take their lobe's colour (right lobe, v < 0:
        P(right) ≈ 0.6–0.7).
      - **Why not colour by v:** it would only repeat the y-axis.
    - **Carry-over read off PC space (`fig_I11c_carry_in_pcs`).** A static scatter shows where v
      is stored, not how it moves, so this figure pairs consecutive states.
      - Only the first state after each choice switch is shown (548,793 switches), in
        PC1–PC2 and PC1–PC4.
      - Each state is coloured by its own v, by the v of the state before the switch, and by
        that pre-switch v shuffled among switches of the same direction.
      - **The pre-switch v reproduces the post-switch gradient** (Spearman 0.975 with own v).
      - **The shuffle scrambles it** (0.20, the side-level part only).
      - So where a state lands after a lobe jump is set by the v it brought from before the
        jump. The post-switch states fill the middle of the arch because v lies mostly along
        −PC1 and is kept.
    - **By construction vs finding.** v is fitted to the logit (I2), so its overall agreement
      with the logit is by construction. The lag between v and c at a switch, and the
      horizontal jumps, are findings.

<!-- BEGIN result-6 -->
[regenerated by `analysis/report_r6.py` — do not edit by hand]

### I1 Per-cluster encoding axes (context and trial position partialled out)

Cluster sizes: R rewarded 965,709, R unrewarded 1,220,044, L rewarded 974,607, L unrewarded 1,170,139.

Cosine of the logit(right) encoding axis across clusters:

| | R rewarded | R unrewarded | L rewarded | L unrewarded |
|---|---:|---:|---:|---:|
| R rewarded | +1.00 | +0.92 | +0.52 | +0.39 |
| R unrewarded | +0.92 | +1.00 | +0.46 | +0.35 |
| L rewarded | +0.52 | +0.46 | +1.00 | +0.92 |
| L unrewarded | +0.39 | +0.35 | +0.92 | +1.00 |

Cosine with the readout r: R rewarded +0.36, R unrewarded +0.41, L rewarded +0.35, L unrewarded +0.40; share of within-cluster state variance along the axis: 0.32, 0.24, 0.32, 0.23.

Cosine of the Hattori ΔQ encoding axis across clusters:

| | R rewarded | R unrewarded | L rewarded | L unrewarded |
|---|---:|---:|---:|---:|
| R rewarded | +1.00 | +0.94 | +0.71 | +0.56 |
| R unrewarded | +0.94 | +1.00 | +0.59 | +0.49 |
| L rewarded | +0.71 | +0.59 | +1.00 | +0.93 |
| L unrewarded | +0.56 | +0.49 | +0.93 | +1.00 |

Cosine with the readout r: R rewarded +0.28, R unrewarded +0.35, L rewarded +0.28, L unrewarded +0.34; share of within-cluster state variance along the axis: 0.27, 0.16, 0.28, 0.17.

Within each cluster, logit axis · ΔQ axis: +0.94, +0.95, +0.96, +0.95; logit axis · P(rew|R) − P(rew|L) axis: +0.14, +0.20, +0.12, +0.26.
Curvature (local logit axis per value tercile · the cluster's axis): R rewarded 0.91/0.98/0.75; R unrewarded 0.76/0.96/0.80; L rewarded 0.69/0.96/0.92; L unrewarded 0.80/0.95/0.69.

![I1](../fig_I1_cluster_axes.png)

### I2 Targeted dimensionality reduction

1,000,000 trials; regressors value, choice, reward, value×choice, choice×reward + context e0..e3, trial_position. Task + nuisance R² of the state 0.729; the 5-d TDR subspace holds 73.7% of state variance (top 5 PCs: 81.5%). Lobe value axes (β_v ± β_vc) cosine +0.40; with the readout: R lobe (β_v + β_vc) +0.42, L lobe (β_v − β_vc) +0.42.

![I2](../fig_I2_tdr.png)

### I3 The value coordinate across transitions

| input (trial t) | from | n | slope v(t+1) on v(t) | R² | mean jump: value | choice | reward |
|---|---|---:|---:|---:|---:|---:|---:|
| R rewarded | from R lobe (stay) | 183,033 | 0.79 | 0.97 | +0.26 | +0.17 | +0.84 |
| R rewarded | from L lobe (switch) | 39,872 | 0.66 | 0.93 | +0.43 | +2.40 | +1.63 |
| R unrewarded | from R lobe (stay) | 258,359 | 0.80 | 0.97 | +0.04 | -0.07 | -0.76 |
| R unrewarded | from L lobe (switch) | 23,537 | 0.68 | 0.96 | -0.00 | +1.95 | -0.90 |
| L rewarded | from R lobe (switch) | 40,475 | 0.73 | 0.95 | -0.22 | -1.95 | +1.70 |
| L rewarded | from L lobe (stay) | 184,555 | 0.79 | 0.98 | -0.23 | -0.01 | +0.93 |
| L unrewarded | from R lobe (switch) | 22,437 | 0.74 | 0.97 | +0.03 | -2.17 | -0.99 |
| L unrewarded | from L lobe (stay) | 247,732 | 0.78 | 0.98 | -0.10 | -0.10 | -0.89 |

Stable fixed point of each constant input (all-sessions mean context) on the value axis: R rewarded +3.07 (logit +4.34), R unrewarded +2.59 (logit +3.05), L rewarded -2.60 (logit -4.04), L unrewarded -2.56 (logit -3.05).
Implied per-trial update fraction 1 − slope: 0.20–0.34; fitted RL for reference: bari learn_rate median 0.19, ctt learn_rate median 0.29.

![I3](../fig_I3_tdr_dynamics.png)

### I4 Distilled one-variable model

| group | n | λ | v* |
|---|---:|---:|---:|
| L unrewarded (stay) | 1,070,349 | 0.78 | -2.03 |
| L unrewarded (switch) | 97,545 | 0.74 | +0.35 |
| L unrewarded (first) | 2,245 | 0.00 | -0.73 |
| L rewarded (stay) | 795,493 | 0.79 | -2.71 |
| L rewarded (switch) | 176,792 | 0.73 | -0.04 |
| L rewarded (first) | 2,322 | 0.00 | -0.67 |
| R unrewarded (stay) | 1,115,683 | 0.80 | +1.71 |
| R unrewarded (switch) | 102,224 | 0.68 | -0.25 |
| R unrewarded (first) | 2,137 | 0.00 | +0.06 |
| R rewarded (stay) | 791,283 | 0.79 | +2.91 |
| R rewarded (switch) | 172,232 | 0.66 | +0.49 |
| R rewarded (first) | 2,194 | 0.00 | +0.15 |

Eval likelihood: GRU 0.7339, distilled, per mouse 0.7180, distilled, pooled 0.7069, distilled readout on GRU's own v (teacher-forced) 0.7191, GRU, 1-d carried state (E2) 0.7126, bari 0.7181, ctt 0.7210, hattori 0.7159.
Teacher-forced readout R² (logit ≈ a·v + b[last input]) 0.944; simulated v vs the GRU's v: corr 0.985 per mouse, 0.976 pooled; logit corr 0.967. Per-mouse distilled model beats CTT in 21.3% of 614 mice.

![I4](../fig_I4_distilled_model.png)

### I5 Per-mouse dynamics parameters vs behaviour and the embedding

Spearman across mice (rows: I4 per-mouse parameters; columns: F3 GRU kernel features), and rank R² decoding each parameter from the 4-d embedding (shuffle 95th percentile):

| parameter | bias | reward gain (1 back) | persistence (1 back) | reward decay (2/1 back) | embedding rank R² | null 95% | 10/50/90th pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| λ stay | +0.19 | -0.74 | -0.60 | -0.35 | 0.56 | 0.00 | 0.74 / 0.76 / 0.80 |
| λ switch | +0.13 | -0.79 | -0.74 | -0.44 | 0.55 | 0.00 | 0.67 / 0.70 / 0.74 |
| reward push | +0.20 | +0.61 | +0.60 | +0.38 | 0.41 | -0.00 | 0.71 / 0.85 / 0.95 |
| choice push | +0.04 | -0.06 | +0.23 | +0.20 | 0.59 | 0.00 | 1.60 / 1.78 / 1.99 |
| target asymmetry | +0.91 | +0.11 | +0.12 | -0.09 | 0.73 | -0.00 | -0.27 / -0.06 / 0.16 |
| switch target | +0.00 | +0.93 | +0.92 | +0.55 | 0.74 | 0.00 | -0.29 / 0.10 / 0.42 |
| readout slope a | +0.03 | -0.47 | -0.25 | +0.05 | 0.53 | 0.00 | 0.78 / 0.87 / 0.99 |
| readout offset | +0.78 | -0.06 | -0.09 | -0.28 | 0.74 | -0.00 | -0.11 / 0.01 / 0.14 |

![I5](../fig_I5_mouse_dynamics.png)

### I6 The same transitions in the RL models

Per group: pooled λ (median of per-mouse λ) / target / R².

| group | GRU state | GRU logit | Hattori (L2F1) state | Hattori (L2F1) logit | Bari (L1F1_CK1) state | Bari (L1F1_CK1) logit |
|---|---:|---:|---:|---:|---:|---:|
| L unrewarded (stay) | 0.78 (0.76) / -2.03 / 0.98 | 0.79 (0.76) / -1.18 / 0.94 | 0.91 (0.90) / -0.33 / 0.98 | 0.89 (0.90) / -1.15 / 0.98 | 0.74 (0.79) / -0.07 / 0.97 | 0.81 (0.79) / -1.05 / 0.97 |
| L unrewarded (switch) | 0.74 (0.73) / +0.35 / 0.97 | 0.31 (0.41) / -0.83 / 0.29 | 0.33 (0.31) / -0.10 / 0.63 | 0.31 (0.31) / -0.27 / 0.55 | 0.46 (0.49) / -0.03 / 0.85 | 0.28 (0.49) / -1.37 / 0.22 |
| L rewarded (stay) | 0.79 (0.76) / -2.71 / 0.98 | 0.81 (0.70) / -5.69 / 0.87 | 0.57 (0.48) / -1.01 / 0.90 | 0.70 (0.48) / -3.75 / 0.87 | 0.87 (0.79) / -1.23 / 0.94 | 0.88 (0.79) / -6.35 / 0.96 |
| L rewarded (switch) | 0.73 (0.74) / -0.04 / 0.95 | 0.17 (0.32) / -1.62 / 0.05 | 0.18 (0.24) / -0.52 / 0.13 | 0.13 (0.24) / -1.43 / 0.05 | 0.35 (0.47) / -0.25 / 0.31 | 0.23 (0.47) / -1.99 / 0.09 |
| R unrewarded (stay) | 0.80 (0.77) / +1.71 / 0.97 | 0.81 (0.76) / +1.23 / 0.94 | 0.91 (0.90) / +0.33 / 0.98 | 0.90 (0.90) / +1.15 / 0.98 | 0.74 (0.79) / +0.07 / 0.97 | 0.81 (0.79) / +1.09 / 0.97 |
| R unrewarded (switch) | 0.68 (0.67) / -0.25 / 0.96 | 0.32 (0.44) / +0.91 / 0.30 | 0.33 (0.29) / +0.10 / 0.63 | 0.32 (0.29) / +0.34 / 0.57 | 0.46 (0.48) / +0.03 / 0.86 | 0.28 (0.48) / +1.39 / 0.22 |
| R rewarded (stay) | 0.79 (0.76) / +2.91 / 0.97 | 0.82 (0.72) / +5.80 / 0.88 | 0.56 (0.49) / +1.01 / 0.90 | 0.69 (0.49) / +3.75 / 0.87 | 0.87 (0.79) / +1.22 / 0.94 | 0.88 (0.79) / +6.36 / 0.96 |
| R rewarded (switch) | 0.66 (0.67) / +0.49 / 0.93 | 0.19 (0.32) / +1.78 / 0.07 | 0.18 (0.23) / +0.52 / 0.12 | 0.13 (0.23) / +1.49 / 0.05 | 0.35 (0.48) / +0.25 / 0.31 | 0.23 (0.48) / +2.07 / 0.11 |

State: GRU: I2 value coordinate; RL: Q_R − Q_L; logit: log P(R)/P(L).

![I6](../fig_I6_rl_transitions.png)

### I7 A second carried coordinate

GRU 0.7339, CTT 0.7210. Teacher-forced = the readout on the GRU's own coordinates; simulated = the distilled model run from the animals' inputs.

| carried state | teacher-forced (pooled) | teacher-forced (per mouse) | simulated (per mouse) | simulated (pooled) | gap closed | update R² of u | sim corr of u |
|---|---:|---:|---:|---:|---:|---:|---:|
| v only | 0.7191 | 0.7241 | 0.7180 | 0.7069 | 0% | — | — |
| + readout residual | 0.7339 | 0.7339 | 0.7232 | 0.7118 | 33% | 0.92 | 0.93 |
| + choice axis | 0.7193 | 0.7243 | 0.7188 | 0.7081 | 5% | 0.99 | 0.99 |
| + reward axis | 0.7193 | 0.7245 | 0.7195 | 0.7079 | 10% | 0.98 | 0.99 |
| + value×choice axis | 0.7191 | 0.7244 | 0.7188 | 0.7075 | 5% | 0.93 | 0.82 |
| + choice×reward axis | 0.7297 | 0.7310 | 0.7234 | 0.7129 | 34% | 0.92 | 0.91 |
| + PC5 (time) | 0.7192 | 0.7242 | 0.7181 | 0.7074 | 1% | 0.93 | 0.82 |
| + choice + reward axes | 0.7195 | 0.7247 | 0.7200 | 0.7089 | 13% | 0.99 | 0.99 |

Readout-residual axis: 1.8% of state variance; cosine with the TDR axes value -0.00, choice +0.16, reward -0.04, value×choice +0.03, choice×reward +0.56. Within-cluster R² from choice and choice×reward history (lags 2–10): v 0.86, residual 0.59.

| lag | v: choice | v: choice×reward | residual: choice | residual: choice×reward |
|---:|---:|---:|---:|---:|
| 2 | +0.411 | +0.233 | -0.093 | +0.109 |
| 3 | +0.317 | +0.134 | -0.105 | +0.058 |
| 4 | +0.228 | +0.060 | -0.080 | +0.044 |
| 5 | +0.167 | +0.017 | -0.059 | +0.038 |
| 6 | +0.130 | -0.007 | -0.044 | +0.029 |
| 7 | +0.110 | -0.024 | -0.035 | +0.021 |
| 8 | +0.102 | -0.039 | -0.029 | +0.013 |
| 9 | +0.110 | -0.056 | -0.031 | +0.008 |
| 10 | +0.170 | -0.101 | -0.054 | +0.015 |

![I7](../fig_I7_second_variable.png)

### I8 Causal tests of v

Closure, held-out R² of v(t+1):

| group | v(t) only | v + top 20 PCs | full 128-d state |
|---|---:|---:|---:|
| L unrewarded (stay) | 0.9785 | 0.9968 | 0.9997 |
| L unrewarded (switch) | 0.9706 | 0.9967 | 0.9996 |
| L rewarded (stay) | 0.9831 | 0.9974 | 0.9997 |
| L rewarded (switch) | 0.9444 | 0.9970 | 0.9996 |
| R unrewarded (stay) | 0.9709 | 0.9980 | 0.9997 |
| R unrewarded (switch) | 0.9614 | 0.9959 | 0.9996 |
| R rewarded (stay) | 0.9697 | 0.9982 | 0.9996 |
| R rewarded (switch) | 0.9332 | 0.9962 | 0.9995 |

Perturbation of v by ±0.5 (k = 1…10):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU Δv/δ | 0.69 | 0.48 | 0.36 | 0.29 | 0.25 | 0.22 | 0.20 | 0.18 | 0.16 | 0.15 |
| Π λ (I4) | 0.78 | 0.61 | 0.47 | 0.37 | 0.29 | 0.22 | 0.17 | 0.14 | 0.11 | 0.08 |
| share off v's axis | 0.49 | 0.65 | 0.73 | 0.77 | 0.79 | 0.81 | 0.83 | 0.84 | 0.86 | 0.87 |
| Δlogit/δ | 0.71 | 0.44 | 0.33 | 0.27 | 0.23 | 0.20 | 0.18 | 0.16 | 0.15 | 0.14 |

Flip kernels in log-odds units (k = 1…10): reward effect / choice effect unrewarded / choice effect rewarded:

- GRU, a·v: R 0.22, 0.26, 0.17, 0.10, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01 | C₀ 0.47, 0.85, 0.65, 0.46, 0.33, 0.25, 0.19, 0.16, 0.13, 0.12 | C₁ 0.93, 1.35, 0.97, 0.67, 0.49, 0.38, 0.30, 0.25, 0.21, 0.17
- GRU, full logit: R 1.01, 0.60, 0.35, 0.22, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02 | C₀ 2.01, 0.81, 0.51, 0.35, 0.26, 0.21, 0.17, 0.15, 0.14, 0.12 | C₁ 4.32, 1.96, 1.16, 0.82, 0.65, 0.53, 0.44, 0.37, 0.30, 0.25
- hattori, β·ΔQ: R 0.68, 0.46, 0.32, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.04 | C₀ 1.53, 1.08, 0.80, 0.60, 0.46, 0.35, 0.28, 0.22, 0.18, 0.15 | C₁ 3.95, 1.85, 1.11, 0.73, 0.51, 0.36, 0.26, 0.20, 0.15, 0.11
- bari, β·ΔQ: R 0.76, 0.56, 0.41, 0.31, 0.23, 0.18, 0.14, 0.11, 0.09, 0.07 | C₀ 0.52, 0.41, 0.32, 0.26, 0.21, 0.17, 0.14, 0.12, 0.10, 0.09 | C₁ 2.27, 1.46, 1.02, 0.76, 0.58, 0.45, 0.35, 0.28, 0.23, 0.19
- bari, choice-kernel term: R 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 | C₀ 1.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 | C₁ 1.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
- bari, full logit: R 0.76, 0.56, 0.41, 0.31, 0.23, 0.18, 0.14, 0.11, 0.09, 0.07 | C₀ 2.32, 0.41, 0.32, 0.26, 0.21, 0.17, 0.14, 0.12, 0.10, 0.09 | C₁ 4.07, 1.46, 1.02, 0.76, 0.58, 0.45, 0.35, 0.28, 0.23, 0.19

Checks: {'cpu_vs_cached_v_median_abs': 0.00020619474713912123, 'hattori': {'max_abs_q_update_error': 0.0, 'max_abs_logit_error': 2.398081733190338e-14}, 'bari': {'max_abs_q_update_error': 0.0, 'max_abs_logit_error': 1.199040866595169e-13}}.

![I8](../fig_I8_v_tests.png)
<!-- END result-6 -->

## Method notes and caveats

- **Circularity.** Using the model's logit as the value regressor makes every other TDR axis
  exactly orthogonal to the readout (the logit *is* r·h) — a property of the method. The
  informative results are (i) the encoding axes' alignment *across* clusters, (ii) their
  agreement with an external value (Hattori ΔQ), and (iii) the transition maps, which are
  dynamic and not fitted.
- **Curvature.** Local axes per value tercile agree with their cluster's axis at 0.95–0.98 in
  the middle tercile but only 0.69–0.92 at the extremes: the clusters are curved, so a linear
  axis is an approximation that is best near the middle of each cluster.
- **Collinearity.** Previous choice and value are strongly correlated; TDR separates them only
  through their non-shared variation, which is why I1 (within clusters, previous choice
  fixed) is the primary test and I2 the picture.
- **The λ–learning-rate comparison is descriptive:** a pooled slope in the GRU's own value
  coordinate set against per-mouse parameters of other models.
- One seed of one model; contexts partialled out linearly (4-d context + trial position).
- **I4–I6 fits.** v is I2's value coordinate, which was fitted on the logit, so v and the
  logit are not independent. Pooled RL fits mix mice with different learning rates, so the
  per-mouse medians are the better comparison of λ. ΔQ on switch trials covers a narrower
  range than on stays, which lowers R² there by itself; the GRU's switch R² is high despite
  the same selection. I5's per-mouse fits use the pooled values for groups with < 50
  transitions in a mouse.
- **I7 history weights** are fitted within each last-input cluster, because lag 1 is fixed
  by the cluster. Lag 10 absorbs all older history, which is why it is larger than lag 9.
  The readout residual is an exact bound by construction (the logit is linear in h). The
  informative comparisons are the simulated scores and how close an interpretable axis
  comes to that bound.
- **Composition of v is correlational.** v is defined from the logit (I2), so its relation
  to the choice holds by construction. Its history weights come from a within-cluster
  regression. The raw value and choice×reward axes have cosine −0.70 before
  orthogonalisation, so reward history is split between v and u in a way that depends on
  the order. The causal check is PLAN.md H2; I8 has done its flip part.
- **I8 flips** are input-level counterfactuals averaged over states. Flipping a choice
  while keeping its outcome is not always possible in a baited task. RL flips use each
  mouse's fitted parameters, replayed in numpy from the cached Q values (validated:
  update error 0, logit error < 1e-13). The perturbation uses δ = ±0.5, about 0.4 of v's
  within-cluster SD, and is symmetrised.

## What this suggests next

- E2b (r4) and F2 (r5) have since been done. E2b found the memory beyond two trials on the
  readout coordinate and both lobe axes, not on the pooled value axis alone. F2 found gain
  in v's retention, not in attractor depth.
- **Close the rest of the gap (after I7):** give the choice×reward coordinate a nonlinear
  (e.g. piecewise or quadratic in v, u) update per group, since its affine update is the
  bottleneck.
- **Test hypothesis 5:** read the logit out of the TDR coordinates and check whether the
  B4 nonlinearity is reproduced by the readout of a linearly carried value.
