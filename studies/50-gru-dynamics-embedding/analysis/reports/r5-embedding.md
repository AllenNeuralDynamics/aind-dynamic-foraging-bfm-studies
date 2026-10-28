---
schema_version: 1
id: r5
slug: embedding
status: draft
authors: [pck, claude]
wandb_groups:
  - e4-e8-d614-source@20260906-195409
  - rl-baseline-bari@20260713-010244
  - rl-baseline-ctt@20260713-010305
  - rl-baseline-hattori@20260713-010255
inputs:
  scripts:
    - analysis/f3_embedding_features.py
    - analysis/f1_embedding_extremes.py
    - analysis/f4_early_late.py
    - analysis/f2_mouse_dynamics.py
    - analysis/f5_embedding_atlas.py
    - analysis/f6_embedding_types.py
    - analysis/f4b_context_drift.py
    - analysis/f4c_context_trajectories.py
    - analysis/f7_feature_library.py
    - analysis/f8_embedding_traversal.py
    - analysis/f9_strategy_cards.py
    - analysis/f10_embedding_behaviour_maps.py
    - analysis/f11_prototype_examples.py
    - analysis/f12_context_learning_examples.py
    - analysis/f13_prototype_state_space.py
    - analysis/report_r4_r5.py
  data:
    - analysis/f3_embedding_features.json
    - analysis/f1_embedding_extremes.json
    - analysis/f4_early_late.json
    - analysis/f2_mouse_dynamics.json
    - analysis/f5_embedding_atlas.json
    - analysis/f6_embedding_types.json
    - analysis/f4b_context_drift.json
    - analysis/f4c_context_trajectories.json
    - analysis/f7_feature_library.json
    - analysis/f8_embedding_traversal.json
    - analysis/f9_strategy_cards.json
    - analysis/f9_mouse_strategy_profiles.csv
    - analysis/f10_embedding_behaviour_maps.json
    - analysis/f11_prototype_examples.json
    - analysis/f12_context_learning_examples.json
reproduce: make -C studies/50-gru-dynamics-embedding r5
---

# Result 5 — what the subject embedding does (PLAN.md phase F)

**Done:** the behavioural halves of F1, F3 and F4; F2, the attractors and timescales of all
mouse contexts (2026-10-24, GPU; rerun 2026-10-26 with 96 starts). F4b, the context half of F4 (2026-10-26).

## Summary

1. **Study 01 r10 replicates on model C.** The 4-d embedding predicts each mouse's fitted RL
   side bias (rank R² 0.73–0.74) and learning rate (0.65–0.67), as r10 found on model A.
2. **It predicts the GRU's own per-mouse behaviour at rank R² 0.78–0.80** (bias, reward gain,
   persistence) — expected to exceed the animals' (0.52–0.70), since GRU features are fit to
   noise-free soft labels — and those GRU features track the animals' per mouse (Spearman
   0.95 bias, 0.80 gain, 0.96 persistence).
3. **Not a task-mix effect.** Which tasks a mouse ran explains ≤ 0.26 of any feature; the
   embedding still predicts 0.60–0.78 of bias, gain and persistence after that is removed.
4. **Two separate directions: bias, and gain + persistence.** In a coordinate-free metric
   (embedding covariance, i.e. correlation of the embedding-predicted features) bias is
   orthogonal to reward gain (−0.05; Spearman of the features −0.03) and to persistence
   (−0.08); gain and persistence are one axis (0.94; Spearman 0.90), shared with the RL
   learning rates (CTT learn_rate 0.92). Mice that update strongly on reward also repeat
   more regardless of outcome.
5. **The gain axis behaviourally (F1):** the update maps are the GRU's logits for each group.
   After a reward, the GRU for high-gain mice jumps toward the rewarded side (log-odds
   ≈ +2) when it leaned away or was neutral (logit < 0.5), and moves along the diagonal
   beyond that; for low-gain mice it updates incrementally. After no reward, the high-gain
   GRU still leans toward repeating; the low-gain GRU moves toward the chosen side when it
   leaned away and shifts away once it leaned toward it beyond logit ≈ 1.2. The GRU's reward
   kernels match the animals' at both ends (1.30 vs 1.33 and 0.59 vs 0.53 one back).
6. **The bias axis shifts occupancy, not structure:** it moves time between the two choice
   lobes (mean P(right) 0.37 vs 0.64, model and animal alike).
7. **Within-mouse change (F4): tracked to the noise ceiling.** Across mice, the GRU's
   early→late change correlates with the animal's at 0.92 (bias), 0.59 (gain) and 0.93
   (persistence) — each at or above the ceiling a perfect model would reach given the
   animal-side sampling noise (0.85, 0.59, 0.90). The one discrepancy is the *average* gain
   shift: animals +0.05, GRU −0.035 (−0.055 when refit on choices sampled from the GRU, so not
   a soft-label effect); with six features × two splits tested, the animals' p-values (0.003 all
   sessions, 0.021 within one task) are suggestive rather than conclusive.
8. **The two embedding axes act on different parts of the dynamics (F2).**
   - **Attractors are robust.** For each mouse and input, 96 starts from the mouse's own
     states were searched. 612 of 614 mice have a stable attractor for all four inputs on
     the chosen side; for 2 mice no "left rewarded" search converged.
     - The medians are R+ +4.36, R− +3.08, L+ −3.72, L− −3.07.
     - "L rewarded" is bistable in 78 mice, a lower bound: at the all-sessions mean context
       this search finds −4.04 but misses D's second point, −4.98, which D found with a
       fuller search. "L unrewarded" is bistable in 1 mouse; R+ and R− in none.
     - *(Corrected 2026-10-26: the first run used 32 starts and was reported as "every
       mouse", "614", "52 bistable".)*
   - **Bias is a shift of the attractors.** It correlates with the mean attractor logit
     (Spearman ρ +0.57) and with the rewarded-minus-unrewarded depth (+0.61). The second
     relation is not yet explained.
   - **Reward gain is a speed, not a depth.** Attractor depth barely tracks gain (ρ +0.28;
     the rewarded-minus-unrewarded depth gives −0.05).
     - Gain tracks v's one-step retention at the attractor, e_vᵀ J e_v (−0.46): high-gain
       mice let v relax faster.
     - That retention agrees with I5's fitted λ_stay (+0.41). It sits at about 0.70 for
       every mouse (10th–90th percentile 0.693–0.706): a small range, but well set by the
       embedding (rank R² 0.80).
     - Persistence tracks depth (+0.50).
   - **D2 reconciled.** D2's "gain deepens both attractors" compared contexts chosen along
     F3's combined gain + persistence direction, so the deeper attractors there go with the
     persistence part.
   - **Slow modes are generic.** τ is about 270–890 trials, with readout overlap about 0.11.
     They are barely set by the embedding (rank R² 0.14), so they are a property of the
     network, not of the mouse.
9. **The embedding encodes the choice strategy, and setup only through behaviour (F5).**
   - **Strongly decodable** from the 4-d embedding (rank R² 0.5–0.85):
     - the dynamics (attractor depth and asymmetry, v retention);
     - most of the GRU's and the animals' kernel features (the 2/1-back decay ratios are the
       exception: GRU 0.39, animal 0.03);
     - every RL bias;
     - most RL learning rates (Hattori's α_rew 0.49 and Bari's forgetting 0.16 are weaker).
   - **Moderately decodable:** foraging efficiency (0.31) and the GRU's per-mouse advantage
     over CTT (0.26).
   - **Not decodable** (≤ 0.08), so the embedding is about how mice choose, not about these:
     reaction time, weight, daily water, ignore and finished rates, the true reward contrast,
     session length, and number of sessions.
   - **A dictionary on the embedding's own axes:**
     - **PC3 (14% of variance) is the learning-rate / gain axis.** Bari and CTT learning
       rates −0.83 / −0.82, GRU reward gain −0.83, Hattori forgetting −0.75, I5's λ_stay
       +0.75.
     - **PC4 (6%) is the bias axis.** RL biasL +0.60 to +0.63, GRU bias −0.57, attractor
       asymmetry −0.77.
     - **PC2 (22%) is attractor depth / persistence** (+0.92).
     - **PC1 (58%) mixes them**, together with how predictable the mouse is (GRU likelihood
       −0.53, Hattori β −0.48).
     - So the largest-variance direction is not a single known parameter. The two named
       axes are the smaller PCs.
   - **Setup labels (test revised; see below).** Institute, data source, room, trainer,
     curriculum and first year are decodable from the embedding above chance (institute
     0.79 against a null of 0.62; trainer 0.18 against 0.13).
     - The task mix alone decodes most of them better (institute 0.93, curriculum 1.00). The
       animals' own kernels do better for institute (0.85) and data source, but not for
       room, curriculum, first year or trainer.
     - **Adding the embedding to animal behaviour + task mix gains nothing** for any label:
       gains −0.05 to +0.028. Room (+0.012, p = 0.05) and PI (+0.028, p = 0.11) are the only
       nominally positive ones, and neither is significant across eight labels.
   - **So setup labels are predictable from the embedding only as far as they are from task
     exposure plus behaviour.** No setup or cohort signature is detectable beyond that. PI
     and rig type are not decodable even from the embedding alone.
   - **Label hygiene:** trainer names are normalised (e.g. "Bowen Tan" and "bowen.tan"
     merged), and missing or "None" values are excluded rather than treated as a class.
   - **Revision.** The first version of this test had three flaws:
     - its "behaviour" was the GRU's own kernels, which is circular;
     - it folded task mix into behaviour;
     - it fed unstandardised features to an L2-penalised logistic, which over-penalised the
       small-scale embedding.

     It now uses the animals' kernels, keeps task mix as a separate control, standardises
     every input, and asks the incremental question with an embedding-shuffle null. The
     conclusion stands.
10. **A continuum with heavy tails, not discrete strategy types (F6).**
    - **Mixture fit.** On the embedding, BIC prefers two Gaussian components over one (gain
      74); in single-Gaussian null samples the best k is always 1 (95th-percentile gain 0).
    - **But the components overlap.** Their means are about 1 SD apart and one is 1.7×
      broader: a compact core (65%) plus a broad halo (35%), not two separate groups.
    - **Heavy tails alone produce this.** A unimodal heavy-tailed null (a multivariate t,
      df matched to the embedding's kurtosis, 8.6) gives gains up to 74 at its 95th
      percentile, so the observed gain is at the edge of what heavy tails explain; the
      result depends on the fitted tail weight.
    - **The animals' own behaviour.** The two components are 97% and 3%. The small one is a
      handful of outlier mice in one feature, the 2/1-back reward-decay ratio (excess
      kurtosis 240; the ratio blows up when the 1-back term is near 0). It is not a type.
    - **Mice close in the embedding behave alike.** Embedding distance correlates with
      behavioural distance: Spearman 0.49 with the animals' kernels, 0.57 with the GRU's
      and 0.55 with the RL parameters, against permutation nulls of about 0.03. That fits a
      smooth map from embedding to strategy.

**Corrections after review (2026-10-19).** The "choice kernel" feature came from the
rank-deficient Hattori2019 design and is replaced by persistence ((rewarded + unrewarded)/2,
Su2022); direction cosines are now reported in the embedding-covariance metric (the earlier
raw-coordinate value 0.19 for bias · gain depended on how the correlated, unequally scaled
embedding dimensions are laid out); F4's "GRU does not track gain changes" was at the noise
ceiling and is withdrawn; F1's description of low-gain mice after no reward was wrong.

11. **Training moves every mouse's context along a shared, front-loaded path, and that explains
    part of the GRU's average change, not its per-mouse changes (F4b, F4c).**
    - **Size: small in variance, not small in distance.** The context (embedding + session
      delta) varies only 3% within mice, against 97% between mice. But that variance share
      understates the movement. A steady drift adds little variance, and the between-mouse
      spread covers all four PCs and has heavy tails. In distance, a mouse's context moves a
      median 0.80 between-mouse SDs along PC2 from its first to its last session (in 99.5%
      of mice), and its per-mouse range along PC4 reaches 1.2 SDs (90th percentile).
      *(Corrected 2026-10-26: an earlier version said the context "barely moves".)*
    - **Shape (F4c, every session of every mouse).** The movement is front-loaded and not
      monotonic. Relative to the first tenth of training, the median mouse moves:
      - PC2 (attractor depth): up, about +0.75 SD by mid-training, then flat;
      - PC4 (bias): up about +0.6 SD within the first 30% of training, then partly back
        (+0.28 at the end);
      - PC3 (learning rate / gain): down about −0.2 SD early, then up (+0.2 at the end).
    - **Direction, as F4b's early-half vs late-half means:**
      - +0.26 between-mouse SDs along embedding PC2, attractor depth (97% of mice);
      - +0.22 along PC3, toward a lower learning rate / gain (90%);
      - −0.11 along PC4 (80% negative). This sign is opposite to the first→last change,
        because the half means average over the early PC4 swing;
      - −0.18 along PC1 (89% negative).
    - **Effect (counterfactual).** Each mouse's late sessions were rerun with its mean early
      context. The context drift accounts for:
      - **Gain:** −0.011 of the GRU's −0.035 median change. That is about a third, and
        opposite to the animals' +0.047. So F4's opposite-sign average gain shift is partly
        the session delta's shared trend.
      - **Bias:** essentially all of the GRU's median change (+0.011 of +0.012).
      - **Persistence:** little (+0.005 of +0.037).
    - **Per mouse.** The context-caused changes point the same way as the GRU's total
      changes (Spearman 0.55 gain, 0.62 persistence, 0.18 bias) and partly as the animals'
      (0.30, 0.49, 0.11). But they hold only 0.4–5.5% of the variance of the GRU's changes:
      most of each mouse's early→late change comes from its input history (tasks, block
      structure), not from the context.
    - **Behavioural reading of the trajectories (F4c, correlational).** Mapping each
      session's context through cross-mouse linear maps to the GRU's kernel features (CV R²
      0.84–0.89) puts most of the implied change in the first 40% of training. A few mice
      move by a sizeable fraction of the between-mouse range, but the median implied change
      is small. The causal estimate is F4b's counterfactual above; the cross-mouse map need
      not hold within a mouse.
    - **Caveat.** The session delta is a function of training phase learned on the train
      sessions, so this measures what the network learned to attribute to phase. It is not
      evidence that the animals' learning follows the same trend.

12. **Interpreting the embedding mouse by mouse: three named strategy axes and one dynamics-only
    axis (F7–F9).**
    - **Feature library (F7).** Thirteen behavioural features per mouse, from the animal's
      choices and the GRU's predictions, each with a split-half (odd/even session)
      reliability. They include determinism (the overall scale of history dependence),
      stickiness ratio UnrC/RewC, win-stay, lose-stay, reward and choice memory spans, bias,
      matching sensitivity and bias, poorer-side rate, switch rate, and the two C4e
      signatures.
    - **Stickiness, gain and persistence were one dimension.** They correlate 0.80–0.95
      across mice in the GRU: all scale with determinism, like an RL β. Shape features
      separate them:
      - the stickiness *ratio* is nearly uncorrelated with determinism in the animals
        (+0.1);
      - and it is still decodable from the embedding after determinism is removed (GRU
        0.36).
    - **What the embedding carries, against each feature's reliability.**
      - Animal determinism, win-stay, lose-stay, switch rate, bias and matching bias:
        rank R² 0.6–0.74, close to their reliability² (0.6–0.88).
      - The animals' stickiness ratio is reliable (0.73) but barely decodable (0.10): the
        GRU captures stickiness mostly through determinism.
      - Matching sensitivity (0.13 of 0.44) and the reward memory span (0.03 of 0.15, itself
        unreliable) are weakly carried.
    - **Named axes.** A varimax rotation of the GRU features' loadings on the whitened
      embedding gives:
      1. **Determinism / stickiness** (loadings: determinism +0.91, lose-stay +0.90,
         win-stay +0.85, switch rate −0.89, stay through losses +0.78; GRU likelihood
         +0.88). History-driven, sticky mice against noisy, switchy ones.
      2. **Side bias** (bias +0.92, matching bias +0.88).
      3. **Slow learning / long choice memory** (choice memory span +0.66, poorer-side
         rate +0.58, return after a failed switch +0.45, stickiness ratio +0.42; RL learning
         rates −0.70 to −0.74, I5 λ_stay +0.67). Mice that integrate choices over longer
         windows, adapt slowly to a new richer side, and return after failed switches.
      4. **A dynamics-only axis** (v retention −0.62, attractor asymmetry +0.54) with almost
         no behavioural loading (R² sum 0.0).
    - **The axes are causal (F8).** Synthetic mice at −2…+2 SD along each axis (GRU run on
      600 real sessions):
      - On axes 1–3, the features change as the loadings predict: causal slopes vs loadings
        correlate 0.94, 0.92 and 0.78. Examples: axis 1 raises determinism and lose-stay
        and lowers switching; axis 2 shifts bias only; axis 3 raises the stickiness ratio
        and choice span.
      - Axis 4 changes the reward memory span and reduces the stickiness ratio a little
        (0.71). It is a real but subtle direction.
    - **Per-mouse profiles (F9).**
      - `f9_mouse_strategy_profiles.csv` lists every mouse's axis coordinates and
        percentiles with its animal features.
      - `fig_F9_strategy_prototypes.png` shows the mice at the ends of each axis.
      - `_cache/explore/cards/` holds one strategy card per mouse. Regenerate any mouse
        with `python analysis/f9_strategy_cards.py --mouse <id>`.
    - **Caveats.**
      - The axes are a rotation chosen to be interpretable, not unique; the names come from
        loadings.
      - The axes are orthogonal **by construction** (varimax is an orthogonal rotation of the
        whitened embedding). The behaviours they are named after are not: determinism and the
        RL learning rate are 52° apart in the embedding (item 13). *(Added 2026-10-27.)*
      - Axis 1 partly measures how predictable the mouse is (GRU likelihood +0.88).
      - The causal test uses the GRU's own behaviour, open-loop on real sessions.

13. **Where each behavioural aspect lives in the embedding (F10, 2026-10-27).** F7's 13
    features mapped over the embedding's PCs (`fig_F10a`–`e`).
    - **Stickiness is one direction spread over three PCs.** Determinism, win-stay,
      lose-stay and staying through losses all rise toward −PC1/+PC2/−PC3, and switch rate
      falls. In PC1–PC2 this is a diagonal gradient; on PC3 it is a band.
    - **So the stickiness family is not a single PC.** PC1 (58%) and PC2 (22%) mix it with
      bias (item 9). Its strength is near the reliability ceiling (animal CV rank R²
      0.61–0.74 against reliability² 0.83–0.88).
    - **Slow learning is PC3 alone.** Choice memory span, the poorer-side rate and return
      after a failed switch take ≥ 75% of their decodable variance from PC3 (94%, 94%, 76%),
      and little from PC1–PC2.
    - **The stickiness ratio's shape part is also on PC3.** UnrC/RewC points +PC3, against
      −PC3 for determinism and win-stay. So PC3 trades reward-driven stickiness for
      outcome-independent repetition and longer choice memory. This is the
      determinism-free component F7 found.
    - **Bias is the other PC1–PC2 diagonal plus PC4.** It is orthogonal to stickiness in
      PC1–PC2. PC4 carries about 30% of the bias and matching-bias variance and almost
      nothing else.
    - **Are stickiness, learning rate and bias orthogonal? Only bias is.** Cosines between
      the animal features' embedding directions, in the embedding-covariance metric (held-out
      predictions agree within 0.08; raw features across mice in brackets):

      | pair | cosine | angle | raw Spearman |
      |---|---:|---:|---:|
      | bias · determinism | −0.11 | 96° | −0.06 |
      | bias · CTT learning rate | +0.18 | 80° | +0.18 |
      | bias · reward gain | −0.10 | 96° | −0.05 |
      | determinism · CTT / Bari learning rate | +0.61 / +0.63 | 52° / 51° | +0.53 / +0.55 |
      | determinism · reward gain | +0.84 | 33° | +0.65 |
      | determinism · persistence | +0.96 | 16° | +0.89 |
      | determinism · switch rate | −1.00 | 180° | −0.98 |
      | CTT learning rate · choice memory span | −0.96 | 164° | −0.72 |
      | CTT learning rate · reward gain | +0.92 | 23° | +0.59 |

      - **Bias is its own axis,** within 10° of orthogonal to everything else, in the
        embedding and in raw behaviour alike.
      - **Learning rate, reward gain and choice memory span are one axis** (|cosine|
        0.92–0.96).
      - **Stickiness (determinism) is a second axis that is *not* orthogonal to the first**:
        52° from the learning rate and 33° from reward gain. More history-driven mice also
        learn faster.
      - **Inside the stickiness family the two lean different ways.** Win-stay leans toward
        the learning rate (0.75–0.78); lose-stay leans toward outcome-independent
        repetition (0.47 with the stickiness ratio, 0.44 with the learning rate).
      - **The embedding aligns these more than raw behaviour does** (0.92 vs 0.59, 0.96 vs
        0.72): it keeps each feature's shared, reliable part and drops the noise.
      - **A display basis for the three aspects (`fig_F10f`).**
        - Prototypes: switch rate for stickiness, the CTT learning rate, and bias.
        - Construction: their embedding directions are orthogonalised in that order, in the
          whitened embedding.
        - Switch rate is axis 1. The learning rate is −0.62 on axis 1 and +0.78 on axis 2,
          so its gradient is a diagonal, not a separate axis. Bias is 0.95 on axis 3. The
          fourth axis carries none of them.
        - So plane 1–2 shows the stickiness/learning-rate pair at its true angle, and
          plane 1–3 or 2–3 shows bias as an independent gradient.
      - *(Corrected 2026-10-27: the first version of this item, and FIGURES' F10c entry,
        described stickiness, slow learning and bias as three separate directions. F7's
        named axes are orthogonal only by construction, since varimax is an orthogonal
        rotation; the behavioural directions they are named after are not.)*
    - **In raw units (held-out deciles, F10e).** From the bottom to the top decile of the
      embedding prediction:
      - win-stay 0.81 → 0.97, lose-stay 0.76 → 0.91 and switch rate 0.07 → 0.21;
      - choice memory span 0.6 → 1.2; bias −0.28 → +0.30.
      - These are 86–95% of what a perfect sort reaches (stickiness core and bias), 64%
        for staying through losses, 45% for the animals' stickiness ratio and matching
        sensitivity, and 18% for reward memory span.
    - **What the embedding misses in the animals:** stickiness ratio (CV 0.10 of
      reliability² 0.53), matching sensitivity (0.13 of 0.44) and reward memory span (0.03
      of 0.15). The GRU's versions of these are mapped more strongly (0.24–0.52), so the
      model expresses them through the embedding more cleanly than the animals do.
    - **A side finding.** The GRU's matching sensitivity sits about 0.1 below the animals'
      at every decile, with the same slope: the GRU undermatches more than the mice.
    - **What is by construction, and what is a finding.**
      - **Partly by construction:** the GRU maps. The GRU's features are a function of the
        embedding, given the mouse's own sessions.
      - **Findings:** the animal maps and the held-out tuning. The directions are fitted
        by cross-validation on the animals' features.
      - **Caveats.** Each 2-d map averages over the two PCs not shown, and the PC basis is
        F5's (variance-ordered, not unique).

14. **Example mice along each prototype gradient (F11, 2026-10-27).**
    - **Selection.** One pair per prototype (switch rate, CTT learning rate, bias). Both mice
      have Uncoupled Baiting as their dominant task, with ≥ 3,000 trials in it (330 mice).
      - Low: embedding-predicted percentile ≤ 0.2 and animal ≤ 0.3.
      - High: predicted ≥ 0.8 and animal ≥ 0.7.
      - The other two prototypes: predicted percentiles within 0.2–0.8. The chosen pairs
        differ by ≤ 0.03 on them.
    - **Switch rate** (802252: 0.06 vs 795711: 0.24). The low mouse stays for long runs:
      P(run ≥ 40) is about 0.15, against about 10⁻³ for the high mouse.
    - **Bias** (756566: −0.29 vs 729678: +0.38). P(right) at each value of pR − pL (−0.6 … +0.6)
      is 0.21 → 0.52 for the low-bias mouse and 0.56 → 0.75 for the high-bias mouse. The
      curves are offset by 0.21–0.35 at every value difference.
      - At equal value (0): 0.40 vs 0.61.
      - The high-bias mouse chooses right even when left is much better (0.56 at −0.6).
      - The low-bias mouse barely prefers right when right is better (0.52 at +0.6).
      - *(Corrected 2026-10-27: the first version binned pR − pL with edges at ±0.3, which split
        the ±0.3 condition by float rounding and drew a spurious peak at +0.2.)*
    - **Learning rate** (806664: 0.17 vs 789197: 0.39). After a reward-probability change (pR − pL jumps by ≥ 0.3),
      the fast learner reaches about 0.7 on the newly favoured side by trial 15; the slow
      learner reaches about 0.4.
      - **Population check:** across the 330 mice, this late rise correlates with the
        embedding-predicted learning rate at ρ = 0.45 (with CTT's fitted rate, 0.38).
      - **Caveat:** a single pair is an illustration, not evidence. The first draft pair
        (761407, 36 usable changes) showed no difference. The pair now requires ≥ 40 changes
        per mouse; selection is still by embedding and fitted rate, never by the curve.
    - **The same six mice in the GRU's state space (F13, B3f's view).**
      - **Switch rate:** the low-switch mouse fills the outer, confident ends of both lobes
        (P(right) near 0 or 1). The high-switch mouse stays nearer the middle, with paler
        P(right).
      - **Bias:** each bias mouse fills its preferred side's lobes densely and deeply; the other
        side's are sparse and pale.
      - **Learning rate:** the two mice occupy the space similarly. Adaptation speed is not
        visible in where the state sits in PC1–PC2.
      - Point density also scales with trial count (4,488–18,981).

15. **Does a mouse's path through context space follow its behavioural change? Only for switch
    rate, and only partly (F12, 2026-10-27).**
    - **What moves.** The subject embedding is fixed per mouse. What changes over training is the
      context, embedding + δ(embedding, phase) with phase = session / number of sessions (F4c).
      So each mouse's path is a smooth, phase-driven curve fixed by its embedding: the network's
      expected learning curve for that kind of mouse. It cannot react to individual sessions.
    - **Test.**
      - Each eval session's context is mapped to predicted behaviour by a cross-mouse linear map
        from mean context to the animal's feature (dominant task only). Cross-validated R²:
        switch rate 0.80, side preference 0.65, adaptation after a reward-probability change
        0.17.
      - Within each mouse (191 mice with ≥ 15 eval sessions), that predicted curve is correlated
        with the mouse's measured per-session feature.
      - The null pairs each mouse's measured series with another mouse's predicted curve at the
        same training phases. It keeps the shared training trend and breaks the mouse-specific
        path.

      | feature | median within-mouse ρ | null median, 95th pct | share ρ > 0 |
      |---|---:|---:|---:|
      | switch rate | 0.18 | 0.09 | 73% |
      | side preference | 0.02 | 0.03 | 51% |
      | adaptation | 0.04 | 0.04 | 55% |

    - **Switch rate beats the null.** The context path carries mouse-specific information about
      how a mouse's switch rate changes, beyond the shared training trend.
    - **Side preference does not** (0.02 against 0.03).
    - **Adaptation is at the null:** 0.042 against 0.040, nominally just above it but
      negligible, and its cross-mouse map is weak (R² 0.17).
    - For both, the predicted paths are nearly flat within a mouse.
    - **Example (`fig_F12`).** Mouse 642221 (Coupled Baiting, 15 eval sessions) switches less
      as training goes on: 0.23 → 0.03.
      - Its context moves along axis 1 toward the low-switch side.
      - The context-predicted switch rate falls 0.19 → 0.12, early in training.
      - The GRU tracks the full change, so most of it comes from the session's inputs, not the
        context. This agrees with F4b: the context holds ≤ 6% of the variance of each mouse's
        change.
      - The mouse is the best of 191 (largest ρ × predicted range), so it is an illustration.
        The population row is the evidence.

<!-- BEGIN result-5 -->
[regenerated by `analysis/report_r4_r5.py` — do not edit by hand]

### F3 What the embedding carries (614 mice)

| feature | rank R² | task mix only | embedding beyond task mix |
|---|---:|---:|---:|
| gru bias | 0.782 | -0.020 | 0.777 |
| gru reward gain (1 back) | 0.780 | 0.128 | 0.619 |
| gru reward effect (sum 1–5) | 0.781 | 0.151 | 0.607 |
| gru reward decay (2/1 back) | 0.391 | 0.097 | 0.283 |
| gru persistence (1 back) | 0.805 | 0.185 | 0.603 |
| gru rewarded choice (1 back) | 0.811 | 0.173 | 0.615 |
| gru unrewarded choice (1 back) | 0.778 | 0.200 | 0.574 |
| animal bias | 0.648 | -0.022 | 0.643 |
| animal reward gain (1 back) | 0.518 | 0.082 | 0.390 |
| animal reward effect (sum 1–5) | 0.581 | 0.118 | 0.431 |
| animal reward decay (2/1 back) | 0.028 | -0.007 | 0.021 |
| animal persistence (1 back) | 0.704 | 0.159 | 0.517 |
| animal rewarded choice (1 back) | 0.727 | 0.149 | 0.542 |
| animal unrewarded choice (1 back) | 0.518 | 0.151 | 0.354 |
| bari learn_rate | 0.653 | 0.185 | 0.536 |
| bari forget_rate_unchosen | 0.164 | 0.014 | 0.142 |
| bari choice_kernel_relative_weight | 0.303 | 0.257 | 0.162 |
| bari biasL | 0.737 | -0.017 | 0.733 |
| bari softmax_inverse_temperature | 0.233 | 0.209 | 0.214 |
| ctt learn_rate | 0.673 | 0.166 | 0.568 |
| ctt threshold | 0.410 | 0.039 | 0.382 |
| ctt softmax_inverse_temperature | 0.326 | 0.034 | 0.295 |
| ctt biasL | 0.728 | -0.017 | 0.724 |
| hattori learn_rate_rew | 0.495 | 0.072 | 0.371 |
| hattori learn_rate_unrew | 0.516 | 0.145 | 0.431 |
| hattori forget_rate_unchosen | 0.582 | 0.075 | 0.469 |
| hattori biasL | 0.732 | -0.017 | 0.728 |
| hattori softmax_inverse_temperature | 0.469 | 0.173 | 0.304 |

All features beat the 200-permutation shuffle null (p = 1/201, the minimum).
Embedding-direction cosines in the embedding-covariance metric (= correlation of the embedding-predicted features), with the Spearman correlation of the features themselves across mice in brackets: gru bias · gru reward gain (1 back) = -0.05 [-0.03]; gru bias · ctt biasL = -1.00 [-0.90]; gru reward gain (1 back) · gru persistence (1 back) = +0.94 [+0.90]; gru reward gain (1 back) · ctt learn_rate = +0.92 [+0.79]; gru bias · gru persistence (1 back) = -0.08 [-0.05].
GRU vs animal per-mouse Spearman: bias 0.95, reward gain (1 back) 0.80, reward effect (sum 1–5) 0.85, reward decay (2/1 back) 0.36, persistence (1 back) 0.96, rewarded choice (1 back) 0.96, unrewarded choice (1 back) 0.86.

![F3](../fig_F3_embedding_features.png)

### F1 Mice at the ends of the gain and bias directions

10% at each end (61 mice). Reward effect 1 back, median over mice: low gain GRU 0.59 / animal 0.53; high gain GRU 1.30 / animal 1.33. Mean P(right), bias low vs high: model 0.374 vs 0.637; animal 0.374 vs 0.635.

![F1](../fig_F1_embedding_extremes.png)

### F4 Early vs late sessions within each mouse

| split | feature | mice | Spearman Δ GRU vs Δ animal | noise ceiling | median Δ animal (p) | median Δ GRU (p) | median Δ GRU sampled |
|---|---|---:|---:|---:|---:|---:|---:|
| all sessions | bias | 533 | 0.92 | 0.85 | +0.030 (0.0164) | +0.012 (0.00673) | +0.023 |
| all sessions | reward gain (1 back) | 533 | 0.59 | 0.59 | +0.047 (0.00305) | -0.035 (1.24e-09) | -0.055 |
| all sessions | persistence (1 back) | 533 | 0.93 | 0.90 | +0.005 (0.595) | +0.037 (0.00566) | +0.028 |
| dominant task only | bias | 465 | 0.92 | 0.83 | +0.019 (0.212) | +0.012 (0.0232) | +0.022 |
| dominant task only | reward gain (1 back) | 465 | 0.56 | 0.54 | +0.028 (0.0207) | -0.024 (4.02e-05) | -0.036 |
| dominant task only | persistence (1 back) | 465 | 0.91 | 0.87 | -0.010 (0.385) | -0.000 (0.801) | -0.012 |

![F4](../fig_F4_early_late.png)

### F2 Attractors and timescales of every mouse's context

612 mice × 4 inputs, 96 starts each (forward iteration + damped Newton, float64). Converged fraction: L+ 0.958, L− 1.000, R+ 1.000, R− 1.000. Bistable mice per input: R+ 0, R− 0, L+ 78, L− 1. Mean-context attractor logits {'R+': [4.34], 'R−': [3.05], 'L+': [-4.04], 'L−': [-3.05]} (D2: {'R rewarded': [4.34], 'R unrewarded': [3.05], 'L rewarded': [-4.98, -4.04], 'L unrewarded': [-3.05]}).

| feature | 10th / 50th / 90th pct | embedding rank R² |
|---|---|---:|
| attractor depth | 3.2 / 3.55 / 3.93 | 0.85 |
| attractor asymmetry | -0.124 / 0.152 / 0.479 | 0.82 |
| reward depth (rew − unrew) | 0.496 / 1.07 / 1.3 | 0.64 |
| bistable (any input) | 0 / 0 / 1 | — |
| slowest τ (trials) | 267 / 355 / 890 | 0.13 |
| slow-mode readout overlap | 0.0719 / 0.107 / 0.133 | 0.21 |
| v retention (e_vᵀJe_v) | 0.693 / 0.701 / 0.706 | 0.80 |
| attractor logit R+ | 3.62 / 4.36 / 5.14 | — |
| attractor logit R− | 2.59 / 3.08 / 3.53 | — |
| attractor logit L+ | -4.22 / -3.72 / -3.16 | — |
| attractor logit L− | -3.44 / -3.07 / -2.74 | — |

Spearman with the GRU's kernel features (F3):

| | bias | reward gain (1 back) | persistence (1 back) | reward decay (2/1 back) |
|---|---:|---:|---:|---:|
| attractor depth | +0.33 | +0.28 | +0.50 | +0.27 |
| attractor asymmetry | +0.57 | +0.32 | +0.39 | +0.30 |
| reward depth (rew − unrew) | +0.61 | -0.05 | -0.13 | -0.34 |
| bistable (any input) | -0.33 | +0.24 | +0.29 | +0.32 |
| slowest τ (trials) | +0.04 | +0.40 | +0.36 | +0.12 |
| slow-mode readout overlap | -0.19 | +0.10 | +0.18 | +0.29 |
| v retention (e_vᵀJe_v) | +0.37 | -0.46 | -0.38 | -0.42 |

Spearman with I5's per-mouse dynamics fits:

| | λ stay | λ switch | reward push | choice push | target asymmetry | switch target |
|---|---:|---:|---:|---:|---:|---:|
| attractor depth | +0.09 | -0.21 | +0.47 | +0.68 | +0.48 | +0.33 |
| attractor asymmetry | +0.01 | -0.21 | +0.45 | +0.28 | +0.58 | +0.36 |
| reward depth (rew − unrew) | +0.06 | +0.13 | +0.07 | -0.10 | +0.61 | -0.09 |
| bistable (any input) | -0.19 | -0.25 | +0.18 | +0.02 | -0.28 | +0.25 |
| slowest τ (trials) | -0.27 | -0.34 | +0.29 | +0.01 | +0.10 | +0.35 |
| slow-mode readout overlap | +0.01 | -0.13 | +0.11 | +0.19 | -0.19 | +0.14 |
| v retention (e_vᵀJe_v) | +0.41 | +0.42 | -0.17 | +0.25 | +0.36 | -0.45 |

![F2](../fig_F2_mouse_dynamics.png)

### F5 Embedding atlas, dictionary and batch effects

614 mice; embedding PC variance shares 58%, 22%, 14%, 6%; session metadata matched for a median 100% of sessions.

Decodability from the 4-d embedding (rank R², 5-fold CV; shuffle 95th percentile ≈ 0):

| variable | rank R² | | variable | rank R² |
|---|---:|---|---|---:|
| attractor depth | 0.85 | | hattori learn_rate_rew | 0.49 |
| attractor asymmetry | 0.82 | | hattori softmax_inverse_temperature | 0.47 |
| gru rewarded choice (1 back) | 0.81 | | ctt threshold | 0.41 |
| gru persistence (1 back) | 0.81 | | gru reward decay (2/1 back) | 0.39 |
| v retention (e_vᵀJe_v) | 0.80 | | ctt softmax_inverse_temperature | 0.33 |
| gru bias | 0.78 | | foraging_eff | 0.31 |
| gru reward effect (sum 1–5) | 0.78 | | bari choice_kernel_relative_weight | 0.30 |
| gru reward gain (1 back) | 0.78 | | gru advantage over CTT | 0.26 |
| gru unrewarded choice (1 back) | 0.78 | | bari softmax_inverse_temperature | 0.23 |
| I5 switch target | 0.75 | | bari forget_rate_unchosen | 0.16 |
| bari biasL | 0.74 | | task share: Uncoupled Without Baiting | 0.13 |
| hattori biasL | 0.73 | | task share: Coupled Baiting | 0.11 |
| ctt biasL | 0.73 | | task share: Coupled Without Baiting | 0.09 |
| animal rewarded choice (1 back) | 0.73 | | effective_block_length_mean | 0.08 |
| gru likelihood | 0.73 | | total_trials | 0.08 |
| animal persistence (1 back) | 0.70 | | session_run_time_in_min | 0.05 |
| ctt learn_rate | 0.67 | | animal reward decay (2/1 back) | 0.03 |
| bari learn_rate | 0.65 | | reaction_time_median | 0.03 |
| animal bias | 0.65 | | water_day_total | 0.02 |
| hattori forget_rate_unchosen | 0.58 | | base_weight | 0.01 |
| animal reward effect (sum 1–5) | 0.58 | | task share: Uncoupled Baiting | 0.00 |
| I5 λ stay | 0.56 | | p_reward_contrast_mean | -0.00 |
| animal unrewarded choice (1 back) | 0.52 | | finished_rate | -0.00 |
| animal reward gain (1 back) | 0.52 | | ignore_rate | -0.00 |
| hattori learn_rate_unrew | 0.52 | | n eval sessions | -0.02 |

Dictionary (Spearman with the embedding PCs, decodable variables):

| variable | PC1 | PC2 | PC3 | PC4 |
|---|---:|---:|---:|---:|
| bari learn_rate | -0.13 | -0.01 | -0.83 | -0.11 |
| bari forget_rate_unchosen | -0.05 | -0.08 | -0.36 | -0.07 |
| bari choice_kernel_relative_weight | -0.18 | +0.27 | -0.44 | -0.00 |
| bari biasL | -0.47 | -0.39 | -0.03 | +0.63 |
| bari softmax_inverse_temperature | -0.16 | +0.18 | +0.43 | +0.04 |
| ctt learn_rate | +0.01 | +0.10 | -0.82 | -0.25 |
| ctt threshold | -0.19 | -0.31 | -0.51 | +0.07 |
| ctt softmax_inverse_temperature | -0.39 | +0.18 | -0.44 | -0.05 |
| ctt biasL | -0.49 | -0.40 | -0.01 | +0.60 |
| hattori learn_rate_rew | -0.23 | +0.22 | -0.65 | -0.03 |
| hattori learn_rate_unrew | -0.09 | -0.16 | -0.71 | -0.10 |
| hattori forget_rate_unchosen | -0.14 | +0.03 | -0.75 | -0.14 |
| hattori biasL | -0.50 | -0.40 | -0.01 | +0.60 |
| hattori softmax_inverse_temperature | -0.48 | +0.39 | -0.43 | -0.06 |
| gru bias | +0.55 | +0.43 | -0.01 | -0.57 |
| gru reward gain (1 back) | -0.31 | +0.20 | -0.83 | -0.14 |
| gru reward effect (sum 1–5) | -0.41 | +0.24 | -0.77 | -0.12 |
| gru reward decay (2/1 back) | -0.55 | +0.20 | -0.34 | -0.05 |
| gru persistence (1 back) | -0.45 | +0.41 | -0.68 | -0.06 |
| gru rewarded choice (1 back) | -0.42 | +0.37 | -0.73 | -0.08 |
| gru unrewarded choice (1 back) | -0.49 | +0.49 | -0.57 | -0.03 |
| animal bias | +0.51 | +0.38 | +0.00 | -0.51 |
| animal reward gain (1 back) | -0.28 | +0.15 | -0.66 | -0.12 |
| animal reward effect (sum 1–5) | -0.36 | +0.23 | -0.65 | -0.07 |
| animal persistence (1 back) | -0.41 | +0.36 | -0.65 | -0.07 |
| animal rewarded choice (1 back) | -0.39 | +0.32 | -0.70 | -0.09 |
| animal unrewarded choice (1 back) | -0.39 | +0.40 | -0.47 | -0.01 |
| attractor asymmetry | -0.18 | +0.50 | -0.10 | -0.77 |
| attractor depth | -0.20 | +0.92 | -0.01 | -0.12 |
| v retention (e_vᵀJe_v) | +0.59 | +0.50 | +0.38 | +0.28 |
| I5 λ stay | +0.12 | +0.11 | +0.75 | -0.05 |
| I5 switch target | -0.34 | +0.26 | -0.79 | -0.13 |
| gru likelihood | -0.53 | +0.53 | -0.46 | -0.03 |
| gru advantage over CTT | +0.24 | -0.30 | +0.31 | -0.00 |
| foraging_eff | +0.24 | -0.40 | -0.31 | +0.01 |
| task share: Coupled Baiting | +0.13 | -0.18 | +0.22 | -0.01 |
| task share: Uncoupled Without Baiting | -0.22 | +0.29 | -0.11 | -0.01 |

Categorical metadata, balanced accuracy (label-shuffle 95th pct); last column: gain from adding the embedding to animal behaviour + task mix (embedding-shuffle 95th pct, p):

| variable | classes | embedding | task mix | animal behaviour | behaviour + task mix | embedding gain |
|---|---:|---:|---:|---:|---:|---:|
| institute | 2 | 0.79 (0.62) | 0.93 (0.59) | 0.85 (0.61) | 0.94 (0.61) | -0.015 (0.009, p 0.41) |
| data_source | 4 | 0.51 (0.31) | 0.62 (0.30) | 0.53 (0.30) | 0.67 (0.30) | -0.004 (0.018, p 0.40) |
| rig_type | 2 | 0.53 (0.59) | 0.44 (0.60) | 0.57 (0.60) | 0.50 (0.58) | -0.051 (0.061, p 0.97) |
| room | 4 | 0.54 (0.30) | 0.62 (0.31) | 0.50 (0.32) | 0.69 (0.30) | +0.012 (0.001, p 0.05) |
| trainer | 10 | 0.18 (0.13) | 0.28 (0.13) | 0.17 (0.13) | 0.31 (0.13) | -0.005 (-0.023, p 0.02) |
| PI | 5 | 0.20 (0.25) | 0.51 (0.25) | 0.35 (0.26) | 0.50 (0.27) | +0.028 (0.036, p 0.11) |
| curriculum_name | 2 | 0.70 (0.57) | 1.00 (0.56) | 0.61 (0.56) | 1.00 (0.56) | +0.000 (0.000, p 1.00) |
| first_year | 5 | 0.33 (0.26) | 0.45 (0.24) | 0.31 (0.24) | 0.52 (0.24) | -0.030 (0.005, p 0.70) |

![F5a](../fig_F5a_embedding_atlas.png)

![F5a PC3–4](../fig_F5a_embedding_atlas_pc3_pc4.png)

![F5b](../fig_F5b_embedding_directions.png)

![F5c](../fig_F5c_embedding_batch.png)

### F6 Types or a continuum?

| data | best k (BIC) | BIC gain over k = 1 | Gaussian null 95% | heavy-tailed null (t, df) median / 95% | k = 2 weights | k = 2 mean separation (Mahalanobis) | k = 2 covariance-scale ratio |
|---|---:|---:|---:|---|---|---:|---:|
| embedding | 2 | 74 | 0 | 2 / 74 (df 8.6) | 0.65/0.35 | 0.97 | 1.70 |
| animal behaviour | 2 | 856 | 0 | 285 / 597 (df 4.2) | 0.97/0.03 | 2.66 | 1.48 |

Mantel (Spearman, embedding Mahalanobis distance vs feature distance; permutation 95th pct): animal behaviour 0.49 (0.03), GRU behaviour 0.57 (0.03), RL parameters 0.55 (0.04).

![F6](../fig_F6_embedding_types.png)

### F4b How the context moves with training, and what it does

Split: all sessions (F4's median split). Share of context variance within mice: 3.0%. Replay parity (median |ΔP|): 1.7e-05.

| embedding PC | median drift (between-mouse SDs) | median |drift| | share of mice positive | Wilcoxon p |
|---|---:|---:|---:|---:|
| PC1 | -0.18 | 0.18 | 11% | 2.0e-75 |
| PC2 | +0.26 | 0.26 | 97% | 1.1e-100 |
| PC3 | +0.22 | 0.22 | 90% | 1.5e-89 |
| PC4 | -0.11 | 0.13 | 20% | 5.5e-48 |

Counterfactual (late sessions rerun with the mouse's mean early-half context; Su2022 soft-label refit on late rows), 533 mice:

| feature | median Δ caused by context | median Δ GRU total | median Δ animal | Spearman (context, GRU total) | Spearman (context, animal) | variance share of the GRU's change |
|---|---:|---:|---:|---:|---:|---:|
| bias | +0.011 | +0.012 | +0.030 | +0.18 | +0.11 | 0.004 |
| reward gain (1 back) | -0.011 | -0.035 | +0.047 | +0.55 | +0.30 | 0.055 |
| persistence (1 back) | +0.005 | +0.037 | +0.005 | +0.62 | +0.49 | 0.031 |

![F4b](../fig_F4b_context_drift.png)

### F4c Context trajectories over every training session

18,124 session contexts of 614 mice (recomputed by the network; max |Δ| vs the cached eval contexts 2.9e-05). Within-mouse share of context variance: 3.3%.

| embedding PC | median net change first → last (SDs) | median |net| | median path length | median range | 90th pct range | mice net positive |
|---|---:|---:|---:|---:|---:|---:|
| PC1 (mixed) | -0.24 | 0.27 | 0.44 | 0.34 | 0.50 | 20% |
| PC2 (attractor depth) | +0.80 | 0.80 | 0.93 | 0.86 | 1.17 | 100% |
| PC3 (learning rate / gain) | +0.12 | 0.30 | 0.85 | 0.56 | 0.96 | 61% |
| PC4 (bias) | +0.36 | 0.39 | 1.10 | 0.73 | 1.24 | 83% |

Median change from the first tenth of training, by phase decile:

- PC1 (mixed): +0.00, +0.04, +0.02, -0.10, -0.19, -0.22, -0.24, -0.24, -0.24, -0.24
- PC2 (attractor depth): +0.00, +0.29, +0.50, +0.67, +0.75, +0.78, +0.79, +0.78, +0.77, +0.75
- PC3 (learning rate / gain): +0.00, -0.23, -0.15, -0.07, +0.00, +0.07, +0.11, +0.14, +0.17, +0.20
- PC4 (bias): +0.00, +0.43, +0.59, +0.55, +0.46, +0.39, +0.35, +0.32, +0.30, +0.26

![F4c](../fig_F4c_context_trajectories.png)
<!-- END result-5 -->

## Caveats

- Embedding directions are fitted in raw embedding coordinates (the space the GRU's input
  weights see). The reported cosines use the embedding-covariance metric, which is invariant
  to linear re-parameterisation; raw-coordinate cosines would not be.
- The F4 split is by training-session order within each mouse's eval sessions; part of the
  compression for gain (and the < 1 slopes for bias) is attenuation from noise in the
  animals' estimates — the opposite-sign average shift is not.
- F1 groups are the 10% tails of a GRU-feature direction estimated on the same mice, so the
  GRU-side contrast is circular by construction; only the animal-side contrast is evidence.
