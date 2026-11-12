# CONTEXT — study 30 vocabulary

Study-specific terms. The repository vocabulary (study, variant, report, freeze, ...) is in the root `CONTEXT.md`.

## Language

**State**:
A discrete GLM-HMM state z_t. Never a GRU quantity.
_Avoid_: GRU state.

**GRU latents**:
The GRU hidden vector h_t (256 numbers at H256) that produces the prediction for trial t, i.e. after the GRU has
seen trials up to t − 1.
_Avoid_: GRU state, hidden state.

**Embedding**:
The GRU's learned per-mouse vector (4 numbers), fixed within a mouse.

**Predictive posterior**:
p(z_t | trials < t), the GLM-HMM's belief about trial t's state before trial t's choice; filtered_{t−1} · A.
It is aligned with the GRU latents and is the decoding target.

**Filtered posterior**:
p(z_t | trials ≤ t): the predictive posterior updated by trial t's own choice. Study 14 saves it as `filtered_k`.

**Smoothed posterior**:
p(z_t | whole session): uses later trials too.

**Probe** (readout):
A softmax linear map from features to state probabilities, π̂_t = softmax(W·h_t + b), fitted to the posterior as
soft labels by cross-entropy, cross-validated by mouse.

**Pooled R²**:
1 − Σ_k SSE_k / Σ_k SST_k over a posterior's K states; comparable across K.

**Within-session R²**:
Pooled R² after subtracting each session's mean from target and prediction; scores only changes inside a session.

**Information recovered**:
1 − KL(probe) / KL(best constant guess): the share of the gap between guessing the average posterior and
reproducing it exactly that the probe closes.

**Session swap** / **circular shift**:
The two shuffle nulls: pair a session's latents with another mouse's session; rotate a session's target in time.
