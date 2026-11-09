# 30-gru-vs-glmhmm-latents — reports

| report | question | verdict |
|---|---|---|
| [r1 — Pilot: GLM-HMM states decoded from GRU latents on training mice](r1-pilot-decoding.md) | Can a linear readout of the GRU latents recover the GLM-HMM state posterior on training mice? | Yes at every K (within-session R² 0.72–0.88; shuffle nulls ≤ 0.03); shared variables, not discrete states (draft) |
