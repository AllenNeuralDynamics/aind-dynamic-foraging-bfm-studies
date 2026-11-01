# Hill-climb one score, then simplify; behaviour is feedback, never score

The search maximizes one number: a program's pooled held-out likelihood per trial on the validation
subjects, the same axis the GRU ceiling and the cognitive baselines are measured on. A second stage
then minimizes code complexity while keeping the score above a fit floor, as DataDIVER did.
Behavioural fingerprints (win-stay rates, switch curves, history kernels) and closed-loop rollouts are
computed for the best programs, shown to the LLM as feedback, used to keep the population diverse and
to break ties, but never added to the score (decided with Han, 2026-10-31).

## Considered options

- **A weighted sum of likelihood and behavioural distances.** The weights would be arbitrary, summary
  statistics are easy to game, and rollouts for every candidate would multiply evaluation cost.
- **Likelihood only, behaviour checked after the search.** Simplest, but a program that fits choices
  while generating unrealistic behaviour would only be caught at the end.
