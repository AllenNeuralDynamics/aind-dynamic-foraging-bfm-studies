# 12-cognitive-literature-leaderboard

Vocabulary used only in this study. Repo-wide terms are in the root
[CONTEXT.md](../../CONTEXT.md).

## Language

**Literature model** (in full, a model in the literature):
A cognitive model of two-armed-bandit choice that a paper proposed and fit to its own data, as
implemented in `aind-dynamic-foraging-models` (`ForagerCollection.FORAGER_PRESETS`, plus the two
Costa sensitivity variants and the free LossCounting model). This study fits each one per subject on our subjects.
_Avoid_: author model, published model

**Per subject** (regime 1):
One parameter set per subject, fit by maximum likelihood on that subject's fitting sessions alone,
with no prior and nothing shared across subjects (report r1).

**2-level EM** (regime 2):
Each session's parameters are drawn from its subject's distribution (session → subject, no
population level); fit by Laplace-EM under study 13's hyperprior and scored by the session marginal
(report r3). Models are labelled EM-<model>, e.g. EM-CTT. Its NUTS counterpart is 2-level HB
(study 13).
_Avoid_: session level (reads as per-session independent fits, which is "per session")
