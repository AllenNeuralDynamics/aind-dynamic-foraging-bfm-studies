# One launch mints one W&B group; interventions join it

A **launch** (one initial submission of a variant) mints exactly one W&B group named
`<variant>@<launch_id>`, where the variant prefix equals the variant's directory name. Every
later **intervention** on that work (resubmit, rescue, probe, tier-change, backfill, extend)
writes its own launch record but joins the existing group, and reuses the launch's manifests,
differing only in fields listed under `deviations`. Only `extend` may change `n_steps`. Any
other change to scientific parameters is a new launch, usually of a new variant.

We chose this because a report cites groups: if interventions joined their group, a citation
automatically includes every rescue and resubmit, and a group means one configuration.
Minting a group per submission would make citations go stale with every rescue, and need a
`supersedes` chain to be followed for each one.

## Considered options

- **Every submission mints its own group, linked by `supersedes`.** This is what study 01's
  `heldout-rerun-v1-retry@20260623-011405` / `-011406` did. Rejected: citations go stale
  and two groups can differ by one second for the same work.
- **A probe that changes a hyperparameter stays in the group.** Rejected: its runs would
  mix a second configuration into results that cite the group.

## Consequences

- The dispatcher's launchers enforce both rules **before submitting**: a group whose prefix
  is not the variant name, or an intervention whose manifests differ from its launch's in an
  undeclared field, is refused. A W&B group cannot be renamed once runs exist, so a PR-time
  check would come too late. The studies validator re-checks committed records afterwards.
- The existing exceptions are frozen history and go in the provenance baseline, not edited:
  study 01's `-retry` groups, and six groups whose prefix is not their variant name
  (05 `generative-dscan-mult2@…` under `generative-dscan`, 08 `hb-one_stage@…` under
  `one-stage-ladder`).
