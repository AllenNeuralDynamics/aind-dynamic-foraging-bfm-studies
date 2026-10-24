Beaker experiment `01M5PRPW26QP67C4SH5Z906TN6` (2026-10-24 03:53 PT, same sweep and
template) was stopped before any of its 15 tasks started: the launch ran from a worktree
whose directory was named `studies`, so the launcher derived `meta.study=studies` from the
first `studies` path component. No W&B run exists for it; `01M5PRT2DQ0RJJ8YV1SV98EW45`
(this record) is the resubmission from a correctly named checkout.
