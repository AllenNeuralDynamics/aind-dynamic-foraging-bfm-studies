"""Source revisions of the study 14 GLM-HMM groups that r1 reads, from study 14's own launch records.

build_meta(study_root=<study 30>) looks for launch records in study 30, which has none for these groups.
This reads them where they live, so the freeze and the producer carry the wrapper / dispatcher /
foraging-models SHAs of the runs, and lists the record files as digest-tracked inputs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import _source_shas  # noqa: E402

STUDY14 = HERE.parents[1] / "14-glmhmm-vs-gru-heldout"
GLM_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
RECORDS = [STUDY14 / "variants/hurdle/launch_record/hurdle-k1-10/hpc_sweep.json",
           STUDY14 / "variants/reference/launch_record/reference-k1-10/hpc_sweep.json"]
GRU_NOTE = ("GRU run nxd-3way-20260707-075401-80a6d102 (study 02, group nxd-3way@20260707-075401) has no launch "
            "record in study 02; it is read through its W&B artifact and recorded as a wandb_run source")


def apply(meta: dict) -> dict:
    """Replace build_meta's study-30 lookup with study 14's records for the GLM-HMM groups."""
    shas, findings = _source_shas(STUDY14, GLM_GROUPS)
    meta.update(shas)
    meta["provenance_findings"] = findings + [GRU_NOTE]
    return meta
