#!/usr/bin/env python
"""P1: the library's agent adapters reproduce the models study 15 replays, trial by trial.

foraging-models' ``fingerprints.replay`` drives any agent through MEvolve's program protocol. Its
own tests cover the forager and GLM adapters; a MEvolve program needs jax, which that repository
does not install, so the check on real programs runs here, in the mevolve env:

- ``ProgramAgent``: for ``MODELS`` and the first ``N_SUBJECTS`` test subjects, study 12's
  per-subject parameters replayed through ``replay(ProgramAgent(program, params), ...)`` against
  MEvolve's own ``scoring.point_p_right`` (what ``freeze_p1_replay.py`` tallies);
- ``GLMAgent``: the committed per-subject GLM fits (``p1_glm.json``) replayed through
  ``replay(GLMAgent(...), ...)`` against ``freeze_p1_glm.py``'s design-matrix prediction.

Pass: every per-trial p(right) within ``TOL`` (1e-6). Needs /allen and the mevolve env:

    srun --ntasks=1 --cpus-per-task=8 --mem=32G -p aind --time=01:00:00 \\
      env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
      analysis/freeze_p1_agent_check.py --models-repo <foraging-models> --mevolve-repo <mevolve>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _sha256  # noqa: E402

MODELS = ("Bari2019", "Lebedeva-PR", "CompareToThreshold")
N_SUBJECTS = 3
TOL = 1e-6
GLM = HERE / "p1_glm.json"
OUT = HERE / "p1_agent_check.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    a = ap.parse_args()
    import aind_dynamic_foraging_models

    lib_path = Path(aind_dynamic_foraging_models.__file__).resolve()
    assert (a.models_repo / "src").resolve() in lib_path.parents, f"library from {lib_path}"
    models_sha = _git_sha(a.models_repo)
    from mevolve import fitting, literature, program, scoring
    from mevolve.data import Freeze

    from aind_dynamic_foraging_models.fingerprints import GLMAgent, ProgramAgent, replay
    from freeze_p1_glm import design

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    subjects = freeze.subjects("heldout")[:N_SUBJECTS]
    rows = []
    for name in MODELS:
        prog = program.literature(literature.LITERATURE[name][0])
        params = lit.literature_fitted_params(name)
        for sid in subjects:
            sessions = freeze.subject_sessions(sid, "score")
            p = {k: params[sid][k] for k in prog.params}
            want = scoring.point_p_right(prog, fitting.from_params(prog, p), fitting.pad(sessions))
            diff = max(float(np.max(np.abs(replay(ProgramAgent(prog, p), s.choices, s.rewards)
                                            - np.asarray(want[i, : len(s.choices)]))))
                       for i, s in enumerate(sessions))
            rows.append({"adapter": "ProgramAgent", "model": name, "subject_id": sid,
                         "n_trials": int(sum(len(s.choices) for s in sessions)), "max_abs_diff": diff})
            print(rows[-1], flush=True)
    glm = {r["subject_id"]: r for r in json.loads(GLM.read_text())["subjects"]}
    for sid in subjects:
        r = glm[sid]
        agent = GLMAgent(r["coef_rewc"], r["coef_unrc"], r["bias"])
        w = np.r_[r["coef_rewc"], r["coef_unrc"]]
        sessions = freeze.subject_sessions(sid, "score")
        diff = max(float(np.max(np.abs(replay(agent, s.choices, s.rewards)
                                        - 1 / (1 + np.exp(-(design(s.choices, s.rewards) @ w + r["bias"]))))))
                   for s in sessions)
        rows.append({"adapter": "GLMAgent", "model": json.loads(GLM.read_text())["model"], "subject_id": sid,
                     "n_trials": int(sum(len(s.choices) for s in sessions)), "max_abs_diff": diff})
        print(rows[-1], flush=True)

    worst = max(r["max_abs_diff"] for r in rows)
    meta = build_meta("analysis/freeze_p1_agent_check.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py", HERE / "freeze_p1_glm.py",
                            HERE / "freeze_p1_replay.py"],
                      inputs=[GLM, lit.LITERATURE_SOURCE],
                      sources=[{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST}])
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "mevolve_git_sha": (a.mevolve_repo / "MEVOLVE_GIT_SHA").read_text().strip(),
        "tolerance": TOL, "max_abs_diff": worst, "passed": worst <= TOL, "rows": rows,
    })
    print(f"{'PASS' if worst <= TOL else 'FAIL'} (max {worst:.1e}) -> {OUT}")


if __name__ == "__main__":
    main()
