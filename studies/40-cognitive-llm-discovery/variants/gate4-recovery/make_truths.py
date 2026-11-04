"""Gate 4, second half: hand-set generating parameters, the same for every simulated mouse.

Writes a truths file in the form ``mevolve.gate4 --truths`` reads: for each real template mouse of
the generation and validation cohorts, the typical values of ``generator/two_timescale_ck.py`` and
a spread of SPREAD on the probit scale for every parameter. Only the session layout is taken from
the real mice; nothing is fitted. The settings run from mild to strong mechanisms; the first one
whose pre-check passes is used (notes.md).

    python make_truths.py --setting a --cohorts cohorts.json --out truths.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

GENERATOR = Path(__file__).parent / "generator" / "two_timescale_ck.py"
SPREAD = 0.3            # each session's parameters, around the typical values, on the probit scale
SETTINGS = {
    "a": {"rate_fast": 0.6, "rate_slow": 0.05, "weight_fast": 3.0, "weight_slow": 6.0,
          "kernel_rate": 0.3, "kernel_weight": 1.5, "bias_right": 0.2},
    "b": {"rate_fast": 0.7, "rate_slow": 0.03, "weight_fast": 3.0, "weight_slow": 8.0,
          "kernel_rate": 0.3, "kernel_weight": 2.0, "bias_right": 0.2},
    "c": {"rate_fast": 0.8, "rate_slow": 0.02, "weight_fast": 4.0, "weight_slow": 10.0,
          "kernel_rate": 0.3, "kernel_weight": 2.5, "bias_right": 0.2},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--cohorts", type=Path, required=True, help="cohort lists (mevolve.cohorts output)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    params = SETTINGS[args.setting]
    cohorts = json.loads(args.cohorts.read_text())["cohorts"]
    subjects = cohorts["generation"] + cohorts["validation"]
    record = {"typical_params": params, "log_sigma": [math.log(SPREAD)] * len(params), "converged": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "program": GENERATOR.name, "program_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        "setting": args.setting, "spread": SPREAD,
        "cohorts_sha256": hashlib.sha256(args.cohorts.read_bytes()).hexdigest(),
        "subjects": {s: record for s in subjects}}, indent=1) + "\n")
    print(f"setting {args.setting}: {len(subjects)} mice -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
