#!/usr/bin/env python3
"""Fail when a tracked file exceeds the size limit (default 10 MiB).

A large file merged into `main` stays in history for good: `main` refuses force-pushes, so
removing it would mean rewriting history. Large inputs belong in S3, a Beaker dataset or a
W&B artifact, referenced by id and digest (posthoc-reporting); the repo holds code,
reports, figures and curated summaries. Git LFS was considered and declined (dispatcher
#180): no AllenNeuralDynamics repo uses it, and it adds a git-lfs dependency to every clone.

ALLOWED lists the existing exceptions, each capped at its size when it was allowed, so an
exception cannot quietly grow. Add an entry only with a reason.

Usage: python3 scripts/check_file_sizes.py [--limit-mib 10]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

MIB = 1024 * 1024
# path -> (byte cap, reason)
ALLOWED = {
    "studies/09-gru-cross-species-transfer/analysis/matched_half_results.json": (
        16_531_384,
        "frozen per-session likelihoods for every matched-half cell (r1-r4 input); predates this gate",
    ),
}


def tracked_sizes() -> list[tuple[int, str]]:
    out = subprocess.run(["git", "ls-tree", "-r", "-l", "-z", "HEAD"], capture_output=True,
                         check=True).stdout.decode()
    sizes = []
    for entry in filter(None, out.split("\0")):
        meta, path = entry.split("\t", 1)
        size = meta.split()[3]
        if size != "-":          # submodules have no size
            sizes.append((int(size), path))
    return sizes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-mib", type=float, default=10.0)
    args = p.parse_args()
    limit = int(args.limit_mib * MIB)
    failed = 0
    for size, path in sorted(tracked_sizes(), reverse=True):
        if size <= limit:
            break
        cap, reason = ALLOWED.get(path, (None, None))
        if cap is not None and size <= cap:
            print(f"allowed  {size / MIB:6.1f} MiB  {path} ({reason})")
            continue
        why = f"over its allowed {cap / MIB:.1f} MiB" if cap is not None else f"over {args.limit_mib:g} MiB"
        print(f"TOO BIG  {size / MIB:6.1f} MiB  {path} ({why})")
        failed += 1
    if failed:
        print(f"\n{failed} file(s) too large. Store large inputs in S3 / a Beaker dataset / a W&B "
              "artifact and commit its id + digest instead; see scripts/check_file_sizes.py.")
        return 1
    print(f"OK: no tracked file over {args.limit_mib:g} MiB outside the allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
