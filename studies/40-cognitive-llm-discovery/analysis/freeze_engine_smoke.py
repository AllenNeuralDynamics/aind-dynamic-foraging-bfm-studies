#!/usr/bin/env python
"""Freeze gate 3's engine-smoke run into committed files (``make freeze-engine-smoke``).

Reads, read-only, the run directory of ``engine-smoke@20261103-030341`` and the preflight ledger
on /allen, and writes under ``analysis/engine_smoke/``:

* ``ledger_preflight.jsonl``, ``ledger_medium.jsonl``, ``ledger_high.jsonl``: MEvolve's ledgers,
  one line per Claude call, copied byte for byte;
* ``programs.json``: per effort, Shinka's program table (one row per program: island, parent,
  score, metrics, migrations), the proposals whose edit failed (``attempt_log``) and the run
  config;
* ``console.json``: the Anthropic Console's cost and caching figures, typed in by hand from
  screenshots (the Console cannot be read by a script).

Shinka's database is opened with ``mode=ro&immutable=1``, so SQLite takes no lock and touches no
``-wal``/``-shm`` file (the database is in WAL mode and its ``-wal`` is empty, which the freeze
checks); the freeze also checks that every source file is unchanged afterwards. The ``-shm`` file
is left out of that check: the run's web viewer (SLURM job 27586077) reads the database and
touches it. Needs /allen, so it runs on a compute node; ``gate3_engine_smoke.py`` reads only the
committed files.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

OUT = HERE / "engine_smoke"
GROUP = "engine-smoke@20261103-030341"
RUN = Path("/allen/aind/scratch/han.hou/mevolve/runs/engine-smoke/20261103-030341")
PREFLIGHT = Path("/allen/aind/scratch/han.hou/mevolve/search-dev/preflight/ledger.jsonl")
EFFORTS = ("medium", "high")
LEDGERS = {"preflight": PREFLIGHT, **{e: RUN / e / "ledger.jsonl" for e in EFFORTS}}
N_CALLS = {"preflight": 2, "medium": 10, "high": 10}
METADATA_KEYS = ("patch_name", "patch_type", "api_costs", "mevolve_seed")

# The Anthropic Console figures, as Han read them from his Console on 2026-11-03 and sent them as
# screenshots. Typed in by hand; the screenshots are not committed. `shown` is the text on the
# page, `value` the number it stands for (the page rounds: 51K is 50.5K to 51.5K).
CONSOLE = {
    "transcription": "Typed in by hand from screenshots of the Anthropic Console that Han sent on "
                     "2026-11-03. The Console cannot be read by a script, so these figures cannot be "
                     "frozen any other way; the screenshots themselves are not committed.",
    "workspace": "Han's Credits",
    "range": "last 7 days",
    "cost_page": {
        "data_as_of": "Nov 3, 7:48 PM UTC",
        "data_as_of_utc": "2026-11-03T19:48:00Z",
        "models": ["Claude Sonnet 5.5"],
        "days": ["2026-11-03"],
        "total_cost_usd": {"shown": "$0.29", "value": 0.29},
        "token_cost_usd": {"shown": "$0.29", "value": 0.29},
        "token_list_price_usd": {"shown": "List price before discount $0.59", "value": 0.59},
        "web_search_usd": {"shown": "$0.00", "value": 0.0},
        "code_execution_usd": {"shown": "$0.00", "value": 0.0},
        "session_runtime_usd": {"shown": "$0.00", "value": 0.0},
    },
    "caching_page": {
        "data_as_of": "Nov 3, 11:00 AM UTC, updates hourly",
        "data_as_of_utc": "2026-11-03T11:00:00Z",
        "uncached_input_tokens": {"shown": "51K", "value": 51000},
        "cache_read_tokens": {"shown": "259K", "value": 259000},
        "cache_read_ratio": {"shown": "83.5%", "value": 0.835},
        "by_model": {
            "Claude Sonnet 5.5": {
                "input_tokens": {"shown": "367K", "value": 367000},
                "read_ratio": {"shown": "83.5%", "value": 0.835},
                "write_amortization": {"shown": "4.50×", "value": 4.5},
            },
        },
        "legend": ["cache read", "cache write (5m)", "cache write (1h)", "uncached"],
    },
}
CONSOLE_SOURCES = [
    {"type": "url", "id": "Anthropic Console, Cost page (workspace Han's Credits, last 7 days)",
     "state": "Data as of Nov 3, 7:48 PM UTC; read by hand from a screenshot",
     "updated_at": CONSOLE["cost_page"]["data_as_of_utc"]},
    {"type": "url", "id": "Anthropic Console, Caching page (workspace Han's Credits, last 7 days)",
     "state": "Data as of Nov 3, 11:00 AM UTC, updates hourly; read by hand from a screenshot",
     "updated_at": CONSOLE["caching_page"]["data_as_of_utc"]},
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists()}


def _programs(db: Path) -> tuple[list[dict], list[dict]]:
    wal = db.with_name(db.name + "-wal")
    # immutable=1 ignores the WAL, so it must hold nothing.
    assert not wal.exists() or wal.stat().st_size == 0, f"{wal} is not empty"
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    rows = list(con.execute(
        "SELECT id, generation, island_idx, parent_id, timestamp, combined_score, correct, public_metrics, "
        "private_metrics, metadata, migration_history, text_feedback FROM programs ORDER BY generation, timestamp"))
    attempts = list(con.execute("SELECT generation, stage, status, details, created_at FROM attempt_log ORDER BY id"))
    con.close()
    programs = []
    for r in rows:
        metadata = json.loads(r["metadata"] or "{}")
        row = {"id": r["id"], "generation": r["generation"], "island_idx": r["island_idx"],
               "parent_id": r["parent_id"], "timestamp": r["timestamp"], "combined_score": r["combined_score"],
               "correct": bool(r["correct"]),
               "public_metrics": json.loads(r["public_metrics"]) if r["public_metrics"] else None,
               "private_metrics": json.loads(r["private_metrics"]) if r["private_metrics"] else None,
               "metadata": {k: metadata[k] for k in METADATA_KEYS if k in metadata},
               "migration_history": json.loads(r["migration_history"] or "[]")}
        if not row["correct"]:
            row["text_feedback"] = r["text_feedback"]       # why it was not scored
        programs.append(row)
    failed = []
    for a in attempts:
        details = json.loads(a["details"] or "{}")
        failed.append({"generation": a["generation"], "stage": a["stage"], "status": a["status"],
                       "failure_class": details.get("failure_class"),
                       "failure_reason": (details.get("failure_reason") or "").splitlines()[0],
                       "api_costs": details.get("api_costs"), "created_at": a["created_at"]})
    return programs, failed


def main() -> None:
    watched = [LEDGERS["preflight"]] + [RUN / e / f for e in EFFORTS for f in (
        "ledger.jsonl", "programs.sqlite", "programs.sqlite-wal", "run_config.json", "run_summary.json")]
    before = _stat(watched)

    ledgers, sources = {}, []
    for run, path in LEDGERS.items():
        data = path.read_bytes()
        calls = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        assert len(calls) == N_CALLS[run], f"{path}: {len(calls)} calls, expected {N_CALLS[run]}"
        ledgers[run] = (data, calls)
        sources.append({"type": "file", "id": str(path), "sha256": _sha256(data)})

    runs = {}
    for effort in EFFORTS:
        config_bytes = (RUN / effort / "run_config.json").read_bytes()
        summary_bytes = (RUN / effort / "run_summary.json").read_bytes()
        db = RUN / effort / "programs.sqlite"
        db_sha = _sha256(db.read_bytes())
        config, summary = json.loads(config_bytes), json.loads(summary_bytes)
        programs, failed = _programs(db)
        assert _sha256(db.read_bytes()) == db_sha, f"{db} changed while it was read"
        calls = ledgers[effort][1]
        # The files belong to this launch and this effort, and agree with each other.
        assert f"{config['wandb']['group']}" == GROUP, f"{effort}: group {config['wandb']['group']}"
        assert config["effort"] == effort and all(c["effort"] == effort for c in calls), effort
        assert summary["config"] == config, f"{effort}: run_summary.json has another config"
        assert summary["calls"] == len(calls) == config["n_proposals"], effort
        assert abs(summary["cost_usd"] - sum(c["cost"] for c in calls)) < 1e-9, effort
        sources += [{"type": "file", "id": str(RUN / effort / "run_config.json"), "sha256": _sha256(config_bytes)},
                    {"type": "file", "id": str(RUN / effort / "run_summary.json"), "sha256": _sha256(summary_bytes)},
                    {"type": "file", "id": str(db), "sha256": db_sha}]
        runs[effort] = {"run_config": config, "programs": programs, "failed_attempts": failed}

    assert _stat(watched) == before, "a source file changed during the freeze"

    # Every check ran; now write.
    OUT.mkdir(exist_ok=True)
    ledger_files = {}
    for run, (data, _) in ledgers.items():
        path = OUT / f"ledger_{run}.jsonl"
        path.write_bytes(data)
        assert _sha256(path.read_bytes()) == _sha256(data)
        ledger_files[run] = path
    meta = build_meta("analysis/freeze_engine_smoke.py", [GROUP], study_root=STUDY,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(OUT / "programs.json", {
        "_meta": meta,
        "about": f"Shinka's program table and failed edits for {GROUP}, one entry per effort, read-only "
                 "from programs.sqlite; the ledgers beside this file are byte-for-byte copies.",
        "wandb_group": GROUP,
        "run_dir": str(RUN),
        "ledgers": {run: p.name for run, p in ledger_files.items()},
        "runs": runs,
    }, outputs=list(ledger_files.values()))
    console_meta = build_meta("analysis/freeze_engine_smoke.py", [], study_root=STUDY,
                              kind="freeze", code=[__file__], sources=CONSOLE_SOURCES)
    write_artifact(OUT / "console.json", {"_meta": console_meta, **CONSOLE})
    print(f"froze {sum(len(c) for _, c in ledgers.values())} calls and "
          f"{sum(len(r['programs']) for r in runs.values())} programs to {OUT}")


if __name__ == "__main__":
    main()
