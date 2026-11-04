#!/usr/bin/env python
"""Gate 3's numbers for the engine-smoke run (``make gate3``), from committed frozen inputs only.

Reads the files ``freeze_engine_smoke.py`` wrote under ``analysis/engine_smoke/`` (the three
ledgers, Shinka's program table and the Console transcription) and writes
``analysis/gate3_engine_smoke.json``: the calls and how they ended, the ledger against the
Console, caching, the islands check, medium against high effort, and the evaluations. The
Result section of ``variants/engine-smoke/notes.md`` cites this file. Standard library only.
"""
from __future__ import annotations

import datetime
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FROZEN = HERE / "engine_smoke"
RUNS = ("preflight", "medium", "high")
EFFORTS = ("medium", "high")
LEDGERS = {run: FROZEN / f"ledger_{run}.jsonl" for run in RUNS}
PROGRAMS = FROZEN / "programs.json"
CONSOLE = FROZEN / "console.json"
OUT = HERE / "gate3_engine_smoke.json"
GROUP = "engine-smoke@20261103-030341"

# Claude Sonnet 5.5 list prices, USD per million tokens, as in MEvolve's price table
# (mevolve.search.claude.PRICES); checked below against the cost of every ledger line.
PRICE = {"input": 2.0, "output": 10.0, "cache_write_5m": 2.5, "cache_read": 0.2}
IMPROVED = 0.05            # the PLAN's "improved on the parent" margin, in validation-score points
CACHE_LIFETIME_S = 300     # Anthropic's default (5-minute) prompt cache, refreshed by each read


def _utc(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")


def _shown(value: float, unit: float) -> float:
    """``value`` rounded the way the Console shows it (51,092 tokens -> 51K)."""
    return round(value / unit) * unit


def _price(c: dict) -> float:
    return (c["input_tokens"] * PRICE["input"] + c["output_tokens"] * PRICE["output"]
            + c["cache_creation_input_tokens"] * PRICE["cache_write_5m"]
            + c["cache_read_input_tokens"] * PRICE["cache_read"]) / 1e6


def _gaps(calls: list[dict]) -> dict:
    """Longest gap between consecutive calls, start to start and end to start, in seconds."""
    calls = sorted(calls, key=lambda c: c["time"])
    pairs = list(zip(calls, calls[1:]))
    return {"start_to_start": round(max(b["time"] - a["time"] for a, b in pairs), 1),
            "end_to_start": round(max(b["time"] - (a["time"] + a["seconds"]) for a, b in pairs), 1)}


def _between(first: list[dict], second: list[dict]) -> dict:
    """Gap from the last call of ``first`` to the first call of ``second``."""
    a = max(first, key=lambda c: c["time"])
    b = min(second, key=lambda c: c["time"])
    return {"start_to_start": round(b["time"] - a["time"], 1),
            "end_to_start": round(b["time"] - (a["time"] + a["seconds"]), 1)}


def calls_and_cost(ledgers: dict, console: dict) -> dict:
    every = [c for run in RUNS for c in ledgers[run]]
    total = {k: sum(c.get(k, 0) for c in every) for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    all_input = (total["input_tokens"] + total["cache_creation_input_tokens"]
                 + total["cache_read_input_tokens"])
    cost = sum(c["cost"] for c in every)
    read_ratio = total["cache_read_input_tokens"] / (total["cache_read_input_tokens"] + total["input_tokens"])
    reads_per_write = total["cache_read_input_tokens"] / total["cache_creation_input_tokens"]

    cp, kp = console["cost_page"], console["caching_page"]
    sonnet = kp["by_model"]["Claude Sonnet 5.5"]
    con = {"uncached_input_tokens": kp["uncached_input_tokens"]["value"],
           "cache_read_tokens": kp["cache_read_tokens"]["value"],
           "all_input_tokens": sonnet["input_tokens"]["value"],
           "cache_write_tokens_implied": round(kp["cache_read_tokens"]["value"]
                                               / sonnet["write_amortization"]["value"]),
           "cache_read_ratio": kp["cache_read_ratio"]["value"],
           "write_amortization": sonnet["write_amortization"]["value"],
           "token_list_price_usd": cp["token_list_price_usd"]["value"],
           "token_cost_billed_usd": cp["token_cost_usd"]["value"]}
    led = {"uncached_input_tokens": total["input_tokens"],
           "cache_write_tokens": total["cache_creation_input_tokens"],
           "cache_read_tokens": total["cache_read_input_tokens"],
           "all_input_tokens": all_input, "output_tokens": total["output_tokens"],
           "cost_list_usd": round(cost, 6), "cache_read_ratio": round(read_ratio, 4),
           "cache_reads_per_write": round(reads_per_write, 4)}
    # Each ledger count, rounded the way the Console shows it, against the Console.
    match = {"uncached_input_tokens": _shown(led["uncached_input_tokens"], 1000) == con["uncached_input_tokens"],
             "cache_read_tokens": _shown(led["cache_read_tokens"], 1000) == con["cache_read_tokens"],
             "all_input_tokens": _shown(led["all_input_tokens"], 1000) == con["all_input_tokens"],
             "cache_read_ratio": round(read_ratio, 3) == con["cache_read_ratio"],
             "write_amortization": round(reads_per_write, 2) == con["write_amortization"]}
    above = 100 * (cost / con["token_list_price_usd"] - 1)
    price_diff = max(abs(_price(c) - c["cost"]) for c in every)
    run_calls = [c for run in EFFORTS for c in ledgers[run]]
    return {
        "calls": {
            "n": len(every), "by_run": {run: len(ledgers[run]) for run in RUNS},
            "errors": sum("error" in c for c in every),
            "refusals": sum(c.get("stop_reason") == "refusal" for c in every),
            "cut_off": sum(c.get("stop_reason") == "max_tokens" for c in every),
            "stop_reasons": {r: sum(c.get("stop_reason") == r for c in every)
                             for r in sorted({str(c.get("stop_reason")) for c in every})},
            "models": sorted({c["model"] for c in every}),
            "served_models": sorted({c["served_model"] for c in every}),
            "first_call_utc": _utc(min(c["time"] for c in every)),
            "last_call_utc": _utc(max(c["time"] for c in every)),
        },
        "ledger_vs_console": {
            "ledger": led, "console": con, "console_rounding_matches": match,
            "ledger_cost_above_console_list_pct": round(above, 2),
            "within_5pct": abs(above) <= 5,
            "billed_over_list_pct": round(100 * con["token_cost_billed_usd"] / con["token_list_price_usd"], 1),
            "ledger_over_billed": round(cost / con["token_cost_billed_usd"], 2),
        },
        "price_table": {
            "usd_per_million_tokens": PRICE,
            "reproduces_every_ledger_cost": price_diff < 1e-9,
            "max_abs_diff_usd": round(price_diff, 12),
        },
        "cost_split": {
            "output_share_of_cost_pct": round(100 * total["output_tokens"] * PRICE["output"] / 1e6 / cost, 1),
            "uncached_share_of_input_pct": round(100 * total["input_tokens"] / all_input, 1),
            "mean_uncached_input_per_run_call": round(statistics.mean(c["input_tokens"] for c in run_calls)),
        },
    }


def caching(ledgers: dict) -> dict:
    every = [c for run in RUNS for c in ledgers[run]]
    writers = {run: sorted((c for c in ledgers[run] if c["cache_creation_input_tokens"]), key=lambda c: c["time"])
               for run in RUNS}
    return {
        "prefix_tokens": sorted({c["cache_creation_input_tokens"] for c in every if c["cache_creation_input_tokens"]}
                                | {c["cache_read_input_tokens"] for c in every if c["cache_read_input_tokens"]}),
        "writes": {run: len(writers[run]) for run in RUNS},
        "reads": {run: sum(c["cache_read_input_tokens"] > 0 for c in ledgers[run]) for run in RUNS},
        # the medium run's writers: which calls (1 = first, by start time), and how close their starts
        "medium_writers": {
            "call_numbers": [sorted(ledgers["medium"], key=lambda c: c["time"]).index(c) + 1
                             for c in writers["medium"]],
            "start_spread_s": round(writers["medium"][-1]["time"] - writers["medium"][0]["time"], 2)},
        "longest_gap_within_run_s": {run: _gaps(ledgers[run]) for run in RUNS},
        "gap_preflight_to_medium_s": _between(ledgers["preflight"], ledgers["medium"]),
        "gap_medium_to_high_s": _between(ledgers["medium"], ledgers["high"]),
        "cache_lifetime_s": CACHE_LIFETIME_S,
    }


def _island_at(prog: dict, t: float) -> int:
    """The island ``prog`` was on at time ``t``: undo its migrations that happened after ``t``."""
    island = prog["island_idx"]
    for m in sorted(prog["migration_history"], key=lambda m: m["timestamp"], reverse=True):
        if m["timestamp"] > t:
            assert m["to"] == island, f"{prog['id']}: migration history does not chain"
            island = m["from"]
    return island


def _label(prog: dict) -> str:
    if prog["metadata"].get("mevolve_seed"):
        return prog["metadata"]["mevolve_seed"]
    if prog["metadata"].get("patch_type") == "init":
        return "template"
    return f"generation {prog['generation']}"


def islands(runs: dict) -> dict:
    out, other = {}, []
    for effort in EFFORTS:
        progs = runs[effort]["programs"]
        by_id = {p["id"]: p for p in progs}
        start = [p for p in progs if p["generation"] == 0]
        children = [p for p in progs if p["generation"] > 0]
        for c in children:
            parent = by_id[c["parent_id"]]
            if c["island_idx"] == parent["island_idx"]:
                continue
            born_on, parent_then = _island_at(c, c["timestamp"]), _island_at(parent, c["timestamp"])
            if born_on != parent_then:
                why = "unexplained"
            elif born_on != c["island_idx"]:
                why = "child migrated"
            else:
                why = "born before its parent migrated"
            other.append({"run": effort, "generation": c["generation"], "island": c["island_idx"],
                          "parent": _label(parent), "parent_island": parent["island_idx"],
                          "born_on_island": born_on, "why": why,
                          "migrations": [{"generation": m["generation"], "from": m["from"], "to": m["to"]}
                                         for m in (c["migration_history"] if why == "child migrated"
                                                   else parent["migration_history"])]})
        out[effort] = {
            "start": {str(p["island_idx"]): _label(p) for p in sorted(start, key=lambda p: p["island_idx"])},
            "one_program_per_island_at_start": len({p["island_idx"] for p in start}) == len(start),
            "children": len(children),
            "children_on_another_island": sum(o["run"] == effort for o in other),
        }
    whys = [o["why"] for o in other]
    out["children_on_another_island"] = len(other)
    out["by_reason"] = {w: whys.count(w) for w in ("child migrated", "born before its parent migrated", "unexplained")}
    out["migration_generations"] = sorted({m["generation"] for o in other if o["why"] == "child migrated"
                                           for m in o["migrations"]})
    out["detail"] = other
    return out


def effort(runs: dict, ledgers: dict) -> dict:
    out, improved = {}, []
    for e in EFFORTS:
        run, calls = runs[e], ledgers[e]
        progs = run["programs"]
        by_id = {p["id"]: p for p in progs}
        children = [p for p in progs if p["generation"] > 0]
        valid = [p for p in children if p["correct"]]
        seeds = [p for p in progs if p["generation"] == 0 and p["metadata"].get("mevolve_seed")]
        best_seed = max(seeds, key=lambda p: p["combined_score"])
        best = max((p for p in progs if p["correct"]), key=lambda p: p["combined_score"])
        n = run["run_config"]["n_proposals"]
        missing = sorted(set(range(1, n + 1)) - {p["generation"] for p in children})
        for p in valid:
            parent = by_id[p["parent_id"]]
            delta = p["combined_score"] - parent["combined_score"]
            if delta > IMPROVED:
                improved.append({"run": e, "generation": p["generation"], "parent": _label(parent),
                                 "parent_score": round(parent["combined_score"], 4),
                                 "score": round(p["combined_score"], 4), "gain": round(delta, 4)})
        reading = [c for c in calls if c["cache_creation_input_tokens"] == 0]
        out[e] = {
            "proposals": len(calls),
            "edits_not_applied": len(missing),
            "edits_not_applied_generations": missing,
            "edits_not_applied_reasons": sorted({f"{a['failure_class']}: {a['failure_reason']}"
                                                 for a in run["failed_attempts"]}),
            "rejected": len(children) - len(valid),
            "rejected_reasons": [p.get("text_feedback") for p in children if not p["correct"]],
            "valid": len(valid),
            "improved_on_parent_by_more_than_0.05": sum(i["run"] == e for i in improved),
            "best_validation_score": round(best["combined_score"], 4),
            "best_program": _label(best) if best["generation"] == 0 else f"generation {best['generation']}",
            "best_seed": {"name": best_seed["metadata"]["mevolve_seed"],
                          "validation_score": round(best_seed["combined_score"], 4)},
            "best_minus_best_seed": round(best["combined_score"] - best_seed["combined_score"], 4),
            "output_tokens_per_call": round(statistics.mean(c["output_tokens"] for c in calls), 1),
            "calls_reading_the_cache": len(reading),
            "cost_per_proposal_reading_the_cache_usd": round(statistics.mean(c["cost"] for c in reading), 6),
            "cache_writes": len(calls) - len(reading),
            "run_cost_list_usd": round(sum(c["cost"] for c in calls), 6),
        }
    m, h = out["medium"], out["high"]
    out["high_over_medium"] = {
        "cost_per_proposal_reading_the_cache": round(h["cost_per_proposal_reading_the_cache_usd"]
                                                     / m["cost_per_proposal_reading_the_cache_usd"], 3),
        "output_tokens_per_call": round(h["output_tokens_per_call"] / m["output_tokens_per_call"], 3),
    }
    out["improved"] = improved
    return out


def evaluation(runs: dict) -> dict:
    out = {}
    for e in EFFORTS:
        progs = runs[e]["programs"]
        scored = [p for p in progs if p["generation"] > 0 and p["correct"]]
        seconds = [p["private_metrics"]["seconds"] for p in scored]
        out[e] = {
            "start_programs_seconds": {_label(p): round(p["private_metrics"]["seconds"], 1)
                                       for p in progs if p["generation"] == 0},
            "scored_children": len(scored),
            "two_level_failed": sum(p["public_metrics"]["two_level_failed"] for p in scored),
            "two_level_failed_generations": [p["generation"] for p in scored if p["public_metrics"]["two_level_failed"]],
            "seconds_per_program": {"min": round(min(seconds), 1), "median": round(statistics.median(seconds), 1),
                                    "max": round(max(seconds), 1)},
            "subject_seconds_max": round(max(p["private_metrics"]["subject_seconds_max"] for p in scored), 1),
        }
    return out


def main() -> None:
    ledgers = {run: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               for run, path in LEDGERS.items()}
    frozen = json.loads(PROGRAMS.read_text())
    console = json.loads(CONSOLE.read_text())
    assert frozen["wandb_group"] == GROUP
    result = {
        "question": "Gate 3 (PLAN.md): does the engine run end to end with real Claude calls, and what "
                    "does it cost? Numbers for the Result section of variants/engine-smoke/notes.md.",
        **calls_and_cost(ledgers, console),
        "caching": caching(ledgers),
        "islands": islands(frozen["runs"]),
        "effort": effort(frozen["runs"], ledgers),
        "evaluation": evaluation(frozen["runs"]),
    }
    meta = build_meta("analysis/gate3_engine_smoke.py", [GROUP], study_root=STUDY,
                      code=[__file__], inputs=[*LEDGERS.values(), PROGRAMS, CONSOLE])
    write_artifact(OUT, {"_meta": meta, **result})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
