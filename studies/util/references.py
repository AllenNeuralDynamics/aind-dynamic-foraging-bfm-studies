"""Named versions of the numbers studies compare against, read from the source study's JSON.

Why versions: "the GRU ceiling" is not one number. Study 01 measured it with two training
protocols and two hidden sizes, and reports quote five spellings of it. They are three
measurements (``gru_ceiling.v1_h128``, ``.v2_h128``, ``.v2_h256``); three of the spellings are
the ``v2_h128`` mean rounded to 4, 5 and 6 decimals. A version names one measurement and its
conditions, so a report can say which one it used (ADR 0003, issue #63).

Each entry is a name ``<comparison>.<version>``, a source ``{study, report, json, pointer}``
(``pointer`` is an RFC 6901 JSON Pointer into the committed JSON) and its ``conditions``.
Entries store no values: the value is read from the source JSON when ``ref()`` is called, so a
correction there reaches every reader. Each comparison has one default version
(``DEFAULTS``).

Usage in a producer::

    from references import ref
    gru = ref("gru_ceiling")                     # the default version
    meta = build_meta(..., code=[__file__], inputs=[..., gru.source])
    out["gru_ceiling"] = {"name": gru.name, "value": gru.value}
    print(gru.cite())                            # "<value to 4 dp> (gru_ceiling.v2_h128)"

and the report names ``gru.upstream`` (``{study, report}``) in its frontmatter ``upstream``,
so report-upstream-stale and report-upstream-undeclared keep working. Using a version other
than the default needs one line in the report saying why.

Values come back at full precision. Round when printing, and print the version name next to
the number (``cite()``). ``cross_study.py`` rounds to 4 decimals for the reports already
written against it (studies#20); it is left as it is so their artifacts stay current.

Digests: ``build_meta`` records every imported ``studies/util`` module, so editing this file,
including adding an entry, makes every artifact whose producer imported it stale until it is
regenerated. A new measurement gets a new name; an existing entry is never repointed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
STUDIES = REPO / "studies"

# What every entry's `conditions` states. "protocol" is the training or fitting procedure;
# study 01's v1/v2 are protocols on one data snapshot, not data versions.
CONDITION_KEYS = ("data", "protocol", "model", "D", "mouse_set", "metric", "split")

# One default per comparison. PROPOSED in issue #63; Han decides.
DEFAULTS = {
    # Study 01's production recipe (H128, session conditioning engaged), which studies 07, 08
    # and 10 match; v1 is the record of a run where session conditioning never engaged, and
    # H256 is a capacity probe.
    "gru_ceiling": "v2_h128",
    # The best of the three per-mouse RL fits (rl_baseline.json /best_model) and the bar
    # studies 05, 06, 11 and 13 already use.
    "rl_baseline": "ctt",
    # Only version.
    "gru_generative_match": "v2_d614",
    # Only version.
    "disrnn_tuned": "s06_d614",
}

_S01_GRU = {
    "data": "data=mice_snapshot_scaling",
    "model": "GRU, scalar session encoding",
    "D": "614 training mice (subject_ratio=1.0)",
    "mouse_set": "149 held-out mice (heldout_every_n=5), never trained on; the same at every D and seed",
    "metric": "heldout/final/eval_likelihood: pooled trial-weighted likelihood exp(-sum nll / n) "
              "over all held-out eval trials; mean of 3 seeds",
    "split": "held-out mice: subject embedding fine-tuned on each mouse's train sessions, "
             "scored on its eval sessions",
}
_S01_RL = {
    "data": "data=mice_snapshot_scaling",
    "D": "none: one fit per mouse, no population training",
    "mouse_set": "149 held-out mice, 1,013,172 eval trials; 146 of them shared with the GRU held-out set",
    "metric": "pooled_likelihood_trial_weighted: exp(mean log-likelihood) over all held-out eval "
              "trials (not the per-subject mean or median in the same JSON)",
    "split": "each mouse fitted by MLE on its train sessions, scored on its eval sessions",
}
_V2_SC_ACTIVE = ("v2-sc-active: lambda-forward (session conditioning fully on at 50k steps), "
                 "gated early stop at 70k, n_steps=150000")

_ENTRIES: dict[str, dict] = {
    "gru_ceiling.v1_h128": {
        "study": "01-gru-scaling-law", "report": "r1", "json": "analysis/report_data.json",
        "pointer": "/cell_level/per_ratio/1.0/v1",
        "checks": {"/cell_level/per_ratio/1.0/D_mean": 614},
        "conditions": {**_S01_GRU,
                       "protocol": "v1-pretrain-phase: early stop fired at ~40k steps, inside "
                                   "pretrain, so session conditioning never engaged "
                                   "(W&B group mice-data-scaling-gru)",
                       "model": "GRU, hidden_size=128, scalar session encoding"},
    },
    "gru_ceiling.v2_h128": {
        "study": "01-gru-scaling-law", "report": "r1", "json": "analysis/report_data.json",
        "pointer": "/cell_level/per_ratio/1.0/v2",
        "checks": {"/cell_level/per_ratio/1.0/D_mean": 614},
        "conditions": {**_S01_GRU,
                       "protocol": _V2_SC_ACTIVE + " (W&B group v2-sc-active@20260622-144622; "
                                   "the same runs fill the H128 column of r7's N x D grid)",
                       "model": "GRU, hidden_size=128, scalar session encoding"},
    },
    "gru_ceiling.v2_h256": {
        "study": "01-gru-scaling-law", "report": "r7", "json": "analysis/nxd_scaling.json",
        "pointer": "/mean_grid/3/3",                    # positional: Ns[3] x Ds[3], checked below
        "checks": {"/Ns/3": 256, "/Ds/3": 614},
        "conditions": {**_S01_GRU,
                       "protocol": "nxd-grid: the " + _V2_SC_ACTIVE
                                   + " recipe (W&B group nxd-grid@20260623-102649)",
                       "model": "GRU, hidden_size=256, scalar session encoding"},
    },
    "rl_baseline.ctt": {
        "study": "01-gru-scaling-law", "report": "r8", "json": "analysis/rl_baseline.json",
        "pointer": "/models/ctt/pooled_likelihood_trial_weighted",
        "conditions": {**_S01_RL,
                       "protocol": "per-mouse MLE (W&B group rl-baseline-ctt@20260713-010305)",
                       "model": "compare-to-threshold (ForagerCompareThreshold): learn_rate, "
                                "threshold, softmax_inverse_temperature, biasL"},
    },
    "rl_baseline.bari": {
        "study": "01-gru-scaling-law", "report": "r8", "json": "analysis/rl_baseline.json",
        "pointer": "/models/bari/pooled_likelihood_trial_weighted",
        "conditions": {**_S01_RL,
                       "protocol": "per-mouse MLE (W&B group rl-baseline-bari@20260713-010244)",
                       "model": "Bari L1F1_CK1 (ForagerQLearning): one learning rate, one forget "
                                "rate, one-step choice kernel, softmax"},
    },
    "rl_baseline.hattori": {
        "study": "01-gru-scaling-law", "report": "r8", "json": "analysis/rl_baseline.json",
        "pointer": "/models/hattori/pooled_likelihood_trial_weighted",
        "conditions": {**_S01_RL,
                       "protocol": "per-mouse MLE (W&B group rl-baseline-hattori@20260713-010255)",
                       "model": "Hattori L2F1 (ForagerQLearning): asymmetric learning rates, one "
                                "forget rate, no choice kernel, softmax"},
    },
    "gru_generative_match.v2_d614": {
        "study": "01-gru-scaling-law", "report": "r9", "json": "analysis/generative_match.json",
        "pointer": "/v2_D614/corr_mean",                # r9's headline; RMSE is /v2_D614/rmse_mean
        "checks": {"/v2_D614/D": 614},
        "conditions": {**_S01_GRU,
                       "protocol": _V2_SC_ACTIVE + ", rolled out as a generative agent on the "
                                   "curriculum task (W&B group generative-v2@20260623-180750)",
                       "model": "GRU, hidden_size=128, scalar session encoding",
                       "mouse_set": "the run's subjects, each compared with the model's rollouts "
                                    "as that subject (not the held-out likelihood cohort)",
                       "metric": "switch-triggered curve P(switch | reward, run length): "
                                 "subject-mean Pearson correlation with the mice; mean of 3 seeds",
                       "split": "'combined' session partition: train and eval sessions together"},
    },
    "disrnn_tuned.s06_d614": {
        "study": "06-disrnn-operating-point-at-scale", "report": "r3",
        "json": "analysis/r3_summary.json",
        "pointer": "/cells/8/heldout_mean",             # positional: checked below
        "checks": {"/cells/8/D": 614, "/cells/8/mult": 1, "/cells/8/beta": 3e-4},
        "conditions": {**_S01_GRU,
                       "protocol": "study 06's tuned operating point, n_steps=100000, no early "
                                   "stopping (W&B group mult-d-grid@20260718-151409)",
                       "model": "disRNN, latent_size=5, update-net penalty multiplier 1, beta=3e-4",
                       "mouse_set": "the held-out mice of data=mice_snapshot_scaling, as in study 01",
                       "metric": "heldout/final/eval_likelihood: pooled trial-weighted likelihood "
                                 "over all held-out eval trials; mean of 2 seeds"},
    },
}


class ReferenceUnresolved(LookupError):
    """A registry entry does not resolve against its committed source JSON."""


@dataclass(frozen=True)
class Reference:
    name: str                  # "<comparison>.<version>"
    value: float               # read from the source JSON, full precision
    study: str
    report: str
    json: str                  # study-relative path of the source JSON
    pointer: str               # RFC 6901 pointer into it
    conditions: dict = field(default_factory=dict)
    is_default: bool = False

    @property
    def source(self) -> Path:
        """The source JSON, for ``build_meta(..., inputs=[ref.source])``."""
        return STUDIES / self.study / self.json

    @property
    def upstream(self) -> dict:
        """The frontmatter ``upstream`` entry for the report that quotes this number."""
        return {"study": self.study, "report": self.report}

    def cite(self, fmt: str = ".4f") -> str:
        """The number with its version name, as a report prints it: ``<value> (gru_ceiling.v2_h128)``."""
        return f"{format(self.value, fmt)} ({self.name})"


def resolve_pointer(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer; raise ``ReferenceUnresolved`` if any token is missing."""
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ReferenceUnresolved(f"JSON Pointer {pointer!r} must be empty or start with '/'")
    node = doc
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                raise ReferenceUnresolved(f"{pointer!r}: no key {token!r}")
            node = node[token]
        elif isinstance(node, list):
            if not (token == "0" or (token.isdigit() and not token.startswith("0"))):
                raise ReferenceUnresolved(f"{pointer!r}: {token!r} is not an array index")
            if int(token) >= len(node):
                raise ReferenceUnresolved(f"{pointer!r}: index {token} out of range ({len(node)})")
            node = node[int(token)]
        else:
            raise ReferenceUnresolved(f"{pointer!r}: cannot descend into {type(node).__name__} at {token!r}")
    return node


def comparisons() -> list[str]:
    return sorted({name.split(".", 1)[0] for name in _ENTRIES})


def versions(comparison: str | None = None) -> list[str]:
    """Registered names, all of them or one comparison's, e.g. ``versions("gru_ceiling")``."""
    return sorted(n for n in _ENTRIES if comparison is None or n.split(".", 1)[0] == comparison)


def ref(name: str) -> Reference:
    """Read a reference: ``ref("gru_ceiling")`` (the default) or ``ref("gru_ceiling.v2_h128")``.

    Raises ``KeyError`` for an unknown name and ``ReferenceUnresolved`` when the pointer, or one
    of the entry's coordinate checks, does not resolve against the committed JSON, or what it
    points at is not a number (null, a dict, a string).
    """
    if "." not in name:
        if name not in DEFAULTS:
            raise KeyError(f"no default for comparison {name!r}; comparisons: {comparisons()}")
        name = f"{name}.{DEFAULTS[name]}"
    if name not in _ENTRIES:
        raise KeyError(f"no reference {name!r}; registered: {versions()}")
    entry = _ENTRIES[name]
    path = STUDIES / entry["study"] / entry["json"]
    where = f"{name}: {path.relative_to(REPO)}"
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ReferenceUnresolved(f"{where}: {type(e).__name__}: {e}") from e
    try:
        for pointer, want in entry.get("checks", {}).items():
            got = resolve_pointer(doc, pointer)
            if got != want:
                raise ReferenceUnresolved(f"check {pointer} is {got!r}, expected {want!r}; "
                                          "the pointer no longer names this measurement")
        value = resolve_pointer(doc, entry["pointer"])
    except ReferenceUnresolved as e:
        raise ReferenceUnresolved(f"{where}: {e}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceUnresolved(f"{where}: {entry['pointer']} is {value!r}, not a number")
    comparison, version = name.split(".", 1)
    return Reference(name=name, value=value, study=entry["study"], report=entry["report"],
                     json=entry["json"], pointer=entry["pointer"],
                     conditions=dict(entry["conditions"]),
                     is_default=DEFAULTS.get(comparison) == version)
