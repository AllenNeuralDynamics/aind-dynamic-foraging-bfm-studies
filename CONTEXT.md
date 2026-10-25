# aind-dynamic-foraging-bfm-studies

The record of every scientific question asked of dynamic-foraging-bfm: what was launched, what
it produced, and what was concluded. This glossary fixes the provenance vocabulary; model and
evaluation terms (held-out subject, few-shot, condition, ...) are defined in the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md).

## Language

### Studies and their structure

**Study**:
One scientific question, with everything launched, produced and concluded to answer it. Its
accession number is permanent and never reused.
_Avoid_: experiment, project

**Variant**:
A self-contained configuration of a study's experiment, with its own sweep and experiment
manifests. A study compares its variants.
_Avoid_: condition (the dispatcher glossary uses that for Bayesian inference), arm, setting

**Study status**:
Where a study stands: `planned`, `active`, `complete`, or `superseded` (another study now
answers its question).

### Launches

**Launch**:
One initial submission of a variant's work to a compute platform. Every launch mints exactly
one W&B group.
_Avoid_: run (a run is one W&B run inside a launch), job, sweep

**W&B group**:
The identity of a launch's runs in W&B, named `<variant>@<launch_id>`. One launch, one group.
_Avoid_: sweep (a W&B sweep is a scheduling handle, not the group)

**Intervention**:
A later action on an existing launch that keeps its W&B group: a resubmit, rescue, probe,
tier change, backfill, or extend. It reuses the launch's manifests, differing only in declared
deviations; any other change to scientific parameters is a new launch.
_Avoid_: retry, relaunch

**Launch record**:
The immutable file recording one launch or intervention: what was submitted, from which
commits, and why.

### Results

**Report**:
A written answer to one sub-question of a study, regenerated from committed inputs. Its
numbers live in result blocks.
_Avoid_: writeup, notebook, analysis

**Report status**:
`draft`, `live`, `superseded` (another report replaces it entirely), or `retracted` (its
conclusion is wrong and nothing replaces it).

**Correction**:
A later report's amendment to part of an earlier report's conclusion, declared by the
correcting report. The corrected report stays `live` and is never edited to record it.
_Avoid_: update, erratum

**Result block**:
The delimited region of a report holding the numbers its producer regenerates.

**Upstream**:
Another study's report or launch whose results a report uses as input or comparison. Study
dependencies are the union of their reports' upstreams; a mere link is a citation.
_Avoid_: depends_on, prerequisite

**Producer**:
The committed script that computes an analysis artifact from frozen inputs.
_Avoid_: notebook, analysis script

**Analysis artifact**:
A committed file a producer writes (JSON, CSV, or figure). The producer's JSON artifact
records the provenance of every file it wrote.
_Avoid_: output, result

**Freeze**:
A producer that reads a mutable external source (W&B, a Beaker dataset, a URL) and writes a
committed frozen input, recording which runs or objects, at which step or version, it read.
Every other producer reads only committed files.
_Avoid_: pull, snapshot, export

**Frozen input**:
A committed file written by a freeze, from which reports are regenerated offline.
_Avoid_: cache (a cache is uncommitted and never an input)

**Input digest**:
The SHA-256 of a file a producer read, recorded in the artifact so a number traces to exact
bytes rather than to a commit.

**Provenance baseline**:
The committed list of known provenance findings in existing files. It may only shrink.
