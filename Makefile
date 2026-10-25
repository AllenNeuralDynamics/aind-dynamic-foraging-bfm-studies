# Repository-level targets. Each study has its own Makefile (`make -C studies/<study>`).
PY ?= python

.PHONY: validate
# Check every study against schemas/ (layout, frontmatter, _meta, launch records). Run before
# opening a PR; CI runs the same check. Fix hints are printed with each new finding.
validate:
	$(PY) scripts/check_provenance.py
