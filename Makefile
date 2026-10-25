# Repository-level targets. Each study has its own Makefile (`make -C studies/<study>`).
PY ?= python

.PHONY: validate
# Check every study against schemas/ (layout, frontmatter, _meta, launch records). Run before
# opening a PR; CI runs the same check. Fix hints are printed with each new finding.
validate:
	$(PY) scripts/check_provenance.py

.PHONY: new-study new-variant new-report
# Start every study, variant and report here; the result passes `make validate`.
#   make new-study   NAME=12-gru-foo QUESTION="Does ...?" PROJECT=gru_foo VARIANT=base [ISSUE=...]
#   make new-variant STUDY=12-gru-foo NAME=wider
#   make new-report  STUDY=12-gru-foo SLUG=heldout-curve TITLE="Held-out vs D" [AUTHOR=han]
new-study:
	$(PY) scripts/scaffold.py study --name "$(NAME)" --question "$(QUESTION)" --project "$(PROJECT)" \
	  --variant "$(VARIANT)" $(if $(ISSUE),--issue "$(ISSUE)")
new-variant:
	$(PY) scripts/scaffold.py variant --study "$(STUDY)" --name "$(NAME)"
new-report:
	$(PY) scripts/scaffold.py report --study "$(STUDY)" --slug "$(SLUG)" --title "$(TITLE)" \
	  $(if $(AUTHOR),--author "$(AUTHOR)")
