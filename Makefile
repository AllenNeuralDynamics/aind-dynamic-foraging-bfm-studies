# Repository-level targets. Each study has its own Makefile (`make -C studies/<study>`).
PY ?= python

.PHONY: validate
# Check every study against schemas/ (layout, frontmatter, _meta, launch records). Run before
# opening a PR; CI runs the same check. Fix hints are printed with each new finding.
validate:
	$(PY) scripts/check_provenance.py

.PHONY: new-study new-variant new-report
# Start every study, variant and report here; the result passes `make validate`.
#   make new-study   NAME=12-gru-foo QUESTION="Does ...?" PROJECT=gru_foo VARIANT=base MOTIVATED_BY=06/r2 [ISSUE=...]
#                    (MOTIVATED_BY: what this study follows up, comma-separated 05/r1, 05 or #23; `none` for a root study)
#   make new-variant STUDY=12-gru-foo NAME=wider
#   make new-report  STUDY=12-gru-foo SLUG=heldout-curve TITLE="Held-out vs D" [AUTHOR=han]
# arg = the variable's raw value (no make expansion of `$`), single-quoted for the shell, so
# quotes, `$` and other shell characters in QUESTION or TITLE arrive intact.
arg = '$(subst ','\'',$(value $(1)))'
new-study:
	$(PY) scripts/scaffold.py study --name $(call arg,NAME) --question $(call arg,QUESTION) \
	  --project $(call arg,PROJECT) --variant $(call arg,VARIANT) --issue $(call arg,ISSUE) \
	  --motivated-by $(call arg,MOTIVATED_BY)
new-variant:
	$(PY) scripts/scaffold.py variant --study $(call arg,STUDY) --name $(call arg,NAME)
new-report:
	$(PY) scripts/scaffold.py report --study $(call arg,STUDY) --slug $(call arg,SLUG) \
	  --title $(call arg,TITLE) --author $(if $(AUTHOR),$(call arg,AUTHOR),han)
