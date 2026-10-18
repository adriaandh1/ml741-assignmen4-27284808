REPORT := 27284808RW741assignment4
PYTHON ?= ../.venv/bin/python

.PHONY: all assets verify clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(REPORT) main.tex

assets:
	$(PYTHON) scripts/build_assets.py

verify:
	$(PYTHON) scripts/verify_report.py

clean:
	latexmk -c -jobname=$(REPORT) main.tex
