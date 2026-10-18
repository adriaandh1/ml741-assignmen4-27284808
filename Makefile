PYTHON ?= .venv/bin/python

.PHONY: all setup test check smoke report verify assets prepare som clustering assessment bootstrap clean

all: report

setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

test:
	$(PYTHON) -m pytest

check:
	.venv/bin/ruff format --check src tests report/scripts
	.venv/bin/ruff check src tests report/scripts

report:
	$(MAKE) -C report PYTHON=$(abspath $(PYTHON))

verify:
	$(PYTHON) src/cli.py verify
	$(MAKE) -C report verify PYTHON=$(abspath $(PYTHON))

assets:
	$(MAKE) -C report assets PYTHON=$(abspath $(PYTHON))

smoke prepare som clustering assessment bootstrap:
	$(PYTHON) src/cli.py $@

clean:
	$(MAKE) -C report clean
