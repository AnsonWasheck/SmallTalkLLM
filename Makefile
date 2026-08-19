.PHONY: help install test smoke params lint audit clean

PYTHON ?= python3

help:
	@printf '%s\n' \
	  'make install  Install the package and development dependencies' \
	  'make test     Run the complete test suite' \
	  'make smoke    Run the CPU-safe pre-training smoke gate' \
	  'make params   Verify every documented parameter count' \
	  'make audit    Check git-visible repository hygiene'

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_test.py

params:
	$(PYTHON) scripts/count_params.py --breakdown

audit:
	$(PYTHON) scripts/repo_audit.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
