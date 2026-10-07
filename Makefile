.PHONY: setup data results bundle test smoke lint

setup:
	bash setup_env.sh

data:
	cd data && unzip -oq results_bundle.zip

results: data
	uv run python scripts/reproduction/compute_results.py

bundle:
	uv run python scripts/reproduction/build_results_bundle.py

test:
	uv run pytest tests/ -q --ignore=tests/test_smoke.py

smoke:
	uv run pytest tests/test_smoke.py -q

lint:
	uv run ruff check src/brain_code/ scripts/ tests/
