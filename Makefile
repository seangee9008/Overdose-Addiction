# Pipeline shortcuts. Each target is a named list of shell commands; `make all` runs everything.
#   make data      raw -> validated Parquet (data/interim) -> dbt build (DuckDB marts + tests)
#   make results   run every analysis script against the marts, then build results/RESULTS.md
#   make test      pytest (golden comparison + unit + data-layer tests)
#   make all       data, results, test
#   make ci        what GitHub Actions runs: lint + build on SYNTHETIC fixtures + pytest
# Override the interpreter with e.g. `make all PY=python DBT=dbt`.

PY  ?= .venv/bin/python
DBT ?= .venv/bin/dbt
DBT_ARGS = --project-dir dbt --profiles-dir dbt

.PHONY: all data ingest dbt docs results test lint ci fixtures fixtures-data golden clean

all: data results test

data: ingest dbt

ingest:
	$(PY) -m src.ingest

dbt:
	mkdir -p data/warehouse
	$(DBT) build $(DBT_ARGS)

docs:
	$(DBT) docs generate $(DBT_ARGS)
	@echo "view the lineage graph with: $(DBT) docs serve $(DBT_ARGS)"

results:
	$(PY) analyze_real.py
	$(PY) analyze_real.py --indicator "Opioids (T40.0-T40.4,T40.6)" --tag opioids
	$(PY) -m src.nsduh_estimates
	$(PY) -m src.step_a
	$(PY) -m src.step_b
	$(PY) -m src.step_c
	$(PY) -m src.step_d
	$(PY) -m src.build_results

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

# --- CI on synthetic fixtures (never real data) --------------------------------------------
fixtures:
	$(PY) tests/fixtures/make_fixtures.py

fixtures-data:
	$(PY) -m src.ingest --raw-dir tests/fixtures/raw --out-dir build/fixtures/interim
	mkdir -p build/fixtures
	INTERIM_DIR=build/fixtures/interim DUCKDB_PATH=build/fixtures/warehouse.duckdb \
		$(DBT) build $(DBT_ARGS)

ci: lint fixtures-data
	SYNTHETIC_FIXTURES=1 $(PY) -m pytest -q

# Re-capture the golden snapshot -- only after a deliberate, reviewed change to results.
golden:
	$(PY) -m tests.golden.golden write

clean:
	rm -rf data/interim data/warehouse build dbt/target dbt/logs
