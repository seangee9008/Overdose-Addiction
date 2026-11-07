# Pipeline shortcuts. Each target is a named list of shell commands; `make all` runs everything.
#   make data      raw -> validated Parquet (data/interim) -> dbt build (DuckDB marts + tests)
#   make results   run every analysis script against the marts, then build results/RESULTS.md
#   make test      pytest (golden comparison + unit + data-layer tests)
#   make robustness   Phase 4a robustness checks (src/step_r.py) -> outputs/robustness/, then RESULTS.md
#   make teds      Phase 4b TEDS-A area-level analyses (src/step_t.py) -> outputs/teds/, then RESULTS.md
#   make all       data, results, robustness, teds, test
#   make ci        what GitHub Actions runs: lint + build on SYNTHETIC fixtures + pytest
# Override the interpreter with e.g. `make all PY=python DBT=dbt`.

PY  ?= .venv/bin/python
DBT ?= .venv/bin/dbt
DBT_ARGS = --project-dir dbt --profiles-dir dbt

.PHONY: all data ingest dbt docs results robustness teds test lint ci fixtures fixtures-data golden golden-robustness golden-teds clean

all: data results robustness teds test

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

robustness:
	$(PY) -m src.step_r
	$(PY) -m src.build_results

teds:
	$(PY) -m src.step_t
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
	SYNTHETIC_FIXTURES=1 ROBUST_SIMS=40 $(PY) -m pytest -q

# Re-capture the golden snapshot -- only after a deliberate, reviewed change to results.
golden:
	$(PY) -m tests.golden.golden write

golden-robustness:
	$(PY) -m tests.golden.golden write --robustness

golden-teds:
	$(PY) -m tests.golden.golden write --teds

clean:
	rm -rf data/interim data/warehouse build dbt/target dbt/logs
