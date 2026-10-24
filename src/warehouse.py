"""
Read-only access to the DuckDB warehouse built by dbt (`make data`).

DuckDB is a database that runs inside this Python process: the whole database is the single file
data/warehouse/overdose.duckdb, queried with SQL. The analysis scripts read the dbt marts through
query(); they never read data/raw/ directly.
"""
import os

import duckdb

DB = os.environ.get("DUCKDB_PATH", "data/warehouse/overdose.duckdb")


def query(sql, params=None):
    """Run a SQL query against the warehouse and return a pandas DataFrame."""
    if not os.path.exists(DB):
        raise FileNotFoundError(f"{DB} not found: build it with `make data` "
                                "(python -m src.ingest, then dbt build)")
    con = duckdb.connect(DB, read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()
