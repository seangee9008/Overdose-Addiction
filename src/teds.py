"""
Loaders for the TEDS-A area-level marts (DuckDB warehouse, `make data`).

TEDS-A counts treatment ADMISSIONS (episodes), not people, at facilities that receive public funding
(state-dependent scope). Only state x year aggregates are exposed here; nothing is ever joined to
NSDUH microdata.
"""
from src import warehouse

WINDOW = (2015, 2023)


def panel():
    """State x year 2015-2023: admissions, CDC state opioid / all-drug deaths, Census population,
    reporting and completeness flags (teds_deaths_state_year mart)."""
    return warehouse.query("select * from teds_deaths_state_year order by state_fips, year")
