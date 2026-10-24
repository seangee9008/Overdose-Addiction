"""Data-layer behaviour on the SYNTHETIC fixtures (tests/fixtures/raw), built by `make fixtures-data`:
the dbt marts must encode each cleaning rule correctly. Skipped if the fixture warehouse is absent."""
import os

import numpy as np
import pytest

from src import cdc, nsduh, warehouse

FIXTURE_DB = "build/fixtures/warehouse.duckdb"
pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE_DB), reason="run `make fixtures-data` first")


@pytest.fixture(autouse=True)
def use_fixture_db(monkeypatch):
    monkeypatch.setattr(warehouse, "DB", FIXTURE_DB)


def q(sql):
    return warehouse.query(sql)


def test_virginia_duplicates_collapsed_to_one_row_per_month():
    raw = q("select count(*) n from stg_cdc_county_all where fips = '51005'").n[0]
    clean = q("select count(*) n from county_deaths_svi where fips = '51005'").n[0]
    assert raw == 96 and clean == 48
    assert q("select count(*) n from county_fips_audit where side like 'Duplicate%'").n[0] == 2


def test_suppressed_never_zero_and_true_zeros_kept():
    s = q("select * from county_deaths_svi where is_suppressed")
    assert len(s) > 0 and s.deaths_reported.isna().all()
    assert (s.deaths_lower == 1).all() and (s.deaths_upper == 9).all()
    z = q("select * from stg_cdc_county where fips = '02261'")      # all true zeros
    assert (z.deaths_reported == 0).all() and not z.is_suppressed.any()


def test_fips_zero_padded_and_join_losses_listed():
    assert q("select count(*) n from stg_cdc_county where length(fips) <> 5").n[0] == 0
    audit = q("select side, fips from county_fips_audit order by side_order, order_key")
    assert set(audit[audit.side.str.startswith("CDC")].fips) == {"09001", "02261"}
    assert set(audit[audit.side.str.startswith("SVI")].fips) == {"09110", "02063"}


def test_missing_months_flagged_not_imputed():
    la = q(f"select * from state_deaths_long where state = 'LA' and indicator = '{cdc.OPIOIDS}'")
    assert len(la) == 24 and la.is_row_missing.all() and la.data_value.isna().all()
    with pytest.raises(ValueError, match="no rows"):
        cdc.load_series(state="LA", indicator=cdc.OPIOIDS)
    fl = cdc.load_series(state="FL", indicator=cdc.OPIOIDS, allow_missing=True)
    assert fl.deaths.isna().sum() == 3                              # left blank, not filled
    with pytest.raises(ValueError, match="missing"):
        cdc.load_series(state="FL", indicator=cdc.OPIOIDS)


def test_thousands_comma_parsed():
    v = q("select data_value from state_deaths_long where state = 'US' and indicator = 'Number of Deaths' "
          "order by month_date limit 1").data_value[0]               # fixture writes it as '1,234'-style text
    assert v >= 1000 and float(v).is_integer()


def test_nsduh_codes_and_comparable_outcomes():
    df, labels = nsduh.load_analysis()
    assert sorted(df.YEAR.unique()) == [2021, 2022, 2023, 2024]
    assert df.loc[df.YEAR == 2024, "OPINMYR"].isna().all()          # -9 -> missing, not -9
    exp = ((df.HERYR == 1) | (df.PNRNMYR == 1)).astype(float)
    assert np.array_equal(df.OPI_MISUSE.to_numpy(), exp.to_numpy())
    r = q('select * from nsduh_respondent where "YEAR" = 2024')
    assert np.array_equal(r.PNR_USE_DISORDER.to_numpy(), r.UDYR5PNRANY2.to_numpy())
    assert labels["IRSEX"][1] == "1 - Male"
