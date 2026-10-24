"""Ingestion schemas: good data passes; violations fail loudly, naming file and column."""
import pandas as pd
import pytest

from src.schemas import IngestionError, validate


def county(**over):
    row = {"src_row": 0, "Data as of": "07/05/2026", "Year": 2023, "Month": 12, "ST_ABBREV": "AL",
           "STATE_NAME": "Alabama", "COUNTYNAME": "Autauga", "FIPS": "1001", "STATEFIPS": 1, "COUNTYFIPS": 1,
           "CODE2013": 3, "Provisional Drug Overdose Deaths": 12.0, "Footnote": None,
           "Percentage Of Records Pending Investigation": 0.5, "HistoricalDataCompletenessNote": None,
           "MonthEndingDate": "12/31/2023", "Start Date": "01/31/2020", "End Date": "12/31/2025"}
    row.update(over)
    return pd.DataFrame([row])


def test_valid_county_row_passes():
    assert validate("cdc_county", county(), "raw/county.csv")


@pytest.mark.parametrize("over,column", [
    ({"FIPS": "AB12"}, "FIPS"),
    ({"Provisional Drug Overdose Deaths": 5.0}, "Provisional Drug Overdose Deaths"),   # 1-9 must be suppressed
    ({"Provisional Drug Overdose Deaths": -1.0}, "Provisional Drug Overdose Deaths"),
    ({"Month": 13}, "Month"),
])
def test_violation_names_file_and_column(over, column):
    with pytest.raises(IngestionError) as e:
        validate("cdc_county", county(**over), "raw/county.csv")
    msg = str(e.value)
    assert "raw/county.csv" in msg and repr(column) in msg


def test_missing_required_column_fails():
    with pytest.raises(IngestionError, match="raw/county.csv"):
        validate("cdc_county", county().drop(columns=["FIPS"]), "raw/county.csv")


def test_svi_rejects_out_of_range_percentile_and_bad_fips():
    df = pd.DataFrame({"src_row": [0, 1], "FIPS": ["01001", "1003"], "ST_ABBR": ["AL", "AL"],
                       "LOCATION": ["a", "b"], "E_TOTPOP": [100, 200], "RPL_THEMES": [0.5, 1.7]})
    with pytest.raises(IngestionError) as e:
        validate("svi_county", df, "raw/svi.csv")
    assert "'FIPS'" in str(e.value) and "'RPL_THEMES'" in str(e.value)
