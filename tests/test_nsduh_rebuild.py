"""The rebuilt comparable outcomes reproduce SAMHSA's 2021-2023 variables exactly (needs the data)."""
import os

import pytest

from src import nsduh

from src import warehouse

pytestmark = pytest.mark.skipif(not os.path.exists(warehouse.DB) or os.environ.get("SYNTHETIC_FIXTURES") == "1",
                                reason="real-data warehouse not built (run `make data`)")


@pytest.fixture(scope="module")
def data():
    df, _ = nsduh.load_analysis()
    return df


def test_rebuilt_misuse_equals_opinmyr_2021_2023(data):
    m = data.OPINMYR.notna()
    assert m.sum() == 173808
    assert (data.loc[m, "OPI_MISUSE"] == data.loc[m, "OPINMYR"]).all()


def test_rebuilt_oud_equals_ud5opiany_2021_2023(data):
    m = data.UD5OPIANY.notna()
    assert (data.loc[m, "OPI_OUD"] == data.loc[m, "UD5OPIANY"]).all()


def test_no_missing_comparable_outcomes_any_year(data):
    assert data[["OPI_MISUSE", "OPI_OUD", "HERYR"]].notna().all().all()
    assert sorted(data.YEAR.unique()) == [2021, 2022, 2023, 2024]
