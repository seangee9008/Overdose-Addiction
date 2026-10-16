"""
Loaders for the CDC/NCHS VSRR "Provisional Drug Overdose Death Counts" file
(data.cdc.gov dataset xkb8-kh2a), as downloaded by fetch_data.py.

Every value in this file is a 12-MONTH-ENDING total: the row for (Year, Month) is the
number of deaths in the 12 months ending in that month, not deaths in that month.
"""
import pandas as pd

STATE_CSV = "data/raw/cdc_vsrr_overdose_national_state.csv"

# Exact indicator strings as they appear in the file (verified against the download).
ALL_DRUG = "Number of Drug Overdose Deaths"
OPIOIDS = "Opioids (T40.0-T40.4,T40.6)"
SYNTHETIC_OPIOIDS = "Synthetic opioids, excl. methadone (T40.4)"

VALUE_COLS = ("Data Value", "Predicted Value")


def load_series(path=STATE_CSV, state="US", indicator=ALL_DRUG, value_col="Data Value",
                allow_missing=False):
    """Return a DataFrame [date, year, month, deaths] for one state x indicator,
    sorted by date. Raises if the indicator/state is not in the file. Missing
    (suppressed / low-quality) values raise unless allow_missing=True, in which case
    they stay NaN -- never filled in -- and deaths is float."""
    if value_col not in VALUE_COLS:
        raise ValueError(f"value_col must be one of {VALUE_COLS}")
    raw = pd.read_csv(path, dtype=str)
    if indicator not in set(raw["Indicator"]):
        raise ValueError(f"indicator {indicator!r} not in file; available: "
                         f"{sorted(raw['Indicator'].unique())}")
    sub = raw[(raw["State"] == state) & (raw["Indicator"] == indicator)]
    if sub.empty:
        raise ValueError(f"no rows for state={state!r}, indicator={indicator!r}")
    if not (sub["Period"] == "12 month-ending").all():
        raise ValueError("expected only 12 month-ending rows")
    vals = pd.to_numeric(sub[value_col].str.replace(",", ""), errors="coerce")
    if vals.isna().any() and not allow_missing:
        missing = sub.loc[vals.isna(), ["Year", "Month"]].to_records(index=False).tolist()
        raise ValueError(f"{value_col} missing (suppressed/low quality) for {missing}")
    out = pd.DataFrame({
        "date": pd.to_datetime(sub["Year"] + "-" + sub["Month"], format="%Y-%B"),
        "deaths": vals if allow_missing else vals.astype(int),
    }).sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("duplicate months in series")
    out.insert(1, "year", out["date"].dt.year)
    out.insert(2, "month", out["date"].dt.month)
    return out
