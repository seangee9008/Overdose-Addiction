"""
pandera schemas for each raw source, checked at the ingestion boundary (src/ingest.py).

A schema states what a table must look like: required columns, types, allowed values and ranges.
validate() collects every failure (lazy=True) and raises IngestionError naming the raw file,
the column, the failed check and example offending values.
"""
import pandas as pd
import pandera.pandas as pa
from pandera import Check

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


class IngestionError(ValueError):
    pass


def _numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _int_valued(s):
    s = s.dropna()
    return bool(((s % 1) == 0).all())


def _parses_nonneg_number(s):
    """CDC state 'Data Value'/'Predicted Value' are strings like '1,234'; must parse to >= 0."""
    v = pd.to_numeric(s.dropna().str.replace(",", ""), errors="coerce")
    return bool(v.notna().all() and (v >= 0).all())


NUM = Check(_numeric, element_wise=False, error="column must be numeric")
INT_VALUED = Check(_int_valued, element_wise=False, error="values must be whole numbers")

CDC_STATE = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "State": pa.Column(checks=Check.str_matches(r"^[A-Z]{2}$")),
    "Year": pa.Column(checks=Check.str_matches(r"^(20[1-3][0-9])$")),
    "Month": pa.Column(checks=Check.isin(MONTHS)),
    "Period": pa.Column(checks=Check.eq("12 month-ending")),
    "Indicator": pa.Column(nullable=False),
    "Data Value": pa.Column(nullable=True, checks=Check(_parses_nonneg_number, element_wise=False,
                                                         error="must parse to a number >= 0")),
    "Predicted Value": pa.Column(nullable=True, checks=Check(_parses_nonneg_number, element_wise=False,
                                                              error="must parse to a number >= 0")),
    "Percent Complete": pa.Column(nullable=True),
    "Percent Pending Investigation": pa.Column(nullable=True),
    "State Name": pa.Column(nullable=True),
    "Footnote": pa.Column(nullable=True),
    "Footnote Symbol": pa.Column(nullable=True),
}, strict=True, unique=["State", "Year", "Month", "Indicator"])

CDC_COUNTY = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "Data as of": pa.Column(checks=Check.str_matches(r"^\d{2}/\d{2}/\d{4}$")),
    "Year": pa.Column(checks=[NUM, Check.in_range(2015, 2035)]),
    "Month": pa.Column(checks=[NUM, Check.in_range(1, 12)]),
    "ST_ABBREV": pa.Column(checks=Check.str_matches(r"^[A-Z]{2}$")),
    "STATE_NAME": pa.Column(),
    "COUNTYNAME": pa.Column(),
    "FIPS": pa.Column(checks=Check.str_matches(r"^\d{1,5}$")),     # raw file drops leading zeros
    "STATEFIPS": pa.Column(checks=NUM),
    "COUNTYFIPS": pa.Column(checks=NUM),
    "CODE2013": pa.Column(checks=[NUM, Check.isin([1, 2, 3, 4, 5, 6])]),
    "Provisional Drug Overdose Deaths": pa.Column(nullable=True, checks=[
        NUM, INT_VALUED, Check.ge(0),
        Check(lambda s: ~s.between(1, 9), error="counts 1-9 must be suppressed (blank) in the raw file")]),
    "Footnote": pa.Column(nullable=True),
    "Percentage Of Records Pending Investigation": pa.Column(nullable=True, checks=[NUM, Check.in_range(0, 100)]),
    "HistoricalDataCompletenessNote": pa.Column(nullable=True),
    "MonthEndingDate": pa.Column(checks=Check.str_matches(r"^\d{2}/\d{2}/\d{4}$")),
    "Start Date": pa.Column(checks=Check.str_matches(r"^\d{2}/\d{2}/\d{4}$")),
    "End Date": pa.Column(checks=Check.str_matches(r"^\d{2}/\d{2}/\d{4}$")),
}, strict=True)

SVI_COUNTY = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "FIPS": pa.Column(checks=Check.str_matches(r"^\d{5}$"), unique=True),
    "ST_ABBR": pa.Column(checks=Check.str_matches(r"^[A-Z]{2}$")),
    "LOCATION": pa.Column(),
    "E_TOTPOP": pa.Column(checks=[NUM, INT_VALUED, Check.ge(0)]),
    # SVI uses -999 for missing; otherwise a percentile in [0, 1]
    "RPL_THEMES": pa.Column(checks=[NUM, Check(lambda s: s.eq(-999) | s.between(0, 1),
                                                error="must be in [0, 1] or -999")]),
}, strict=False)   # 158 SVI columns; only the ones the pipeline uses are constrained

BIN_OR_NOT_COLLECTED = Check.isin([-9, 0, 1])
NSDUH_2124 = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "YEAR": pa.Column(checks=Check.isin(["2021", "2022", "2023", "2024"])),   # stored as text in the .dta
    "QUESTID2": pa.Column(checks=[NUM, Check.gt(0)]),
    "ANALWT2_C1": pa.Column(checks=[NUM, Check.gt(0)]),
    "VESTR_C": pa.Column(checks=NUM),
    "VEREP": pa.Column(checks=Check.isin([1, 2])),
    "HERYR": pa.Column(checks=Check.isin([0, 1])),
    "PNRNMYR": pa.Column(checks=Check.isin([0, 1])),
    "IRPYUD5HER": pa.Column(checks=Check.isin([0, 1])),
    "UDYR5PNRANY": pa.Column(checks=BIN_OR_NOT_COLLECTED),
    "OPINMYR": pa.Column(checks=BIN_OR_NOT_COLLECTED),
    "UD5OPIANY": pa.Column(checks=BIN_OR_NOT_COLLECTED),
    "IMFYR": pa.Column(checks=BIN_OR_NOT_COLLECTED),
    "CATAG3": pa.Column(checks=Check.isin([1, 2, 3, 4, 5])),
    "IRSEX": pa.Column(checks=Check.isin([1, 2])),
    "NEWRACE2": pa.Column(checks=Check.isin([1, 2, 3, 4, 5, 6, 7])),
}, strict=True, unique=["YEAR", "QUESTID2"])

NSDUH_2024 = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "QUESTID2": pa.Column(checks=[NUM, Check.gt(0)], unique=True),
    "UDYR5PNRANY2": pa.Column(checks=Check.isin([0, 1])),
}, strict=True)

NSDUH_LABELS = pa.DataFrameSchema({
    "variable": pa.Column(), "value": pa.Column(checks=NUM), "label": pa.Column(),
}, strict=True, unique=["variable", "value"])

CENSUS_POP = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "SUMLEV": pa.Column(checks=Check.isin([40, 50])),            # 40 = state, 50 = county
    "STATE": pa.Column(checks=Check.str_matches(r"^\d{2}$")),
    "COUNTY": pa.Column(checks=Check.str_matches(r"^\d{3}$")),
    "STNAME": pa.Column(),
    "CTYNAME": pa.Column(),
    **{f"POPESTIMATE{y}": pa.Column(checks=[NUM, INT_VALUED, Check.gt(0)]) for y in range(2020, 2026)},
}, strict=True, unique=["STATE", "COUNTY"])

WONDER_MONTHLY = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "series": pa.Column(checks=Check.str_matches(r"^[A-Za-z0-9_]+$")),
    "Month": pa.Column(),
    "Month Code": pa.Column(checks=Check.str_matches(r"^(19|20)\d{2}/(0[1-9]|1[0-2])$")),
    "Deaths": pa.Column(checks=Check.str_matches(r"^\d+$")),     # counts as text; 'Suppressed' would fail loudly
}, strict=True, unique=["series", "Month Code"])

RATE_TEXT = r"^(\d+(\.\d+)?( \(Unreliable\))?|Not Applicable|Unreliable|Suppressed|)$"
WONDER_ANNUAL = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "series": pa.Column(checks=Check.str_matches(r"^[A-Za-z0-9_]+$")),
    "Year": pa.Column(),
    "Year Code": pa.Column(checks=Check.str_matches(r"^(19|20)\d{2}$")),
    "Deaths": pa.Column(checks=Check.str_matches(r"^\d+$")),
    "Population": pa.Column(checks=Check.str_matches(r"^(\d+|Not Applicable|)$")),
    "Crude Rate": pa.Column(checks=Check.str_matches(RATE_TEXT)),
    "Age Adjusted Rate": pa.Column(checks=Check.str_matches(RATE_TEXT)),
    "Age Adjusted Rate Lower 95% Confidence Interval": pa.Column(checks=Check.str_matches(RATE_TEXT)),
    "Age Adjusted Rate Upper 95% Confidence Interval": pa.Column(checks=Check.str_matches(RATE_TEXT)),
}, strict=True, unique=["series", "Year Code"])

def _codes(*vals):
    """Allowed TEDS codes (-9 = missing/unknown/not collected/invalid, per the codebook)."""
    return Check.isin([-9, *vals])


SUBST = _codes(*range(1, 20))
TEDS_A = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM),
    "source_file": pa.Column(checks=Check.isin(["teds_a_2006_2023_csv.zip", "teds_a_2024_csv.zip"])),
    "CASEID": pa.Column(checks=[NUM, Check.gt(0)]),
    "ADMYR": pa.Column(checks=Check.in_range(2006, 2030)),
    "STFIPS": pa.Column(checks=Check.isin([1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45,
                                           46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 72])),
    "SUB1": pa.Column(checks=SUBST), "SUB2": pa.Column(checks=SUBST), "SUB3": pa.Column(checks=SUBST),
    "HERFLG": pa.Column(checks=Check.isin([0, 1])), "METHFLG": pa.Column(checks=Check.isin([0, 1])),
    "OPSYNFLG": pa.Column(checks=Check.isin([0, 1])),
    "AGE": pa.Column(checks=Check.isin(list(range(1, 13)))),
    "SEX": pa.Column(checks=_codes(1, 2)),
    "RACE": pa.Column(checks=_codes(*range(1, 10))),
    "ETHNIC": pa.Column(checks=_codes(*range(1, 6))),
    "NOPRIOR": pa.Column(checks=_codes(0, 1, 2, 3, 4, 5)),
    "METHUSE": pa.Column(checks=_codes(1, 2)),
    "SERVICES": pa.Column(checks=_codes(*range(1, 9))),
    "PSOURCE": pa.Column(checks=_codes(*range(1, 8))),
    "ROUTE1": pa.Column(checks=_codes(*range(1, 6))),
    "IDU": pa.Column(checks=_codes(0, 1)),
}, strict=True)

CENSUS_STATE_AGE = pa.DataFrameSchema({
    "src_row": pa.Column(checks=NUM, unique=True),
    "REGION": pa.Column(checks=Check.isin([1, 2, 3, 4])),          # Census region: NE, MW, S, W
    "STATE": pa.Column(checks=Check.str_matches(r"^\d{2}$")),
    "NAME": pa.Column(),
    "RACE": pa.Column(checks=Check.isin([1, 2, 3, 4, 5, 6])),
    "AGE": pa.Column(checks=Check.in_range(0, 85)),
    "year": pa.Column(checks=Check.in_range(2010, 2025)),
    "population": pa.Column(checks=[NUM, INT_VALUED, Check.ge(0)]),
    "vintage": pa.Column(checks=Check.isin(["intercensal 2010-2020", "Vintage 2025"])),
    "file": pa.Column(),
}, strict=True, unique=["STATE", "RACE", "AGE", "year"])

SCHEMAS = {"cdc_state": CDC_STATE, "cdc_county": CDC_COUNTY, "svi_county": SVI_COUNTY,
           "nsduh_2021_2024": NSDUH_2124, "nsduh_2024": NSDUH_2024, "nsduh_value_labels": NSDUH_LABELS,
           "census_pop": CENSUS_POP, "wonder_monthly": WONDER_MONTHLY,
           "wonder_annual": WONDER_ANNUAL, "teds_a": TEDS_A, "census_state_age": CENSUS_STATE_AGE}


def validate(table, df, raw_path):
    """Validate df against the table's schema; raise IngestionError naming file + columns."""
    try:
        SCHEMAS[table].validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        lines = []
        for (col, check), g in fc.groupby(["column", "check"], dropna=False, sort=False):
            ex = g["failure_case"].astype(str).head(5).tolist()
            lines.append(f"  column {col!r}: check {check!r} failed for {len(g)} value(s), e.g. {ex}")
        raise IngestionError(f"{raw_path} (table {table}) failed validation:\n" + "\n".join(lines)) from None
    return True

