"""
Loader for the combined NSDUH 2021-2024 public-use file (Stata), as downloaded by fetch_data.py
and unzipped to data/raw/nsduh_2021_2024/.

Variable names follow the combined-file codebook (nsduh_2021_2024_concat_codebook.pdf). The .dta
stores some names in lower case (e.g. `year`, `verep`, `opinmyr`), so names are matched
case-insensitively and returned in codebook (upper) case.

Design (codebook addendum, "Weights and Design Variables for the Combined Data File"):
  - ANALWT2_C1: weight for single-year estimates and for comparing single years with each other
    (identical to ANALWT2_C on the single-year PUFs).
  - VESTR_C / VEREP: variance strata / PSUs, valid for all analyses; years share one sample design,
    so a 2021-vs-2024 comparison keeps 50 degrees of freedom.
  - Codebook value -9 = "Data not collected for this year"; returned as NaN.
"""
import numpy as np
import pyreadstat

DTA = "data/raw/nsduh_2021_2024/NSDUH_2021_2024.dta"
WEIGHT, STRATA, PSU = "ANALWT2_C1", "VESTR_C", "VEREP"
DESIGN_DF = 50  # codebook: 50 pseudo-strata, 50 degrees of freedom

# Outcome variables (0/1) and their codebook labels.
OUTCOMES = {
    "OPINMYR": "Opioid misuse, past year (heroin use or Rx pain-reliever misuse)",
    "UD5OPIANY": "Opioid use disorder, past year (DSM-5)",
    "HERYR": "Heroin use, past year",
    "PNRNMYR": "Rx pain-reliever misuse, past year",
}
DEMOGRAPHICS = ["CATAG3", "CATAG6", "IRSEX", "NEWRACE2"]


def load(path=DTA, columns=None):
    """Return (DataFrame with codebook-case column names, value-label dict)."""
    columns = columns or ["YEAR", "QUESTID2", WEIGHT, STRATA, PSU, *OUTCOMES, *DEMOGRAPHICS]
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    actual = {n.upper(): n for n in meta.column_names}
    missing = [c for c in columns if c.upper() not in actual]
    if missing:
        raise KeyError(f"not in {path}: {missing}")
    df, meta = pyreadstat.read_dta(path, usecols=[actual[c.upper()] for c in columns])
    df.columns = [c.upper() for c in df.columns]
    if "YEAR" in df:  # stored as a 4-character string in the .dta
        df["YEAR"] = df["YEAR"].astype(int)
    labels = {k.upper(): v for k, v in meta.variable_value_labels.items()}
    df = df.replace({c: {-9: np.nan} for c in df.columns if c not in (WEIGHT, STRATA, PSU, "QUESTID2")})
    return df, labels
