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
DTA_2024 = "data/raw/nsduh_2024/NSDUH_2024.dta"   # single-year 2024 PUF (for UDYR5PNRANY2)
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


# ---------------------------------------------------------------------------------------------
# Comparable 2021-2024 outcomes.
#
# In 2024 SAMHSA redefined OPINMYR2 / UD5OPIANY2 around prescription OPIOID pain relievers and
# states they are "not comparable" with 2021-2023 (2024 codebook). The 2021-2023 definitions are
# rebuilt from components the codebooks treat as comparable across all four years:
#   OPI_MISUSE = HERYR==1 or PNRNMYR==1           (combined codebook: definition of OPINMYR)
#   OPI_OUD    = IRPYUD5HER==1 or UDYR5PNRANY==1  (reproduces UD5OPIANY exactly on 2021-2023);
#                for 2024 UDYR5PNRANY2 is used ("still considered comparable", 2024 codebook)
# Both rebuilds match SAMHSA's variables on all 173,808 records for 2021-2023 (checked in tests).
COMPARABLE = {
    "OPI_MISUSE": "Opioid misuse, past year (heroin use or Rx pain-reliever misuse; 2021-23 definition)",
    "OPI_OUD": "Opioid use disorder, past year (heroin or Rx pain-reliever use disorder; 2021-23 definition)",
    "HERYR": "Heroin use, past year",
}


def load_analysis(demographics=("CATAG3", "IRSEX", "NEWRACE2")):
    """Combined 2021-2024 file with the comparable outcomes OPI_MISUSE, OPI_OUD, HERYR (0/1),
    SAMHSA's own OPINMYR / UD5OPIANY (2021-2023 only, for checking), design variables, and
    demographics. Returns (DataFrame, value labels)."""
    cols = ["YEAR", "QUESTID2", WEIGHT, STRATA, PSU, "HERYR", "PNRNMYR", "IRPYUD5HER",
            "UDYR5PNRANY", "OPINMYR", "UD5OPIANY", *demographics]
    df, labels = load(columns=cols)
    y24, _ = load(path=DTA_2024, columns=["QUESTID2", "UDYR5PNRANY2"])
    is24 = df.YEAR == 2024
    pnr_ud = df["UDYR5PNRANY"].copy()
    m24 = df.loc[is24, ["QUESTID2"]].merge(y24, on="QUESTID2", how="left", validate="one_to_one")
    if m24["UDYR5PNRANY2"].isna().any():
        raise ValueError("2024 records missing from the single-year file")
    pnr_ud.loc[is24] = m24["UDYR5PNRANY2"].to_numpy()
    if pnr_ud.isna().any():
        raise ValueError("pain-reliever use disorder missing for some records")
    df["OPI_MISUSE"] = ((df.HERYR == 1) | (df.PNRNMYR == 1)).astype(float)
    df["OPI_OUD"] = ((df.IRPYUD5HER == 1) | (pnr_ud == 1)).astype(float)
    return df, labels
