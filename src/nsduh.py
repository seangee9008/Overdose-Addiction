"""
Loader for NSDUH 2021-2024 respondent data, read from the dbt mart `nsduh_respondent` in the DuckDB
warehouse (`make data`). The mart is built from the combined NSDUH 2021-2024 PUF (Stata) and, for
2024 pain-reliever use disorder only, the single-year 2024 PUF; src/ingest.py reads those files
(case-insensitive names; the .dta stores e.g. `year`, `verep`, `opinmyr` in lower case).

Variable names follow the combined-file codebook (nsduh_2021_2024_concat_codebook.pdf).

Design (codebook addendum, "Weights and Design Variables for the Combined Data File"):
  - ANALWT2_C1: weight for single-year estimates and for comparing single years with each other
    (identical to ANALWT2_C on the single-year PUFs).
  - VESTR_C / VEREP: variance strata / PSUs, valid for all analyses; years share one sample design,
    so a 2021-vs-2024 comparison keeps 50 degrees of freedom.
  - Codebook value -9 = "Data not collected for this year"; returned as NaN (done in dbt staging).
"""
from src import warehouse

DTA = "data/raw/nsduh_2021_2024/NSDUH_2021_2024.dta"
DTA_2024 = "data/raw/nsduh_2024/NSDUH_2024.dta"   # single-year 2024 PUF (for UDYR5PNRANY2)
# (DTA / DTA_2024 name the raw sources for provenance strings; data are read from the warehouse.)
WEIGHT, STRATA, PSU = "ANALWT2_C1", "VESTR_C", "VEREP"
DESIGN_DF = 50  # codebook: 50 pseudo-strata, 50 degrees of freedom

# Outcome variables (0/1) and their codebook labels.
OUTCOMES = {
    "OPINMYR": "Opioid misuse, past year (heroin use or Rx pain-reliever misuse)",
    "UD5OPIANY": "Opioid use disorder, past year (DSM-5)",
    "HERYR": "Heroin use, past year",
    "PNRNMYR": "Rx pain-reliever misuse, past year",
}
DEMOGRAPHICS = ["CATAG3", "IRSEX", "NEWRACE2"]


def _labels():
    lab = warehouse.query("select variable, value, label from nsduh_value_labels order by variable, value")
    out = {}
    for r in lab.itertuples(index=False):
        out.setdefault(r.variable, {})[int(r.value) if float(r.value).is_integer() else r.value] = r.label
    return out


def load(columns=None):
    """Return (DataFrame with codebook-case column names, value-label dict), in raw-file row order."""
    columns = columns or ["YEAR", "QUESTID2", WEIGHT, STRATA, PSU, *OUTCOMES, *DEMOGRAPHICS]
    cols = ", ".join(f'"{c}"' for c in columns)
    return warehouse.query(f"select {cols} from nsduh_respondent order by src_row"), _labels()


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
# The derivation lives in dbt/models/marts/nsduh_respondent.sql (dbt tests: not null, 0/1).
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
            "UDYR5PNRANY", "OPINMYR", "UD5OPIANY", *demographics, "OPI_MISUSE", "OPI_OUD"]
    return load(columns=cols)
