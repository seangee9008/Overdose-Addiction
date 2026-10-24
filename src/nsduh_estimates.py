#!/usr/bin/env python3
"""
Step 1: NSDUH 2021-2024 design-based estimates (individual-level data only).

    python -m src.nsduh_estimates

Writes
  outputs/nsduh_variables_by_year.csv   which codebook variables have data in which year
  outputs/nsduh_estimates.csv           weighted % with SE and 95% CI: by year, and by year x age / sex
  outputs/nsduh_validation.csv          our estimates vs the published figures previously hard-coded

Source of every number: combined NSDUH 2021-2024 PUF (data/raw/nsduh_2021_2024/NSDUH_2021_2024.dta)
plus, for 2024 OPI_OUD only, UDYR5PNRANY2 from the single-year 2024 PUF; weight ANALWT2_C1 (codebook:
single-year estimates and single-year comparisons), strata VESTR_C, PSU VEREP, 50 design degrees of
freedom. Values coded -9 ("data not collected for this year") are missing.

Outcomes: the comparable 2021-2024 measures OPI_MISUSE, OPI_OUD, HERYR (see src/nsduh.py), plus
PNRNMYR, and SAMHSA's own OPINMYR / UD5OPIANY (2021-2023 only) for reference.
"""
import os

import numpy as np
import pandas as pd

from src import nsduh
from src import survey as sv

OUT = "outputs"
YEARS = [2021, 2022, 2023, 2024]
LABELS = {**nsduh.COMPARABLE, **nsduh.OUTCOMES}
VARS = [*nsduh.COMPARABLE, "PNRNMYR", "OPINMYR", "UD5OPIANY"]
SOURCE = (f"NSDUH 2021-2024 combined PUF ({nsduh.DTA}; 2024 OPI_OUD uses {nsduh.DTA_2024}), weight {nsduh.WEIGHT}, "
          f"strata {nsduh.STRATA}, PSU {nsduh.PSU}, df={nsduh.DESIGN_DF}; Taylor linearization")
# Previously hard-coded "published" past-year opioid misuse %, for validation only.
PUBLISHED = {2021: 3.2, 2022: 3.0, 2023: (3.0, 3.1), 2024: 2.7}


def low_precision(p, se, n):
    """Partial SAMHSA-style flag: nominal n < 100, or RSE of -ln(p) > 0.175 (for p <= 0.5).
    (SAMHSA's full rule also uses effective sample size, not computed here.) This rule is lenient
    for very small p, so the plain relative SE (column `rse` = SE / p) is reported alongside it."""
    if n < 100 or p <= 0:
        return True
    q = p if p <= 0.5 else 1 - p
    return bool(se / (q * -np.log(q)) > 0.175)


def estimate(df, var, year, group=None, level=None):
    d = (df.YEAR == year).to_numpy()
    if group is not None:
        d = d & (df[group] == level).to_numpy()
    if df.loc[d, var].notna().sum() == 0:
        return None
    r = sv.proportion(df, var, nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU, domain=d, df_design=nsduh.DESIGN_DF)
    return dict(variable=var, label=LABELS[var], year=year, group=group or "all",
                level=level, pct=100 * r["p"], se=100 * r["se"], ci_low=100 * r["ci_low"],
                ci_high=100 * r["ci_high"], n=r["n"], rse=r["se"] / r["p"] if r["p"] > 0 else np.nan,
                low_precision=low_precision(r["p"], r["se"], r["n"]),
                source=SOURCE)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, labels = nsduh.load_analysis()

    avail = pd.DataFrame([{**{"variable": v, "label": LABELS.get(v, "")},
                           **{str(y): int(df.loc[df.YEAR == y, v].notna().sum()) for y in YEARS}}
                          for v in [*VARS, nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU, "CATAG3", "IRSEX"]])
    avail.to_csv(f"{OUT}/nsduh_variables_by_year.csv", index=False)

    rows = []
    for var in VARS:
        for y in YEARS:
            rows.append(estimate(df, var, y))
            for g in ("CATAG3", "IRSEX"):
                for lvl in sorted(df[g].unique()):
                    r = estimate(df, var, y, g, lvl)
                    if r:
                        r["level"] = labels[g][lvl]
                    rows.append(r)
    est = pd.DataFrame([r for r in rows if r]).round(3)
    est.to_csv(f"{OUT}/nsduh_estimates.csv", index=False)

    tot = est[est.group == "all"].set_index(["variable", "year"])["pct"]
    val = []
    for y, pub in PUBLISHED.items():
        lo, hi = (pub, pub) if not isinstance(pub, tuple) else pub
        for var in ("OPI_MISUSE", "PNRNMYR"):
            ours = tot.get((var, y), np.nan)
            gap = 0.0 if lo <= ours <= hi else min(abs(ours - lo), abs(ours - hi))
            val.append(dict(year=y, variable=var, ours_pct=round(ours, 2), published=str(pub),
                            abs_gap_pp=round(gap, 2) if not np.isnan(ours) else np.nan,
                            published_source="SAMHSA NSDUH reports, hand-entered in analyze_real.py (not re-verified)"))
    pd.DataFrame(val).to_csv(f"{OUT}/nsduh_validation.csv", index=False)

    print(avail.to_string(index=False))
    print(est[est.group == "all"][["variable", "year", "pct", "se", "ci_low", "ci_high", "n"]].to_string(index=False))
    print(pd.DataFrame(val).to_string(index=False))


if __name__ == "__main__":
    main()
