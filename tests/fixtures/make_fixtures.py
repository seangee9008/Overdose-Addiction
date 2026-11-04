#!/usr/bin/env python3
"""
Generate SMALL SYNTHETIC fixtures in tests/fixtures/raw/ that mimic the layout of each real raw
file (same file names, columns, formats and quirks), for CI. ALL NUMBERS ARE MADE UP -- never real
data. Each fixture deliberately contains the edge cases the data layer must handle:

  CDC state:  US/FL/LA; a value written with a thousands comma; FL drug-specific months left blank
              (suppressed); LA with NO drug-specific rows at all; 'Number of Deaths' without
              Predicted Value.
  CDC county: FIPS without leading zeros; suppressed (blank + NCHS footnote) cells; true zeros;
              the Virginia duplicate pair (Alleghany / Clifton Forge, both 51005, identical values);
              Connecticut legacy county 09001 and Alaska 02261 (no SVI match).
  SVI:        5-character FIPS; CT planning region 09110 and AK 02063 (no CDC match).
  Census:     Latin-1 encoded (a 'Doña Ana' row), state (SUMLEV 40) and county (SUMLEV 50) rows.
  WONDER:     CDC WONDER-style export (quoted labels, '(provisional)' / '(provisional and partial)'
              month labels, a 'Total' row and a footer with 'Query Parameters').
  NSDUH:      mixed-case column names as in the real .dta; YEAR stored as text; -9 "not collected"
              for 2024 OPINMYR / UD5OPIANY / UDYR5PNRANY and 2021 IMFYR; value labels; a separate
              2024 single-year file carrying UDYR5PNRANY2.

    python tests/fixtures/make_fixtures.py     # rewrites tests/fixtures/raw/ deterministically
"""
import os

import numpy as np
import pandas as pd
import pyreadstat

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
ALL, OPI, SYN, NDEATHS = ("Number of Drug Overdose Deaths", "Opioids (T40.0-T40.4,T40.6)",
                          "Synthetic opioids, excl. methadone (T40.4)", "Number of Deaths")
NOTE = "Numbers may differ from published reports using final data. See Technical Notes."
SUPP = "One or more data cells have counts between 1-9 and have been suppressed in accordance with NCHS confidentiality standards."


def cdc_state(rng):
    rows = []
    names = {"US": "United States", "FL": "Florida", "LA": "Louisiana"}
    base = {"US": 90000, "FL": 7000, "LA": 2000}
    for st in ("US", "FL", "LA"):
        for y in (2021, 2022):
            for mi, m in enumerate(MONTHS):
                t = (y - 2021) * 12 + mi
                for ind, share in ((NDEATHS, 30.0), (ALL, 1.0), (OPI, 0.75), (SYN, 0.65)):
                    if st == "LA" and ind in (OPI, SYN):
                        continue                                  # whole indicator absent
                    v = int(base[st] * share + 40 * t + rng.integers(0, 50))
                    dv = f"{v:,}" if (st == "US" and ind == NDEATHS and t == 0) else str(v)
                    if st == "FL" and ind in (OPI, SYN) and y == 2021 and mi < 3:
                        dv = ""                                   # suppressed / low quality
                    pv = "" if ind == NDEATHS or dv == "" else str(v + int(rng.integers(0, 30)))
                    rows.append({"State": st, "Year": str(y), "Month": m, "Period": "12 month-ending",
                                 "Indicator": ind, "Data Value": dv, "Percent Complete": "100",
                                 "Percent Pending Investigation": f"{rng.uniform(0, 0.3):.6f}",
                                 "State Name": names[st], "Footnote": NOTE, "Footnote Symbol": "**",
                                 "Predicted Value": pv})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cdc_vsrr_overdose_national_state.csv"), index=False)


def cdc_county(rng):
    counties = [("AL", "Alabama", "Autauga", "1001", 1, 1, 3, "suppressed_often"),
                ("AL", "Alabama", "Baldwin", "1003", 1, 3, 4, "large"),
                ("VA", "Virginia", "Alleghany", "51005", 51, 5, 6, "small_with_zero"),
                ("VA", "Virginia", "Clifton Forge (city)", "51005", 51, 5, 6, "small_with_zero"),
                ("CT", "Connecticut", "Fairfield", "9001", 9, 1, 2, "large"),
                ("AK", "Alaska", "Valdez-Cordova (CA)", "2261", 2, 261, 6, "zero")]
    va = {}
    rows = []
    for st, sn, name, fips, sf, cf, code, kind in counties:
        for y in range(2022, 2026):
            for mo in range(1, 13):
                end = pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0)
                if kind == "large":
                    v = float(40 + rng.integers(0, 60))
                elif kind == "suppressed_often":
                    v = np.nan if (mo % 3) else float(10 + rng.integers(0, 5))
                elif kind == "zero":
                    v = 0.0
                else:                                             # Virginia pair: identical values
                    if (y, mo) not in va:
                        va[(y, mo)] = 0.0 if mo == 12 and y == 2023 else (np.nan if mo % 2 else float(11 + mo))
                    v = va[(y, mo)]
                rows.append({"Data as of": "07/05/2026", "Year": y, "Month": mo, "ST_ABBREV": st,
                             "STATE_NAME": sn, "COUNTYNAME": name, "FIPS": fips, "STATEFIPS": sf,
                             "COUNTYFIPS": cf, "CODE2013": code, "Provisional Drug Overdose Deaths": v,
                             "Footnote": SUPP if np.isnan(v) else np.nan,
                             "Percentage Of Records Pending Investigation": round(float(rng.uniform(0, 2)), 2),
                             "HistoricalDataCompletenessNote": "Completeness of provisional 2019 data under 90% after 6 months.",
                             "MonthEndingDate": end.strftime("%m/%d/%Y"), "Start Date": "01/31/2022",
                             "End Date": "12/31/2025"})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cdc_vsrr_overdose_county.csv"), index=False)


def svi():
    rows = [("01", "Alabama", "AL", "01001", "Autauga County", 58761, 0.2663),
            ("01", "Alabama", "AL", "01003", "Baldwin County", 233420, 0.4000),
            ("51", "Virginia", "VA", "51005", "Alleghany County", 15174, 0.7500),
            ("09", "Connecticut", "CT", "09110", "Capitol Planning Region", 976248, 0.5100),
            ("02", "Alaska", "AK", "02063", "Chugach Census Area", 7019, 0.1500)]
    df = pd.DataFrame(rows, columns=["ST", "STATE", "ST_ABBR", "FIPS", "COUNTY", "E_TOTPOP", "RPL_THEMES"])
    df.insert(3, "STCNTY", df.FIPS)
    df["LOCATION"] = df.COUNTY + ", " + df.STATE
    df["EP_POV150"] = [20.2, 18.0, 30.1, 15.5, 22.0]
    df.to_csv(os.path.join(OUT, "svi_2022_us_county.csv"), index=False)


def census():
    areas = [(40, "01", "000", "Alabama", "Alabama", 5_000_000), (50, "01", "001", "Alabama", "Autauga County", 59_000),
             (50, "01", "003", "Alabama", "Baldwin County", 240_000), (40, "51", "000", "Virginia", "Virginia", 8_600_000),
             (50, "51", "005", "Virginia", "Alleghany County", 15_000), (40, "35", "000", "New Mexico", "New Mexico", 2_100_000),
             (50, "35", "013", "New Mexico", "Doña Ana County", 220_000)]
    rows = []
    for sumlev, st, co, stn, ctyn, base in areas:
        r = {"SUMLEV": sumlev, "REGION": 3, "DIVISION": 5, "STATE": st, "COUNTY": co, "STNAME": stn, "CTYNAME": ctyn,
             "ESTIMATESBASE2020": base}
        r.update({f"POPESTIMATE{y}": base + 1000 * (y - 2020) for y in range(2020, 2026)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "census_co_est2025_alldata.csv"), index=False, encoding="latin-1")


def wonder(rng):
    os.makedirs(os.path.join(OUT, "wonder"), exist_ok=True)
    lines = ['"Notes","Month","Month Code",Deaths,Population,Crude Rate']
    total = 0
    for y in range(2023, 2027):
        for mo in range(1, 13):
            if y == 2026 and mo > 4:
                break
            tag = "" if y < 2025 else (" (provisional)" if y == 2025 or mo == 1 else " (provisional and partial)")
            d = int(9000 - 120 * ((y - 2023) * 12 + mo) + rng.integers(0, 300))
            total += d
            lines.append(f',"{MONTHS[mo - 1][:3]}., {y}{tag}","{y}/{mo:02d}",{d},Not Applicable,Not Applicable')
    lines += [f'"Total",,,{total},Not Applicable,Not Applicable', '"---"', '"Dataset: SYNTHETIC fixture (not real data)"',
              '"Query Parameters:"', '"UCD - Drug/Alcohol Induced Causes: Drug-induced causes"', '"Group By: Month"', '"---"']
    with open(os.path.join(OUT, "wonder", "wonder_drug_induced_monthly.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def nsduh(rng):
    n_per_year, rows = 60, []
    for y in (2021, 2022, 2023, 2024):
        for i in range(n_per_year):
            her = int(rng.random() < 0.05)
            pnr = int(rng.random() < 0.10)
            pnr_ud = int(pnr and rng.random() < 0.5)
            her_ud = int(her and rng.random() < 0.5)
            rows.append({
                "year": str(y), "QUESTID2": 10_000 + i, "ANALWT2_C1": float(rng.uniform(1000, 9000)),
                "VESTR_C": 40001 + (i % 3), "verep": 1 + (i // 3) % 2,
                "heryr": her, "pnrnmyr": pnr, "IRPYUD5HER": her_ud,
                "UDYR5PNRANY": -9 if y == 2024 else pnr_ud,
                "opinmyr": -9 if y == 2024 else int(her or pnr),
                "UD5OPIANY": -9 if y == 2024 else int(her_ud or pnr_ud),
                "IMFYR": -9 if y == 2021 else int(rng.random() < 0.02),
                "CATAG3": 1 + i % 5, "irsex": 1 + i % 2, "NEWRACE2": 1 + i % 7,
                "_pnr_ud_2024": pnr_ud})
    df = pd.DataFrame(rows)
    labels = {"CATAG3": {1: "1 - 12-17 Years Old", 2: "2 - 18-25 Years Old", 3: "3 - 26-34 Years Old",
                         4: "4 - 35-49 Years Old", 5: "5 - 50 or Older"},
              "irsex": {1: "1 - Male", 2: "2 - Female"},
              "NEWRACE2": {k: f"{k} - Group {k}" for k in range(1, 8)},
              "heryr": {0: "0 - Did not use in the past year", 1: "1 - Used within the past year"}}
    os.makedirs(os.path.join(OUT, "nsduh_2021_2024"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "nsduh_2024"), exist_ok=True)
    pyreadstat.write_dta(df.drop(columns="_pnr_ud_2024"), os.path.join(OUT, "nsduh_2021_2024", "NSDUH_2021_2024.dta"),
                         variable_value_labels=labels)
    y24 = df[df.year == "2024"][["QUESTID2", "_pnr_ud_2024"]].rename(columns={"_pnr_ud_2024": "UDYR5PNRANY2"})
    pyreadstat.write_dta(y24, os.path.join(OUT, "nsduh_2024", "NSDUH_2024.dta"))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(12345)
    cdc_state(rng)
    cdc_county(rng)
    svi()
    census()
    wonder(rng)
    nsduh(rng)
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("SYNTHETIC test fixtures generated by tests/fixtures/make_fixtures.py. "
                 "All numbers are made up; these are NOT real data.\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
