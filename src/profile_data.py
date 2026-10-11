#!/usr/bin/env python3
"""
Profile every file in data/raw/ and write data/DATA_PROFILE.md.

Describes what is actually in the downloads (rows, columns, date ranges, distinct
indicator values, missingness, join-key coverage). Run from the repo root:

    python -m src.profile_data
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyreadstat

RAW = "data/raw"
OUT = "data/DATA_PROFILE.md"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def profile_cdc_state(path):
    df = pd.read_csv(path, dtype=str)
    df["date"] = pd.to_datetime(df["Year"] + "-" + df["Month"], format="%Y-%B")
    dv = pd.to_numeric(df["Data Value"].str.replace(",", ""), errors="coerce")
    pv = pd.to_numeric(df["Predicted Value"].str.replace(",", ""), errors="coerce")
    ind = (df.assign(dv=dv.notna(), pv=pv.notna())
             .groupby("Indicator")
             .agg(rows=("State", "size"), data_value_present=("dv", "sum"),
                  predicted_value_present=("pv", "sum"))
             .reset_index())
    us = df[df["State"] == "US"]
    us_ind = (us.assign(dv=dv[us.index].notna())
                .groupby("Indicator")
                .agg(months=("date", "size"), first=("date", "min"), last=("date", "max"),
                     data_value_present=("dv", "sum"))
                .reset_index())
    us_ind["first"] = us_ind["first"].dt.strftime("%Y-%m")
    us_ind["last"] = us_ind["last"].dt.strftime("%Y-%m")
    foot = df["Footnote"].value_counts().reset_index()
    foot.columns = ["Footnote", "rows"]
    return f"""## CDC VSRR provisional overdose deaths, national + state (`{os.path.basename(path)}`)

Source: data.cdc.gov dataset xkb8-kh2a.

- Rows: {len(df):,}; columns ({df.shape[1] - 1}): {", ".join(f"`{c}`" for c in df.columns if c != "date")}
- Date range: {df.date.min():%Y-%m} to {df.date.max():%Y-%m}
- `Period` values: {sorted(df["Period"].unique())}. **Every row is a 12-month-ending total.**
- `State` values ({df.State.nunique()}): {", ".join(sorted(df.State.unique()))}.
  This includes `US` (national), `YC` (New York City, reported separately from `NY`), `DC` and `PR`.
- `Percent Complete` values: {sorted(df["Percent Complete"].unique())}

### Distinct `Indicator` values (all states)

{md_table(ind)}

`Number of Deaths` and `Percent with drugs specified` have no `Predicted Value`.
Drug-specific indicators have fewer rows than the three totals because some state × month cells are not reported.

### National (`State == "US"`) coverage per indicator

{md_table(us_ind)}

### Footnotes

{md_table(foot)}
"""


def profile_cdc_county(path, svi_fips):
    df = pd.read_csv(path, dtype=str)
    v = pd.to_numeric(df["Provisional Drug Overdose Deaths"], errors="coerce")
    fips5 = df["FIPS"].str.zfill(5)
    dates = pd.to_datetime(df["MonthEndingDate"], format="%m/%d/%Y")
    dec23 = (df.assign(v=v)[(df.Year == "2023") & (df.Month == "12")]
             .drop_duplicates("FIPS")["v"].sum())
    rows_per_month = df.groupby(["Year", "Month"]).size()
    code = df["CODE2013"].value_counts().sort_index().to_dict()
    one = df[(df.Year == "2023") & (df.Month == "12")]
    dup = one[one.FIPS.duplicated(keep=False)][["ST_ABBREV", "COUNTYNAME", "FIPS",
                                                 "Provisional Drug Overdose Deaths"]]
    in_cdc_not_svi = sorted(set(fips5) - svi_fips)
    in_svi_not_cdc = sorted(svi_fips - set(fips5))
    return f"""## CDC VSRR provisional overdose deaths, county (`{os.path.basename(path)}`)

Source: data.cdc.gov dataset gb4e-yj24.

- Rows: {len(df):,}; columns ({df.shape[1]}): {", ".join(f"`{c}`" for c in df.columns)}
- `Data as of`: {sorted(df["Data as of"].unique())}
- Month-ending dates: {dates.min():%Y-%m} to {dates.max():%Y-%m} ({dates.nunique()} months);
  rows per month: {rows_per_month.min()}–{rows_per_month.max()}
- Distinct `FIPS`: {df.FIPS.nunique():,}; states (`ST_ABBREV`): {df.ST_ABBREV.nunique()}
- **Duplicate FIPS within a month**: two FIPS appear twice per month (Virginia independent cities
  folded into their counties), with identical values. Deduplicate before summing or joining.
  Example, 2023-12:

{md_table(dup)}

- **No drug/indicator column.** This file is all-drug overdose deaths only; it cannot give opioid-specific county counts.
- **Values are 12-month-ending totals**, like the state file. Check: summing every county for 2023-12
  (one row per FIPS) gives {dec23:,.0f}, versus 106,881 for `US` / `Number of Drug Overdose Deaths` / 2023-12 in the state file.
  The shortfall is consistent with suppressed cells.
- Suppressed/missing `Provisional Drug Overdose Deaths`: {v.isna().mean():.1%} of rows
  (footnote: counts 1–9 suppressed under NCHS confidentiality rules).
- `CODE2013` (NCHS 2013 urban–rural code, 1 = most urban … 6 = most rural) row counts: {code}
- `Footnote` values: {sorted(df["Footnote"].dropna().unique())}
- `HistoricalDataCompletenessNote` values: {sorted(df["HistoricalDataCompletenessNote"].dropna().unique())}

### Join key (FIPS) against SVI 2022

- `FIPS` is stored **without leading zeros** (e.g. `1001`); zero-pad to 5 characters before joining.
- After padding, in county file but not SVI: {in_cdc_not_svi}
- In SVI but not county file: {in_svi_not_cdc}
- Connecticut: the county file uses the 8 legacy counties (09001–09015); SVI 2022 uses the 9 planning
  regions (09110–09190). Alaska: 02261 (Valdez-Cordova) vs 02063/02066 (Chugach/Copper River).
  These do not join without a crosswalk.
"""


def profile_svi(path):
    s = pd.read_csv(path, dtype={"ST": str, "STCNTY": str, "FIPS": str})
    num = s.select_dtypes("number")
    sentinel = int((num == -999).sum().sum())
    themes = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES"]
    rng = {c: (float(s[c][s[c] != -999].min()), float(s[c][s[c] != -999].max())) for c in themes}
    return f"""## CDC/ATSDR Social Vulnerability Index 2022, county (`{os.path.basename(path)}`)

Source: svi.cdc.gov official CSV (SVI_2022_US_county.csv).

- Rows: {len(s):,} counties; columns: {s.shape[1]}
- States (`ST_ABBR`): {s.ST_ABBR.nunique()}; `FIPS` is 5-character zero-padded string
  (lengths: {s.FIPS.str.len().value_counts().to_dict()})
- Column families: `E_*` ACS estimates, `M_*` margins of error, `EP_*` percentages,
  `MP_*` percentage MOEs, `EPL_*` percentile ranks, `SPL_*`/`RPL_*` theme sums/ranks, `F_*` flags.
- `-999` (SVI missing-value code) cells: {sentinel:,}
- Theme percentile ranges (excluding -999): {rng}
- All columns: {", ".join(f"`{c}`" for c in s.columns)}
"""


def profile_svi_geo(path):
    g = json.load(open(path))
    feats = g["features"]
    gtypes = pd.Series([f["geometry"]["type"] if f["geometry"] else None for f in feats]).value_counts()
    props = list(feats[0]["properties"])
    return f"""## SVI 2022 county polygons, ArcGIS mirror (`{os.path.basename(path)}`) — opt-in

Source: ArcGIS Online layer `SVI2022_US_county` (metadata credits CDC/ATSDR GRASP; hosted in a
shared public org, so treated as a mirror, not the official source).

- Features: {len(feats):,}; geometry types: {gtypes.to_dict()}
- Properties per feature: {len(props)} (the SVI columns plus ArcGIS fields such as
  {[p for p in props if p.lower().startswith(("objectid", "shape"))]})
"""


def profile_nsduh(path):
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    key = ["QUESTID2", "FILEDATE", "ANALWT2_C", "VESTR_C", "VEREP", "OPINMYR", "PNRNMYR",
           "HERYR", "FENTNMYR", "CATAG6", "IRSEX", "NEWRACE2"]
    present = [k for k in key if k in meta.column_names]
    n_rows, n_cols = meta.number_rows, meta.number_columns
    df, meta = pyreadstat.read_dta(path, usecols=present)
    w = df["ANALWT2_C"]
    rows = []
    for v in ["OPINMYR", "PNRNMYR", "HERYR", "FENTNMYR"]:
        if v in df:
            rows.append(dict(variable=v, label=meta.column_names_to_labels[v],
                             values=df[v].value_counts(dropna=False).sort_index().to_dict(),
                             weighted_pct_eq_1=round(100 * np.average(df[v] == 1, weights=w), 2)))
    var_tbl = pd.DataFrame(rows)
    cat = df["CATAG6"].value_counts().sort_index()
    cat_lbl = meta.variable_value_labels.get("CATAG6", {})
    return f"""## NSDUH 2023 public-use file (`{os.path.relpath(path, RAW)}`)

Source: SAMHSA NSDUH 2023 PUF, Stata bundle (nsduh-2023-ds0001-bndl-data-stata_v2.zip).
Codebook: `nsduh_2023_codebook.pdf`.

- Respondents: {n_rows:,}; variables: {n_cols:,}
- `FILEDATE`: {df["FILEDATE"].unique().tolist()}
- One survey year (2023). **No geography variables** (no state/county), so it cannot be joined to CDC/SVI.
- Survey design: weight `ANALWT2_C` (sum = {w.sum() / 1e6:,.1f} M, the 12+ civilian non-institutional
  population), strata `VESTR_C` ({df.VESTR_C.nunique()} levels), PSU `VEREP` ({sorted(int(x) for x in df.VEREP.unique())}).
- Key-variable presence check: missing = {sorted(set(key) - set(present)) or "none"}
- Age (`CATAG6`) respondent counts: {dict(zip([cat_lbl.get(k, k) for k in cat.index], cat.values.tolist()))}

### Opioid-related past-year variables (weighted with `ANALWT2_C`; point estimates only, no SEs)

{md_table(var_tbl)}

Cross-check: `analyze_real.py` originally hard-coded 2023 opioid misuse as a "3.0–3.1" range.
The PUF weighted `OPINMYR` is {var_tbl.loc[var_tbl.variable == "OPINMYR", "weighted_pct_eq_1"].iloc[0]}%,
while `PNRNMYR` (pain-reliever misuse only) is {var_tbl.loc[var_tbl.variable == "PNRNMYR", "weighted_pct_eq_1"].iloc[0]}%.
The PUF is a disclosure-protected subsample, so small differences from published SAMHSA tables are
expected; check which published measure the hard-coded series is.
"""


def main():
    manifest = json.load(open(os.path.join(RAW, "MANIFEST.json")))
    man = pd.DataFrame([dict(source=k, **v) for k, v in manifest.items()])
    on_disk = sorted(f for f in os.listdir(RAW) if not f.startswith("."))
    pages = subprocess.run(["mdls", "-raw", "-name", "kMDItemNumberOfPages",
                            os.path.join(RAW, "nsduh_2023_codebook.pdf")],
                           capture_output=True, text=True).stdout.strip() or "unknown"
    svi_path = os.path.join(RAW, "svi_2022_us_county.csv")
    svi_fips = set(pd.read_csv(svi_path, dtype={"FIPS": str})["FIPS"])
    parts = [
        "# Data profile\n",
        "Generated by `python -m src.profile_data` from the files in `data/raw/`. "
        "This describes what the downloads actually contain; where it disagrees with `DATA.md`, this file wins.\n",
        "## Files and checksums (`data/raw/MANIFEST.json`)\n",
        md_table(man),
        f"\nOn disk in `data/raw/`: {on_disk}\n",
        profile_cdc_state(os.path.join(RAW, "cdc_vsrr_overdose_national_state.csv")),
        profile_cdc_county(os.path.join(RAW, "cdc_vsrr_overdose_county.csv"), svi_fips),
        profile_svi(svi_path),
        profile_svi_geo(os.path.join(RAW, "svi_2022_county.geojson")),
        profile_nsduh(os.path.join(RAW, "nsduh_2023", "NSDUH_2023.dta")),
        f"""## NSDUH 2023 codebook (`nsduh_2023_codebook.pdf`)

- PDF, {pages} pages. Not parsed; use it to look up variable definitions and value codes.
""",
    ]
    with open(OUT, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
