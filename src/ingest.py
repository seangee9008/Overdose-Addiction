#!/usr/bin/env python3
"""
Ingestion: raw files (data/raw/, never modified) -> validated Parquet (data/interim/).

    python -m src.ingest                       # real data: data/raw -> data/interim
    python -m src.ingest --raw-dir tests/fixtures/raw --out-dir build/fixtures/interim   # CI fixtures

Each raw source is read with EXACTLY the pandas settings the analysis scripts used before this
layer existed, so every parsed value is bit-identical; Parquet then stores those values exactly.
A `src_row` column records each row's position in the raw file so downstream tables can keep the
original order (floating-point sums depend on order). Every table is validated with pandera
(src/schemas.py) before it is written; a violation stops ingestion and names the file and column.

Writes <out-dir>/<table>.parquet and <out-dir>/MANIFEST.json (rows, columns, sha256 of the Parquet
file, sha256 of the raw source).
"""
import argparse
import glob
import hashlib
import json
import os
import zipfile

import pandas as pd
import pyarrow as pa_arrow
import pyarrow.parquet as pq
import pyreadstat

from src import schemas

# NSDUH variables used anywhere in the pipeline (combined 2021-2024 file and single-year 2024 file).
NSDUH_COLUMNS = ["YEAR", "QUESTID2", "ANALWT2_C1", "VESTR_C", "VEREP",            # id, design
                 "HERYR", "PNRNMYR", "IRPYUD5HER", "UDYR5PNRANY",                  # outcome components
                 "OPINMYR", "UD5OPIANY", "IMFYR",                                  # SAMHSA recodes used
                 "CATAG3", "IRSEX", "NEWRACE2"]                                    # demographics
NSDUH_2024_COLUMNS = ["QUESTID2", "UDYR5PNRANY2"]

SOURCES = {
    # table name: (raw file relative to raw dir, reader)
    "cdc_state": "cdc_vsrr_overdose_national_state.csv",
    "cdc_county": "cdc_vsrr_overdose_county.csv",
    "svi_county": "svi_2022_us_county.csv",
    "nsduh_2021_2024": "nsduh_2021_2024/NSDUH_2021_2024.dta",
    "nsduh_2024": "nsduh_2024/NSDUH_2024.dta",
    "census_pop": "census_co_est2025_alldata.csv",
}
CENSUS_COLUMNS = ["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME",
                  *[f"POPESTIMATE{y}" for y in range(2020, 2026)]]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dta(path, columns):
    """Read selected columns case-insensitively (the .dta mixes `year`, `verep`, ...);
    return (DataFrame with codebook upper-case names, value labels)."""
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    actual = {n.upper(): n for n in meta.column_names}
    missing = [c for c in columns if c.upper() not in actual]
    if missing:
        raise KeyError(f"{path}: columns not found: {missing}")
    df, meta = pyreadstat.read_dta(path, usecols=[actual[c] for c in columns])
    df.columns = [c.upper() for c in df.columns]
    labels = {k.upper(): v for k, v in meta.variable_value_labels.items()}
    return df[columns], labels


def read_source(name, path):
    """Return (DataFrame, extra tables dict). Readers mirror the original analysis code."""
    extra = {}
    if name == "cdc_state":
        df = pd.read_csv(path, dtype=str)                         # as src/cdc.py did
    elif name == "cdc_county":
        df = pd.read_csv(path, dtype={"FIPS": str})               # as src/step_d.py did
    elif name == "svi_county":
        df = pd.read_csv(path, dtype={"FIPS": str})               # as src/step_d.py did
    elif name == "nsduh_2021_2024":
        df, labels = read_dta(path, NSDUH_COLUMNS)
        extra["nsduh_value_labels"] = pd.DataFrame(
            [dict(variable=v, value=float(k), label=str(lab))
             for v, d in sorted(labels.items()) if v in NSDUH_COLUMNS for k, lab in d.items()])
    elif name == "nsduh_2024":
        df, _ = read_dta(path, NSDUH_2024_COLUMNS)
    elif name == "census_pop":
        df = pd.read_csv(path, dtype={"STATE": str, "COUNTY": str}, usecols=CENSUS_COLUMNS,
                         encoding="latin-1")[CENSUS_COLUMNS]          # Census files are Latin-1 (e.g. Doña Ana)
    else:
        raise ValueError(name)
    df = df.copy()                       # de-fragment before adding a column
    df.insert(0, "src_row", range(len(df)))
    return df, extra


WONDER_ANNUAL_COLUMNS = ["Year", "Year Code", "Deaths", "Population", "Crude Rate", "Age Adjusted Rate",
                         "Age Adjusted Rate Lower 95% Confidence Interval", "Age Adjusted Rate Upper 95% Confidence Interval"]


def read_wonder(raw_dir):
    """All CDC WONDER exports in <raw_dir>/wonder/ (manual downloads; see README). `series` = file name
    without extension. Each file is classified by its layout:
      * monthly exports (a 'Month Code' column) -> rows Month, Month Code, Deaths
      * annual exports  (a 'Year Code' column)  -> rows Year, Year Code, Deaths, Population, rates
    The 'Total' row and the footer are dropped. Labels carry '(provisional)' / '(partial)' flags.
    The 'Query Parameters' lines are kept so each series records what it filtered.
    Returns (monthly DataFrame, annual DataFrame, query-parameter DataFrame, files)."""
    files = sorted(glob.glob(os.path.join(raw_dir, "wonder", "*.csv")) + glob.glob(os.path.join(raw_dir, "wonder", "*.txt")))
    if not files:
        raise FileNotFoundError(f"no CDC WONDER exports in {raw_dir}/wonder/ (manual export; see README)")
    monthly, annual, params = [], [], []
    for f in files:
        series = os.path.splitext(os.path.basename(f))[0]
        sep = "\t" if f.endswith(".txt") else ","
        raw = pd.read_csv(f, sep=sep, dtype=str, keep_default_na=False)
        if "Month Code" in raw.columns:
            rows = raw[raw["Month Code"].str.match(r"^\d{4}/\d{2}$", na=False)][["Month", "Month Code", "Deaths"]].copy()
            rows.insert(0, "series", series)
            monthly.append(rows)
            layout = "monthly"
        elif "Year Code" in raw.columns:
            rows = raw[raw["Year Code"].str.match(r"^\d{4}$", na=False)].copy()
            for c in WONDER_ANNUAL_COLUMNS:
                if c not in rows.columns:
                    rows[c] = ""                          # e.g. no age-adjusted rate requested
            rows = rows[WONDER_ANNUAL_COLUMNS]
            rows.insert(0, "series", series)
            annual.append(rows)
            layout = "annual"
        else:
            raise ValueError(f"{f}: neither 'Month Code' nor 'Year Code' column; group the WONDER query by Month or Year")
        text = open(f, encoding="utf-8", errors="replace").read()
        q = text.split("Query Parameters:")[1].split("---")[0] if "Query Parameters:" in text else ""
        lines = [x.strip().strip('"').strip() for x in q.splitlines()]
        params.append(dict(series=series, query_parameters=" | ".join(x for x in lines if x),
                           file=os.path.basename(f), layout=layout))
    m = pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(columns=["series", "Month", "Month Code", "Deaths"], dtype=str)
    a = (pd.concat(annual, ignore_index=True) if annual
         else pd.DataFrame(columns=["series", *WONDER_ANNUAL_COLUMNS], dtype=str))
    return m, a, pd.DataFrame(params), files


# ---------------------------------------------------------------------------------------------- TEDS-A
# (file, admission years kept). The combined 2006-2023 file and the 2024 single-year file have identical
# codes and labels for every variable used (src/teds_codebooks.py; the 2024 state list omits the two states
# excluded in 2024), so rows are appended without recoding; source_file records the origin of each row.
TEDS_FILES = [("teds_a_2006_2023_csv.zip", (2015, 2023)), ("teds_a_2024_csv.zip", (2024, 2024))]
TEDS_COLUMNS = ["CASEID", "ADMYR", "STFIPS", "SUB1", "SUB2", "SUB3", "HERFLG", "METHFLG", "OPSYNFLG", "AGE", "SEX",
                "RACE", "ETHNIC", "NOPRIOR", "METHUSE", "SERVICES", "PSOURCE", "ROUTE1", "IDU"]
TEDS_YEARS = (2015, 2024)          # analysis window; the full raw files stay in data/raw/
CHUNK = 1_000_000


def ingest_teds(raw_dir, out_dir, manifest):
    """TEDS-A CSVs read from their zips in chunks (never unzipped to disk): the combined 2006-2023 file
    (years 2015-2023) and the 2024 single-year file. Keeps TEDS_COLUMNS; every chunk is pandera-validated
    before it is appended to data/interim/teds_a.parquet. src_row = 0-based row number within its raw CSV."""
    out = os.path.join(out_dir, "teds_a.parquet")
    writer, files_info, n_kept_total = None, {}, 0
    for fname, years in TEDS_FILES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing (run fetch_data.py)")
        n_in = n_kept = offset = 0
        with zipfile.ZipFile(path) as zf:
            member = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if len(member) != 1:
                raise ValueError(f"{path}: expected one CSV inside, found {member}")
            with zf.open(member[0]) as fh:
                header = pd.read_csv(fh, nrows=0).columns
            actual = {c.upper(): c for c in header}
            missing = [c for c in TEDS_COLUMNS if c not in actual]
            if missing:
                raise KeyError(f"{path}: columns not found: {missing}")
            with zf.open(member[0]) as fh:
                for chunk in pd.read_csv(fh, usecols=[actual[c] for c in TEDS_COLUMNS], chunksize=CHUNK):
                    chunk.columns = [c.upper() for c in chunk.columns]
                    chunk = chunk[TEDS_COLUMNS]
                    chunk.insert(0, "src_row", range(offset, offset + len(chunk)))
                    chunk.insert(1, "source_file", fname)
                    offset += len(chunk)
                    n_in += len(chunk)
                    keep = chunk[chunk.ADMYR.between(*years)]
                    if keep.empty:
                        continue
                    schemas.validate("teds_a", keep, path)
                    table = pa_arrow.Table.from_pandas(keep, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out, table.schema)
                    writer.write_table(table)
                    n_kept += len(keep)
        if n_kept == 0:
            raise ValueError(f"{path}: no admissions in {years}")
        files_info[fname] = dict(raw_sha256=sha256(path), raw_rows=n_in, rows_kept=n_kept, years=list(years),
                                 csv_member=member[0])
        n_kept_total += n_kept
        print(f"  {'teds_a <- ' + fname:40} {n_kept:>10,} rows ({n_in:,} in raw)")
    writer.close()
    manifest["teds_a"] = dict(file="teds_a.parquet", rows=n_kept_total, columns=len(TEDS_COLUMNS) + 2,
                              sha256=sha256(out), years=list(TEDS_YEARS), raw_files=files_info)
    print(f"  {'teds_a':22} {n_kept_total:>8,} rows -> {out}")


CENSUS_ASR = {"census_state_asr_2010_2020": ("census_sc_est2020int_alldata5.csv", range(2010, 2020)),
              "census_state_asr_2020_2025": ("census_sc_est2025_alldata5.csv", range(2020, 2026))}


def ingest_census_asr(raw_dir, out_dir, manifest):
    """Census state population by single year of age (intercensal 2010-2019; Vintage 2025 2020-2025).
    Keeps state rows (SUMLEV 40) for both sexes (SEX 0) and all origins (ORIGIN 0); RACE is kept (summed in dbt)."""
    frames = []
    for table, (fname, years) in CENSUS_ASR.items():
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing (run fetch_data.py --only {table})")
        cols = ["SUMLEV", "REGION", "STATE", "NAME", "SEX", "ORIGIN", "RACE", "AGE", *[f"POPESTIMATE{y}" for y in years]]
        d = pd.read_csv(path, dtype={"STATE": str, "SUMLEV": str}, usecols=cols, encoding="latin-1")[cols]
        d = d[(d.SUMLEV == "040") & (d.SEX == 0) & (d.ORIGIN == 0)]
        long = d.melt(id_vars=["REGION", "STATE", "NAME", "RACE", "AGE"], value_vars=[f"POPESTIMATE{y}" for y in years],
                      var_name="year", value_name="population")
        long["year"] = long.year.str[-4:].astype(int)
        long["vintage"] = "intercensal 2010-2020" if "2010" in table else "Vintage 2025"
        long["file"] = fname
        frames.append(long)
        manifest.setdefault("census_state_age_raw", {})[fname] = sha256(path)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "src_row", range(len(df)))
    schemas.validate("census_state_age", df, os.path.join(raw_dir, "census_sc_est*_alldata5.csv"))
    write(df, out_dir, "census_state_age", os.path.join(raw_dir, CENSUS_ASR["census_state_asr_2020_2025"][0]), manifest)


def write(df, out_dir, table, raw_path, manifest):
    p = os.path.join(out_dir, f"{table}.parquet")
    df.to_parquet(p, index=False)
    manifest[table] = dict(file=os.path.basename(p), rows=len(df), columns=len(df.columns),
                           sha256=sha256(p), raw_source=os.path.relpath(raw_path),
                           raw_sha256=sha256(raw_path))
    print(f"  {table:22} {len(df):>8,} rows  {len(df.columns):>3} cols -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/interim")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = {}
    for table, rel in SOURCES.items():
        raw_path = os.path.join(a.raw_dir, rel)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"{raw_path} missing (run fetch_data.py and unzip the NSDUH bundles)")
        df, extra = read_source(table, raw_path)
        schemas.validate(table, df, raw_path)
        write(df, a.out_dir, table, raw_path, manifest)
        for t, edf in extra.items():
            schemas.validate(t, edf, raw_path)
            write(edf, a.out_dir, t, raw_path, manifest)
    wonder, wonder_annual, params, files = read_wonder(a.raw_dir)
    wonder = wonder.copy()
    wonder.insert(0, "src_row", range(len(wonder)))
    schemas.validate("wonder_monthly", wonder, os.path.join(a.raw_dir, "wonder"))
    write(wonder, a.out_dir, "wonder_monthly", files[0], manifest)
    wonder_annual = wonder_annual.copy()
    wonder_annual.insert(0, "src_row", range(len(wonder_annual)))
    schemas.validate("wonder_annual", wonder_annual, os.path.join(a.raw_dir, "wonder"))
    write(wonder_annual, a.out_dir, "wonder_annual", files[0], manifest)   # may be empty until an annual export exists
    write(params, a.out_dir, "wonder_query_parameters", files[0], manifest)
    manifest["wonder_monthly"]["raw_files"] = {os.path.basename(f): sha256(f) for f in files}
    ingest_teds(a.raw_dir, a.out_dir, manifest)
    ingest_census_asr(a.raw_dir, a.out_dir, manifest)
    with open(os.path.join(a.out_dir, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {os.path.join(a.out_dir, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
