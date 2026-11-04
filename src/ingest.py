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
import hashlib
import json
import os

import pandas as pd
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
    with open(os.path.join(a.out_dir, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {os.path.join(a.out_dir, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
