# Overdose-Addiction

UCLA Statistics 199 directed research project (faculty sponsor: Prof. Dave Zes).

**Question:** is addiction itself declining in the United States, or only overdose mortality?

This repository holds the data pipeline and exploratory analysis tooling. It currently:

- downloads the raw public datasets with checksums,
- profiles what those files actually contain,
- fits 0/1/2-break piecewise-linear trends to CDC national overdose death series
  (all-drug or any drug-specific indicator), and
- compares national death trends with published NSDUH opioid-misuse percentages, indexed to 2021.

It does not contain conclusions. Results so far are exploratory, and have known caveats.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -r requirements.txt
```

If a conda environment is active, call the project interpreter explicitly (`.venv/bin/python`)
or run `conda deactivate` then `source .venv/bin/activate`.

## Running

```bash
# 1. Download raw data into data/raw/ (git-ignored) and write data/raw/MANIFEST.json
.venv/bin/python fetch_data.py
.venv/bin/python fetch_data.py --list            # list sources
unzip -o data/raw/nsduh_2023_puf_stata.zip -d data/raw/nsduh_2023

# 2. Profile the downloads -> data/DATA_PROFILE.md
.venv/bin/python -m src.profile_data

# 3. Break points + NSDUH comparison for one CDC series
.venv/bin/python analyze_real.py                                                      # all-drug deaths
.venv/bin/python analyze_real.py --indicator "Opioids (T40.0-T40.4,T40.6)" --tag opioids
#   options: --state (default US), --value-col "Data Value"|"Predicted Value", --min-seg (default 12)

# 4. Tests
.venv/bin/python -m pytest
```

`analyze_real.py` writes the extracted series to `data/processed/<state>_<tag>_series.csv` and
charts plus a JSON summary to `outputs/<tag>_*`.

## Data sources

| Source | Level | File in `data/raw/` | URL |
|---|---|---|---|
| CDC/NCHS VSRR Provisional Drug Overdose Death Counts, national + state (12-month-ending, 12 drug indicators, 2015-01 onward) | state × month | `cdc_vsrr_overdose_national_state.csv` | https://data.cdc.gov/api/views/xkb8-kh2a/rows.csv?accessType=DOWNLOAD |
| CDC/NCHS VSRR provisional county overdose deaths (12-month-ending, all-drug only, 2020-01 onward) | county × month | `cdc_vsrr_overdose_county.csv` | https://data.cdc.gov/api/views/gb4e-yj24/rows.csv?accessType=DOWNLOAD |
| SAMHSA NSDUH 2023 public-use file, Stata format (no geography) | individual | `nsduh_2023_puf_stata.zip` | https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-bndl-data-stata_v2.zip |
| NSDUH 2023 codebook | — | `nsduh_2023_codebook.pdf` | https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-info-codebook_v2.pdf |
| CDC/ATSDR Social Vulnerability Index 2022, county | county | `svi_2022_us_county.csv` | https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/SVI_2022_US_county.csv |
| SVI 2022 county polygons (ArcGIS mirror, opt-in: `--only svi_2022_arcgis_geo`) | county | `svi_2022_county.geojson` | ArcGIS Online layer `SVI2022_US_county` |
| NSDUH published past-year opioid misuse %, 2021–2024 | national | hard-coded in `analyze_real.py` | SAMHSA NSDUH annual reports |

See [DATA.md](DATA.md) for the inventory and join keys, and [data/DATA_PROFILE.md](data/DATA_PROFILE.md)
for what the downloaded files actually contain (row counts, indicators, date ranges, join-key issues).

NSDUH (individual-level) and CDC county + SVI (area-level) are analysed separately and never
combined in one model.

## Layout

```
fetch_data.py         analyze_real.py        requirements.txt     pytest.ini
src/                  breaks.py, cdc.py, profile_data.py
tests/                test_breaks.py
notebooks/            exploratory notebooks
data/raw/             downloads (git-ignored)
data/processed/       small derived series
outputs/              charts and summaries
```
