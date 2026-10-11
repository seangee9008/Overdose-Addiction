# Data inventory

Everything the project uses, what it's for, and whether it's in hand yet.

| # | Dataset | Level | Gives you | Status | Link |
|---|---|---|---|---|---|
| 1 | CDC VSRR provisional overdose deaths, national + state | state × month | Death trend (12-month-ending totals), by drug indicator | **In hand (national series only)**: `national_deaths_real.json`. Full file via `fetch_data.py` | https://data.cdc.gov/api/views/xkb8-kh2a/rows.csv?accessType=DOWNLOAD |
| 2 | CDC VSRR provisional overdose deaths, county | county × year/period | The "location" layer, FIPS-coded | Not downloaded | https://data.cdc.gov/api/views/gb4e-yj24/rows.csv?accessType=DOWNLOAD |
| 3 | NSDUH 2023 public-use microdata | individual | Use, misuse, depression, treatment, demographics. **No geography** | Not downloaded (large) | https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-bndl-data-sas_v2.zip |
| 3b | NSDUH 2023 codebook | n/a | Variable definitions | Not downloaded | https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-info-codebook_v2.pdf |
| 3c | NSDUH other years (incl. 2024) | individual | More years for trend | Pick files here | https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles |
| 4 | CDC/ATSDR Social Vulnerability Index 2022, county | county | Poverty, unemployment, education, housing, etc. (the SES/deprivation covariates) | Not downloaded; download code untested | Layer: https://services7.arcgis.com/arZnhQhtvIXpgVPD/ArcGIS/rest/services/SVI2022_US_county/FeatureServer · Docs: https://www.atsdr.cdc.gov/place-health/media/pdfs/2024/10/SVI2022Documentation.pdf |
| 5 | NSDUH published opioid-misuse % (2021–24: 3.2, 3.0, ~3.05, 2.7) | national | Quick use-vs-death comparison | Hard-coded in `analyze_real.py`, from SAMHSA annual reports | https://www.samhsa.gov/data/ |
| 6 | FBI Crime Data Explorer | agency/county | Crime covariate (optional) | Needs a free API key; not fetched | https://cde.ucr.cjis.gov |

## Gaps worth closing first (from the demo caveats)

- **Opioid-specific deaths.** Dataset 1 has drug-specific indicators (opioids, synthetic opioids, etc.), so use those instead of the all-drug total before comparing to NSDUH opioid misuse. The all-drug vs opioid mismatch is the biggest hole in the demo.
- **True monthly counts.** Dataset 1 is 12-month-ending. If you want single-month deaths, check CDC's other provisional files or CDC WONDER (manual query export, final data lags ~1 year).
- **Join key.** Datasets 2 and 4 both key on county FIPS; join there. NSDUH cannot be joined to geography. Keep it as a separate individual-level model (don't merge the two levels into one regression).

## Pulling it

```
python fetch_data.py            # all automatic sources into data/raw/ + MANIFEST.json (sha256 per file)
python fetch_data.py --only cdc_state cdc_county
```

If the SVI ArcGIS pull errors, download the 2022 county CSV manually from the ATSDR SVI page and drop it in `data/raw/`.
