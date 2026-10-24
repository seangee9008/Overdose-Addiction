-- One row per county x month. Two Virginia FIPS (51005, 51019) appear twice per month in the raw
-- file (independent cities folded into their counties) with identical values (tested in
-- tests/va_duplicate_rows_agree.sql); the first raw row is kept.
select
    src_row, fips, st_abbrev, county_name, year, month, month_ending_date, nchs_urban_rural_2013,
    deaths_reported, is_suppressed, deaths_lower, deaths_upper, pct_pending_investigation,
    fips || '|' || cast(month_ending_date as varchar) as row_key
from {{ ref('stg_cdc_county_all') }}
where dup_rank = 1
