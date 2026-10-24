-- CDC county file, every raw row, typed. FIPS zero-padded to 5 characters.
-- Suppressed cells (NCHS: counts 1-9) are blank in the raw file: they get is_suppressed = true,
-- deaths_reported = NULL (never 0) and the bounds [1, 9]. Zeros are true zeros.
select
    src_row,
    lpad("FIPS", 5, '0')                                            as fips,
    "ST_ABBREV"                                                     as st_abbrev,
    "COUNTYNAME"                                                    as county_name,
    "Year"                                                          as year,
    "Month"                                                         as month,
    cast(strptime("MonthEndingDate", '%m/%d/%Y') as date)           as month_ending_date,
    "CODE2013"                                                      as nchs_urban_rural_2013,
    "Provisional Drug Overdose Deaths"                              as deaths_reported,
    "Provisional Drug Overdose Deaths" is null                      as is_suppressed,
    case when "Provisional Drug Overdose Deaths" is null then 1
         else "Provisional Drug Overdose Deaths" end                as deaths_lower,
    case when "Provisional Drug Overdose Deaths" is null then 9
         else "Provisional Drug Overdose Deaths" end                as deaths_upper,
    "Percentage Of Records Pending Investigation"                   as pct_pending_investigation,
    count(*) over (partition by lpad("FIPS", 5, '0'), "Year", "Month")        as rows_for_fips_month,
    row_number() over (partition by lpad("FIPS", 5, '0'), "Year", "Month" order by src_row) as dup_rank
from {{ source('interim', 'cdc_county') }}
