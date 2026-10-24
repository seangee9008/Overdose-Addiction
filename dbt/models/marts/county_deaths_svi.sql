-- County x month overdose deaths (all drugs) joined to SVI 2022. Inner join: counties without an
-- SVI match (Connecticut's 8 legacy counties, Alaska 02261) are listed in county_fips_audit.
-- Suppressed cells: is_suppressed = true, deaths_reported NULL, bounds [1, 9] -- never 0.
select
    c.src_row, c.fips, c.st_abbrev, c.county_name, c.year, c.month, c.month_ending_date,
    c.nchs_urban_rural_2013, c.deaths_reported, c.is_suppressed, c.deaths_lower, c.deaths_upper,
    c.pct_pending_investigation,
    s.location as svi_location, s.e_totpop, s.rpl_themes,
    c.row_key
from {{ ref('stg_cdc_county') }} c
join {{ ref('stg_svi') }} s on s.fips = c.fips
