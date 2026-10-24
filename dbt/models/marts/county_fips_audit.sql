-- Audit trail for the county cleaning: counties lost in the CDC-SVI join (both directions) and the
-- raw rows collapsed as duplicate FIPS. order_key reproduces raw-file order.
with cdc_first as (
    select fips, st_abbrev, county_name, min(src_row) as first_row
    from {{ ref('stg_cdc_county_all') }}
    group by fips, st_abbrev, county_name
)
select 'CDC county file, no SVI match' as side, fips, st_abbrev, county_name as name,
       1 as side_order, first_row as order_key
from cdc_first
where fips not in (select fips from {{ ref('stg_svi') }})
union all
select 'SVI, no CDC match', fips, st_abbr, location, 2, src_row
from {{ ref('stg_svi') }}
where fips not in (select fips from {{ ref('stg_cdc_county_all') }})
union all
select 'Duplicate FIPS rows in CDC county file', fips, st_abbrev, county_name, 3, first_row
from cdc_first
where fips in (select fips from {{ ref('stg_cdc_county_all') }} where rows_for_fips_month > 1)
