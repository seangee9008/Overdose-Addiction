-- The raw county file repeats two Virginia FIPS each month; deduplication is only safe if the
-- repeated rows carry identical values.
select fips, year, month
from {{ ref('stg_cdc_county_all') }}
group by fips, year, month
having count(*) > 1
   and (count(distinct coalesce(deaths_reported, -1)) > 1 or count(distinct is_suppressed) > 1)
