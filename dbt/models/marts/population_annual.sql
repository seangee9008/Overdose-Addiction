-- Annual (July 1) resident population by nation / state / county, 2020-2025 (Census Vintage 2025).
-- nation = sum of the 50 states + DC (Census's national total; Puerto Rico is not included).
select level, fips, state_name, area_name, year, population,
       level || '|' || fips || '|' || cast(year as varchar) as row_key
from {{ ref('stg_census_pop') }}
union all
select 'nation', 'US', 'United States', 'United States', year, sum(population),
       'nation|US|' || cast(year as varchar)
from {{ ref('stg_census_pop') }}
where level = 'state'
group by year
