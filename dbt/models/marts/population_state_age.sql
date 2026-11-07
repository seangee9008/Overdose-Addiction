-- State population by year and TEDS-A age group (AGE codes 1-12; group 0 = ages 0-11, not in TEDS),
-- plus all ages. Single years of age 85+ are top-coded at 85 by Census, inside TEDS group 12 (65+).
with p as (
    select *, case
        when age < 12 then 0 when age <= 14 then 1 when age <= 17 then 2 when age <= 20 then 3
        when age <= 24 then 4 when age <= 29 then 5 when age <= 34 then 6 when age <= 39 then 7
        when age <= 44 then 8 when age <= 49 then 9 when age <= 54 then 10 when age <= 64 then 11 else 12 end
        as teds_age_group
    from {{ ref('stg_census_state_age') }}
)
select state_fips, state_name, year, vintage, teds_age_group, sum(population) as population,
       state_fips || '|' || cast(year as varchar) || '|' || cast(teds_age_group as varchar) as row_key
from p group by all
union all
select state_fips, state_name, year, vintage, -1 as teds_age_group, sum(population),
       state_fips || '|' || cast(year as varchar) || '|all'
from p group by all
