-- Calendar-year overdose deaths by state (December 12-month-ending values, CDC VSRR state file).
-- New York = 'NY' (excluding New York City) + 'YC' (New York City). Missing values stay NULL (e.g.
-- Louisiana has no drug-specific rows; some states have suppressed / low-quality drug-specific months).
with d as (
    select case when state = 'YC' then 'NY' else state end as state, year, indicator, data_value, is_row_missing
    from {{ ref('state_deaths_long') }}
    where month = 12 and state not in ('US', 'PR')
      and indicator in ('Opioids (T40.0-T40.4,T40.6)', 'Number of Drug Overdose Deaths')
),
agg as (
    select state, year,
           case when bool_or(indicator = 'Opioids (T40.0-T40.4,T40.6)' and (is_row_missing or data_value is null)) then null
                else sum(data_value) filter (where indicator = 'Opioids (T40.0-T40.4,T40.6)') end as opioid_deaths,
           case when bool_or(indicator = 'Number of Drug Overdose Deaths' and data_value is null) then null
                else sum(data_value) filter (where indicator = 'Number of Drug Overdose Deaths') end as all_drug_deaths
    from d group by state, year
),
names as (select distinct state, state_name from {{ ref('state_deaths_long') }} where state not in ('US', 'YC', 'PR'))
select a.state, n.state_name, a.year, a.opioid_deaths, a.all_drug_deaths,
       a.state || '|' || cast(a.year as varchar) as row_key
from agg a join names n using (state)
