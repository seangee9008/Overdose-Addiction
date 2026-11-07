-- first-time + prior + missing = primary-opioid admissions in every cell; cells sum to teds_state_year.
select state_fips, year, age_group from {{ ref('teds_state_year_age') }}
where primary_opioid_first_time + primary_opioid_prior + primary_opioid_noprior_missing <> primary_opioid
union all
select a.state_fips, a.year, -1 from (
    select state_fips, year, sum(primary_opioid) as po from {{ ref('teds_state_year_age') }} group by 1, 2) a
join {{ ref('teds_state_year') }} t using (state_fips, year)
where a.po <> t.primary_opioid
