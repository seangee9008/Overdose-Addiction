-- TEDS-A primary-opioid admissions by state x year x TEDS age group (area level), split by prior
-- treatment: NOPRIOR = 0 (first-time), >= 1 (prior episodes), NULL (codebook -9: missing / not collected).
select
    state_fips,
    year,
    age_group,
    count(*)                                                              as total_admissions,
    count(*) filter (where sub1 in (5, 6, 7))                             as primary_opioid,
    count(*) filter (where sub1 in (5, 6, 7) and noprior = 0)             as primary_opioid_first_time,
    count(*) filter (where sub1 in (5, 6, 7) and noprior >= 1)            as primary_opioid_prior,
    count(*) filter (where sub1 in (5, 6, 7) and noprior is null)         as primary_opioid_noprior_missing,
    state_fips || '|' || cast(year as varchar) || '|' || cast(age_group as varchar) as row_key
from {{ ref('stg_teds_a') }}
group by state_fips, year, age_group
