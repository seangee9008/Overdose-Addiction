-- AREA-LEVEL ONLY: TEDS-A state x year admission counts next to CDC state opioid deaths and Census state
-- population, 2015-2023, with the completeness flag. No individual-level data and nothing from NSDUH.
select r.state_fips, r.state_name, r.year, r.reports, r.codebook_excluded, r.is_complete, r.exclusion_reason,
       t.total_admissions, t.primary_opioid, t.primary_heroin, t.primary_methadone_nonrx, t.primary_other_opiates,
       t.any_opioid, t.primary_substance_missing,
       d.state, d.opioid_deaths, d.all_drug_deaths,
       p.population, p.vintage as population_vintage,
       r.state_fips || '|' || cast(r.year as varchar) as row_key
from {{ ref('teds_reporting_panel') }} r
left join {{ ref('teds_state_year') }} t on t.state_fips = r.state_fips and t.year = r.year
left join {{ ref('deaths_state_year') }} d on d.state_name = r.state_name and d.year = r.year
left join {{ ref('population_state_age') }} p on p.state_fips = r.state_fips and p.year = r.year and p.teds_age_group = -1
