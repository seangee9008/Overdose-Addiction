-- One row per state x year in the 2015-2023 window (50 states + DC; Puerto Rico has no matching
-- population/death data here). Completeness rule (set by the project owner before analysis):
--   complete = admissions reported in EVERY year 2015-2023 AND total admissions never change by more
--   than 50% from one year to the next (ratio outside [0.5, 1.5]).
with states as (
    select distinct state_fips, state_name from {{ ref('stg_census_state_age') }}
),
years as (select range as year from range(2015, 2024)),
grid as (
    select s.state_fips, s.state_name, y.year,
           coalesce(t.total_admissions, 0)                                   as total_admissions,
           t.total_admissions is not null                                    as reports,
           e.state_name is not null                                          as codebook_excluded
    from states s cross join years y
    left join {{ ref('teds_state_year') }} t on t.state_fips = s.state_fips and t.year = y.year
    left join {{ ref('teds_state_exclusions') }} e on e.state_name = s.state_name and e.year = y.year
),
yoy as (
    select *, total_admissions * 1.0
              / nullif(lag(total_admissions) over (partition by state_fips order by year), 0) as yoy_ratio
    from grid
),
per_state as (
    select state_fips,
           bool_and(reports)                                                     as reports_every_year,
           coalesce(bool_and(yoy_ratio between 0.5 and 1.5 or yoy_ratio is null), false) as yoy_within_50pct,
           max(abs(yoy_ratio - 1))                                               as max_abs_yoy_change,
           string_agg(case when not reports then cast(year as varchar) end, ', ' order by year) as years_not_reported,
           string_agg(case when yoy_ratio not between 0.5 and 1.5 then cast(year as varchar) || ' (x'
                      || cast(round(yoy_ratio, 2) as varchar) || ')' end, ', ' order by year) as years_over_50pct
    from yoy group by state_fips
)
select y.*, p.reports_every_year, p.yoy_within_50pct, p.max_abs_yoy_change,
       p.reports_every_year and p.yoy_within_50pct                               as is_complete,
       case when not p.reports_every_year then 'not reported in: ' || p.years_not_reported
            when not p.yoy_within_50pct then 'year-to-year change > 50%: ' || p.years_over_50pct
            else null end                                                        as exclusion_reason,
       y.state_fips || '|' || cast(y.year as varchar)                            as row_key
from yoy y join per_state p using (state_fips)
