-- Aggregation loses nothing: the state x year totals sum to the number of staged admissions, and the
-- primary-opioid parts add up to primary_opioid.
select 'total' as check_name from (select (select sum(total_admissions) from {{ ref('teds_state_year') }})
                                          - (select count(*) from {{ ref('stg_teds_a') }}) as diff) where diff <> 0
union all
select 'opioid parts' from {{ ref('teds_state_year') }}
where primary_heroin + primary_methadone_nonrx + primary_other_opiates <> primary_opioid
