-- Custom test: a suppressed county cell must never be coded as 0 (or any number). It must have
-- deaths_reported NULL and the interval [1, 9]. Checked in staging AND in the analysis mart.
select 'stg_cdc_county' as model, row_key from {{ ref('stg_cdc_county') }}
where is_suppressed and (deaths_reported is not null or deaths_lower <> 1 or deaths_upper <> 9)
union all
select 'county_deaths_svi', row_key from {{ ref('county_deaths_svi') }}
where is_suppressed and (deaths_reported is not null or deaths_lower <> 1 or deaths_upper <> 9)
union all
-- and the converse: a non-suppressed cell is exact (lower = upper = reported)
select 'county_deaths_svi (exact)', row_key from {{ ref('county_deaths_svi') }}
where not is_suppressed and (deaths_reported is null or deaths_lower <> deaths_reported
                             or deaths_upper <> deaths_reported)
