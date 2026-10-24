-- State (incl. US, YC = NYC, DC, PR) x indicator x month: the COMPLETE grid of every state,
-- indicator and month that appears anywhere in the file. Nothing is imputed:
--   is_row_missing        = the file has no row for this cell (e.g. a state without drug-specific data)
--   is_data_value_missing = the row exists but CDC left the value blank (suppressed / low quality,
--                           e.g. Florida's drug-specific months)
-- Values are 12-month-ending totals (deaths in the 12 months ending in that month).
with grid as (
    select s.state, i.indicator, m.year, m.month, m.month_date
    from (select distinct state from {{ ref('stg_cdc_state') }}) s
    cross join (select distinct indicator from {{ ref('stg_cdc_state') }}) i
    cross join (select distinct year, month, month_date from {{ ref('stg_cdc_state') }}) m
)
select
    g.state,
    max(c.state_name) over (partition by g.state)                 as state_name,
    g.indicator, g.year, g.month, g.month_date,
    c.data_value, c.predicted_value, c.percent_complete, c.percent_pending_investigation, c.footnote,
    c.row_key is null                                             as is_row_missing,
    c.data_value is null                                          as is_data_value_missing,
    c.predicted_value is null                                     as is_predicted_value_missing,
    c.src_row,
    g.state || '|' || g.indicator || '|' || cast(g.month_date as varchar) as row_key
from grid g
left join {{ ref('stg_cdc_state') }} c
  on c.state = g.state and c.indicator = g.indicator and c.month_date = g.month_date
