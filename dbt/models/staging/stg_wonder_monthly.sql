-- CDC WONDER monthly death counts (true single-month counts, not rolling totals).
-- WONDER labels: final data through 2024; '(provisional)' from 2025; '(provisional and partial)'
-- for the most recent months, which are known to be incomplete.
select
    src_row,
    series,
    cast(strptime("Month Code", '%Y/%m') as date)          as month_date,
    cast(substr("Month Code", 1, 4) as integer)            as year,
    cast(substr("Month Code", 6, 2) as integer)            as month,
    cast("Deaths" as bigint)                               as deaths,
    "Month" ilike '%provisional%'                          as is_provisional,
    "Month" ilike '%partial%'                              as is_partial,
    series || '|' || "Month Code"                          as row_key
from {{ source('interim', 'wonder_monthly') }}
