-- Monthly death counts from CDC WONDER, one row per series x month, with final / provisional /
-- partial flags and the export's query parameters (so each series documents its own definition).
select w.series, w.year, w.month, w.month_date, w.deaths, w.is_provisional, w.is_partial,
       q.query_parameters, w.src_row, w.row_key
from {{ ref('stg_wonder_monthly') }} w
left join {{ source('interim', 'wonder_query_parameters') }} q using (series)
