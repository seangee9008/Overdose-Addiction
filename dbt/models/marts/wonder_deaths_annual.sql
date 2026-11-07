-- CDC WONDER annual deaths with population and crude / age-adjusted rates, one row per series x year,
-- plus each export's query parameters. Empty until an annual export is placed in data/raw/wonder/.
select a.series, a.year, a.deaths, a.population, a.crude_rate, a.age_adjusted_rate, a.aa_rate_ci_low,
       a.aa_rate_ci_high, a.is_provisional, a.aa_rate_unreliable, q.query_parameters, a.src_row, a.row_key
from {{ ref('stg_wonder_annual') }} a
left join {{ source('interim', 'wonder_query_parameters') }} q using (series)
