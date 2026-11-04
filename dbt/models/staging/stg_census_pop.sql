-- Census July 1 resident population estimates, one row per area x year (2020-2025).
-- level: 'state' (SUMLEV 40) or 'county' (SUMLEV 50); fips = 2-digit state or 5-digit county.
{% set years = range(2020, 2026) %}
{% for y in years %}
select
    src_row,
    case when "SUMLEV" = 40 then 'state' else 'county' end                  as level,
    case when "SUMLEV" = 40 then "STATE" else "STATE" || "COUNTY" end       as fips,
    "STNAME"                                                                as state_name,
    "CTYNAME"                                                               as area_name,
    {{ y }}                                                                 as year,
    "POPESTIMATE{{ y }}"                                                    as population
from {{ source('interim', 'census_pop') }}
{% if not loop.last %}union all{% endif %}
{% endfor %}
