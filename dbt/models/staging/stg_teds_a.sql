-- TEDS-A admissions (one row = one treatment ADMISSION, not a person), 2015-2023. Codebook value -9
-- ("missing/unknown/not collected/invalid") -> NULL. AGE has no -9 code. Individual records stay in
-- staging: marts aggregate to state x year (area level) and never join NSDUH.
select
    src_row,
    "CASEID"                                              as caseid,
    "ADMYR"                                               as year,
    lpad(cast("STFIPS" as varchar), 2, '0')               as state_fips,
    {% for v in ['SUB1', 'SUB2', 'SUB3', 'SEX', 'RACE', 'ETHNIC', 'NOPRIOR', 'METHUSE', 'SERVICES', 'PSOURCE', 'ROUTE1', 'IDU'] %}
    case when "{{ v }}" = -9 then null else "{{ v }}" end as {{ v | lower }},
    {% endfor %}
    "HERFLG" as herflg, "METHFLG" as methflg, "OPSYNFLG" as opsynflg,
    "AGE"                                                 as age_group
from {{ source('interim', 'teds_a') }}
