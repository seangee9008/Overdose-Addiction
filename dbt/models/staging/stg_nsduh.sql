-- NSDUH combined 2021-2024: codebook (upper-case) names; YEAR text -> integer;
-- codebook value -9 ("data not collected for this year") -> NULL. Design variables untouched.
select
    src_row,
    cast("YEAR" as bigint)                                    as "YEAR",
    "QUESTID2", "ANALWT2_C1", "VESTR_C", "VEREP",
    {% for v in ['HERYR', 'PNRNMYR', 'IRPYUD5HER', 'UDYR5PNRANY', 'OPINMYR', 'UD5OPIANY', 'IMFYR',
                 'CATAG3', 'IRSEX', 'NEWRACE2'] %}
    case when "{{ v }}" = -9 then null else "{{ v }}" end     as "{{ v }}"{{ "," if not loop.last }}
    {% endfor %}
from {{ source('interim', 'nsduh_2021_2024') }}
