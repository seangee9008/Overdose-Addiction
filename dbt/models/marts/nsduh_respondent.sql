-- One row per NSDUH respondent-year, 2021-2024: design variables, demographics, SAMHSA recodes,
-- and the comparable 2021-2024 outcomes (definitions: src/nsduh.py and the codebooks):
--   OPI_MISUSE = HERYR = 1 or PNRNMYR = 1
--   OPI_OUD    = IRPYUD5HER = 1 or pain-reliever use disorder = 1, where the latter is
--                UDYR5PNRANY for 2021-2023 and UDYR5PNRANY2 (single-year 2024 file) for 2024.
-- Individual-level only: never join this to county/area tables.
select
    n.src_row, n."YEAR", n."QUESTID2", n."ANALWT2_C1", n."VESTR_C", n."VEREP",
    n."HERYR", n."PNRNMYR", n."IRPYUD5HER", n."UDYR5PNRANY", y."UDYR5PNRANY2",
    case when n."YEAR" = 2024 then y."UDYR5PNRANY2" else n."UDYR5PNRANY" end as "PNR_USE_DISORDER",
    n."OPINMYR", n."UD5OPIANY", n."IMFYR",
    n."CATAG3", n."IRSEX", n."NEWRACE2",
    case when n."HERYR" = 1 or n."PNRNMYR" = 1 then 1.0 else 0.0 end           as "OPI_MISUSE",
    case when n."IRPYUD5HER" = 1
           or (case when n."YEAR" = 2024 then y."UDYR5PNRANY2" else n."UDYR5PNRANY" end) = 1
         then 1.0 else 0.0 end                                                   as "OPI_OUD",
    cast(n."YEAR" as varchar) || '|' || cast(n."QUESTID2" as varchar)          as row_key
from {{ ref('stg_nsduh') }} n
left join {{ ref('stg_nsduh_2024') }} y
  on n."YEAR" = 2024 and y."QUESTID2" = n."QUESTID2"
