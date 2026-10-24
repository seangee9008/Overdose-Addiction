-- SVI 2022 county: the columns the pipeline uses. SVI's -999 missing code -> NULL.
-- (Add further SVI covariates here when a later phase needs them.)
select
    src_row,
    "FIPS"                                                   as fips,
    "ST_ABBR"                                                as st_abbr,
    "LOCATION"                                               as location,
    "E_TOTPOP"                                               as e_totpop,
    case when "RPL_THEMES" = -999 then null else "RPL_THEMES" end as rpl_themes
from {{ source('interim', 'svi_county') }}
