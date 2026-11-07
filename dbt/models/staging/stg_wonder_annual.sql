-- CDC WONDER annual exports (Group By Year): deaths, population, crude and age-adjusted rates per 100k.
-- Rates are text in the export ('12.3', '12.3 (Unreliable)', 'Not Applicable'); non-numeric -> NULL.
select
    src_row,
    series,
    cast("Year Code" as integer)                                               as year,
    cast("Deaths" as bigint)                                                   as deaths,
    try_cast("Population" as bigint)                                           as population,
    try_cast(split_part("Crude Rate", ' ', 1) as double)                       as crude_rate,
    try_cast(split_part("Age Adjusted Rate", ' ', 1) as double)                as age_adjusted_rate,
    try_cast(split_part("Age Adjusted Rate Lower 95% Confidence Interval", ' ', 1) as double) as aa_rate_ci_low,
    try_cast(split_part("Age Adjusted Rate Upper 95% Confidence Interval", ' ', 1) as double) as aa_rate_ci_high,
    "Year" ilike '%provisional%'                                               as is_provisional,
    "Age Adjusted Rate" ilike '%unreliable%'                                   as aa_rate_unreliable,
    series || '|' || "Year Code"                                               as row_key
from {{ source('interim', 'wonder_annual') }}
