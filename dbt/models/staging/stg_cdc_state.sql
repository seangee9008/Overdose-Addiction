-- CDC national/state file, typed. One row per state x indicator x month as delivered.
-- Values stay exactly as reported: "Data Value" text -> number after removing thousands commas.
select
    src_row,
    "State"                                                         as state,
    "State Name"                                                    as state_name,
    "Indicator"                                                     as indicator,
    "Period"                                                        as period,
    cast("Year" as integer)                                         as year,
    month(strptime("Year" || '-' || "Month" || '-01', '%Y-%B-%d'))  as month,
    cast(strptime("Year" || '-' || "Month" || '-01', '%Y-%B-%d') as date) as month_date,
    try_cast(replace("Data Value", ',', '') as double)              as data_value,
    try_cast(replace("Predicted Value", ',', '') as double)         as predicted_value,
    try_cast("Percent Complete" as double)                          as percent_complete,
    try_cast("Percent Pending Investigation" as double)             as percent_pending_investigation,
    "Footnote"                                                      as footnote,
    "State" || '|' || "Indicator" || '|' || "Year" || '-' || "Month" as row_key
from {{ source('interim', 'cdc_state') }}
