-- NSDUH 2024 single-year file: 2024 pain-reliever use disorder (comparable with 2021-2023
-- UDYR5PNRANY per the 2024 codebook), keyed by QUESTID2.
select src_row, "QUESTID2", "UDYR5PNRANY2"
from {{ source('interim', 'nsduh_2024') }}
