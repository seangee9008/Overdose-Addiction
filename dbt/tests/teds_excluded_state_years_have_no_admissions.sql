-- State-years the TEDS-A codebook lists as excluded must carry no admissions in the data.
select r.state_name, r.year, r.total_admissions
from {{ ref('teds_reporting_panel') }} r
where r.codebook_excluded and r.total_admissions > 0
