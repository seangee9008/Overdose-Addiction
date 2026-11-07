-- State FIPS -> Census region (1 Northeast, 2 Midwest, 3 South, 4 West), from the Census state files.
select distinct "STATE" as state_fips, "NAME" as state_name, "REGION" as region,
       case "REGION" when 1 then 'Northeast' when 2 then 'Midwest' when 3 then 'South' when 4 then 'West' end as region_name
from {{ source('interim', 'census_state_age') }}
