-- National (State = 'US') monthly 12-month-ending death totals, every indicator.
select * exclude (state_name)
from {{ ref('state_deaths_long') }}
where state = 'US'
