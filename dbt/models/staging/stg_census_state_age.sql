-- Census resident population by state x year x single year of age (summed over race; both sexes,
-- all Hispanic origins). Two vintages: intercensal 2010-2020 (years 2010-2019) and Vintage 2025 (2020-2025).
select "STATE" as state_fips, "NAME" as state_name, year, "AGE" as age, vintage, sum(population) as population
from {{ source('interim', 'census_state_age') }}
group by all
