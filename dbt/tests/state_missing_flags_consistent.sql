-- Missing flags must agree with the values (no silent imputation in the grid).
select row_key from {{ ref('state_deaths_long') }}
where (is_data_value_missing and data_value is not null)
   or (not is_data_value_missing and data_value is null)
   or (is_row_missing and (data_value is not null or predicted_value is not null))
