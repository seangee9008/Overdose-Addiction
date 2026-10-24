select variable, value, label from {{ source('interim', 'nsduh_value_labels') }}
