{% test in_range(model, column_name, min_value, max_value) %}
-- fails for every non-NULL value outside [min_value, max_value]
select * from {{ model }}
where {{ column_name }} is not null
  and ({{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }})
{% endtest %}
