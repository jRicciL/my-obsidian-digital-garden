# Aggregating data

### `SUM` - single column
- Sum the values of a single column
	- We can use aliases with `AS`

```sql
SELECT
	SUM(affected_customers) AS total_affected
FROM grid;
```

![[Captura de Pantalla 2022-02-06 a la(s) 22.17.22.png]]

### `COUNT` the number of elements

- Count the number of rows

```sql
SELECT
	COUNT(affected_customers) AS count_affected
FROM grid;
```

- Count the number of unique values with `DISTINCT`

```sql
SELECT 
	COUNT(DISTINCT column_name) AS new_name
FROM table_name;
```

### `MIN` and `MAX` values
- Get the minimum value of a column

```sql
SELECT
	MIN(col_name) AS new_value_name
FROM table_name;
```

- Get the maximum value but excluding 0

```sql
SELECT
	MAX(col_name) AS new_value_name
FROM grid
WHERE col_name > 0
```

```sql
SELECT 
  SUM(demand_loss_mw) AS MRO_demand_loss 
FROM 
  grid 
WHERE
  -- demand_loss_mw should not contain NULL values
  demand_loss_mw IS NOT NULL 
  -- and nerc_region should be 'MRO';
  AND nerc_region = 'MRO';
```