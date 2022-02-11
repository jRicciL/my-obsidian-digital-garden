---
---

# Introduction to SQL server

## SQL Server
- Most of the world's data lives in databases
- SQL => Structured Query Language
- SQL Server =>
	- Relational database system developed by Microsoft
- Transact-SQL -> Microsoft's implementation of SQL -> with additional functionality

### Querying 101
> Queries return specific ==sets== of data

- `SELECT` the key term for retrieving data

```sql
SELECT
	artist_id,
	artist_name
FROM 
	artist;
```

- Select all rows and columns
	- However, it is always better to explicit select the row and column we are interested in
```sql
SELECT *
FROM table_name;
```

- `TOP()` <- number of rows to be selected

```sql
SELECT TOP(5) artist
FROM artists;
```

- `PERCENT` return a percentage number of elements
```sql
SELECT TOP(10) PERCENT artist
FROM artists;
```

- Select ==unique values== => `DISTINCT`
```sql
SELECT DISTINCT artists
FROM artists;
```

- ==Aliasing== column names with `AS`
```sql
SELECT demand_loss AS lost_demand
FROM grid;
```

## Ordering and Filtering

- `ORDER BY`: Ascending order
	- `DESC` => Descending order

```sql
SELECT TOP(10) prod_id, year_intro
FROM products
-- Ordering text and numbers
ORDER BY year_intro DESC, product_id;
```

- `WHERE`: Used to return ==rows== that met certain criteria

```sql
SELECT customer_id, total
FROM invoice
-- Return only those rows where total is greater than 15
WHERE total > 15
```

- Using multiple conditions with `AND` and `OR`
	- We need to nest common conditions
```sql
SELECT song, artist
FROM songlist
WHERE
	artist = 'AC/DC'
	-- We need to nest the queries
	AND (
		release_year < 1980
		OR release_year > 2000
		);
```

- a longer version will be:
```sql
SELECT song
FROM songlist
-- THis is a longer but clearer version
WHERE
	(
		artist = 'Greenday'
		AND release_year = 1994
	)
	OR
	(
		artist = 'Greenday'
		AND relese_year > 2000
	)
```

- Use `IN` for multiple selections

```sql
SELECT song, release_year
FROM songlist
WHERE
	release_year IN (1985, 1991, 1992)
ORDER BY
	song
```

- Use `LIKE` and wildcards
```sql
SELECT song
FROM songlist
-- Return all songs begining with 'a'
WHERE song LIKE 'a%';
```


### Comparing values
- To check equality for character data: 
```sql
SELECT country
FROM countries
WHERE country = 'Mexico' 
```

- For date types
```sql
WHERE event_date = '2022-03-07'
```

- Check for non equality: `<>`
```sql
SELECT cutomer_id, total
FROM invoice
-- Testing for non-equality
WHERE total <> 10;
```

- Use `BETWEEN` to return values between a range (inclusive)
```sql
SELECT customer_id, total
FROM invoice
WHERE total BETWEEN 20 AND 30;
```

- Use `NOT` to negate a *where* statement
```sql
SELECT customer_id, total
FROM invoice
WHERE total NOT BETWEEN 20 AND 30;
```

### `NULL` value
- `NULL` indicates there is no value for that record
- Help highlight gaps in our data
	- `unknown` or `missing` values
- **Finding `NULL` values**:

```sql
SELECT
	TOP(6) total,
	billing_state
FROM invoice
WHERE billing_state IS NOT NULL
```

### Examples
- A query example

```sql
-- Select description, affected_customers and event date
SELECT 
  description, 
  affected_customers,
  event_date
FROM 
  grid 
  -- The affected_customers column should be >= 50000 and <=150000   
WHERE 
  affected_customers BETWEEN 50000
  AND 150000 
   -- Define the order   
ORDER BY 
  event_date DESC;
```