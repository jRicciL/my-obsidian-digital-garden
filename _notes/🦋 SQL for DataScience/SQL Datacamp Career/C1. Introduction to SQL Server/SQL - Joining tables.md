# Joining Tables

## Primary Keys
- ==Primary keys==: Uniquely identify each row in a table
	- integer column
	- each value is different to the others
- ==Foreing key==: Links to other table

## INNER JOIN
- Preserves only the rows that match the two tables

```sql
SELECT
	album_id,
	title,
	album.artist_id,
	name AS artist_name
FROM
	album
INNER JOIN
	artist ON artist.artist_id = album.artist_id
WHERE album.artist_id = 1
```

- We can join more than two tables using Multiple INNER JOINS

```sql
SELECT
	table_A.colX,
	table_A.colY,
	table_B.colZ,
	table_C.colW
FROM
	table_A
INNER JOIN table_B ON table_B.foreing_key = table_A.primary_key
INNER JOIN table_C ON table_C.foreing_key = table_B.primary_key
```

## LEFT and RIGHT JOIN

- We want to preserve all the entries from the left or right table in order to have the full picture of the data and identify those missing values that are not present in one of the tables
- All rows from the main table will appear.

![[Captura de Pantalla 2022-05-02 a la(s) 12.04.26.png]]

```sql
SELECT
	Admitted.Patient_ID,
	Admitted,
	Discharged
FROM
	Admitted
LEFT JOIN
	Discharged ON Discharged.Patitend_ID = Admitted.Patient_ID
```

#### LEFT
![[Captura de Pantalla 2022-05-02 a la(s) 12.05.12.png]]

#### RIGHT
![[Captura de Pantalla 2022-05-02 a la(s) 12.07.58.png]]

#### More Examples

```sql
SELECT 
  invoiceline_id,
  unit_price, 
  quantity,
  billing_state
  -- Specify the source table
FROM invoiceline -- Main table (LEFT)
  -- Complete the join to the invoice table
LEFT JOIN invoice -- Secondary Table
ON invoiceline.invoice_id = invoice.invoice_id;
```

- Another example:
	- `SELECT` the fully qualified column names `album_id` from `album` and `name` from `artist`. Then, join the tables so that only matching rows are returned (non-matches should be discarded).
	- To complete the query, join the `album` table to the `track` table using the relevant fully qualified `album_id` column. The album table is on the left-hand side of the join, and the additional join should return all matches or NULLs.
```sql
SELECT 
  album.album_id,
  title,
  album.artist_id,
  artist.name as artist
FROM album
INNER JOIN artist ON album.artist_id = artist.artist_id
-- Perform the correct join type to return matches or NULLS from the track table
LEFT JOIN track ON album.album_id = track.album_id
WHERE album.album_id IN (213,214)
```

## UNION & UNION ALL
#### UNION
- Combine the results from multiple queries or tables with:
	- The same number of columns, listed in the same order and with similar datatypes
- `UNION` **EXCLUDES** duplicated rows

![[Captura de Pantalla 2022-05-02 a la(s) 12.15.22.png]]

```sql
SELECT
	album_id,
	title,
	artist_id
FROM album
WHERE artist_id IN (1, 3)

UNION

SELECT
	album_id,
	title,
	artist_id
FROM album
WHERE artist_id IN (1, 4, 5);
```

#### UNION ALL
- 🔴  **DOESN'T EXCLUDE DUPLICATES** 

#### Rename columns from different tables
- We can combine rows with different column names if they have the same datatype

```sql
SELECT 
  album_id AS ID,
  title AS description,
  'Album' AS Source
  -- Complete the FROM statement
FROM album
 -- Combine the result set using the relevant keyword
UNION
SELECT 
  artist_id AS ID,
  name AS description,
  'Artist'  AS Source
  -- Complete the FROM statement
FROM artist;
```