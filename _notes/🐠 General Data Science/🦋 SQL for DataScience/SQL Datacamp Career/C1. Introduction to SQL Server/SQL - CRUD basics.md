# Create and Insert data into tables

#CRUD operations

***

## CRUD operations
- **C**REATE:
	- Databases, tables or views
	- Users, permissions, and security groups
- **R**EAD:
	- `SELECT` statements
- **U**PDATE:
	- Amend existing database records
- **D**ELETE:
	- Delete records

### CREATE

```sql
CREATE TABLE test_table(
	test_date date, -- Column and its datatype
	test_name varchar(20),
	test_int int
)
```

```sql
-- Create the table
CREATE TABLE results (
	-- Create track column
	track VARCHAR(200),
    -- Create artist column
	artist VARCHAR(120),
    -- Create album column
	album VARCHAR(160),
	-- Create track_length_mins
	track_length_mins INT,
	);

```

##### Considerations when crating a table
- Table and column names
- Type of the data each column will store
- Size or amount of data stored in the column

![[Captura de Pantalla 2022-05-02 a la(s) 13.20.24.png]]

#### Datatypes
- Dates:
	- date `YYYY-MM-DD`
	- datetime `YYY-MM-DD hh:mm:ss`
	- time
- Numeric:
	- Integer
	- Decimal
	- Float
	- bit (1 = `TRUE`), also accepts `NULL` values
- Strings:
	- `char`
	- `varchar`
	- `nvarchar`

### INSERT

```sql
INSERT INTO 
	table_name -- Table name
	(col1, col2, col3) -- column names
VALUES
	('val1', val2, val3)
```

### INSERT SELECT
- Move values from one table to another

```sql
INSERT INTO table1 (col1, col2, col3)
SELECT
	col1_t2,
	col2_t2,
	col3_t3
FROM table2
WHERE
-- conditions to apply
```

### UPDATE

```sql
UPDATE table
SET 
	column = value,
	column2 = value2
WHERE
-- conditions
```

⚠️ The `WHERE` clause is required to avoid update all the values in the column

### DELETE 🔴 and TRUNCATE
- ⚠️ **TEST** with a `SELECT` statement before use a `DELETE`
- DELETE

```sql
DELETE
FROM table
WHERE
--conditions
```

- TRUNCATE

```sql
TRUNCATE TABLE table
```

## Use variables

### DECLARE and SET

- **DECLARE** Used to create a Variable
	- Access to the variable with `@` and specify its type
- **SET** => Assigns a value to the variable

```sql
-- Declare the variable
DECLARE @test_int INT

-- Set a value
SET @test_int = 5
```

##### Example

```sql
-- Declare your variables
DECLARE @start DATE
DECLARE @stop DATE
DECLARE @affected INT;
-- SET the relevant values for each variable
SET @start = '2014-01-24'
SET @stop  = '2014-07-02'
SET @affected =  5000 ;

SELECT 
  description,
  nerc_region,
  demand_loss_mw,
  affected_customers
FROM 
  grid
-- Specify the date range of the event_date and the value for @affected
WHERE event_date BETWEEN @start AND @stop
AND affected_customers >= @affected;
```


## Temporary tables
Sometimes you might want to 'save' the results of a query so you can do some more work with the data.
- You can do that by creating a temporary table that remains in the database until SQL Server is restarted.
- Use `INTO #tamp_table_name`

```sql
SELECT
	col,
	col2,
	...
	INTO #my_temp_table
FROM my_existing_table
WHERE
-- conditions
```

##### Example