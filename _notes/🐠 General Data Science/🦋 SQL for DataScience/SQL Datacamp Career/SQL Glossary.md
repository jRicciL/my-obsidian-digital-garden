# SQL Glossary

## Basic Query
- [[C1 - Introduction to SQL Server#Querying 101]]
	- `SELECT`
	- `TOP`
	- `PERCENT`
	- `DISTINCT`
	- `AS`

***
## DataTypes
- [[SQL - CRUD basics#Datatypes]]
- [[1. Chossing the appropiate data type]]
	- Data types
	- Data Conversion
	- Data type precedence

***
## Ordering and filtering
- [[C1 - Introduction to SQL Server#Ordering and Filtering]]
	- `ORDER BY`
	- `DESC` and `ASC`
	- `WHERE` - [[C1 - Introduction to SQL Server#WHERE]]
		- `AND` and `OR`
		- `IN`
		- `LIKE` -> Wildcards

```sql
SELECT 
	company,
	bean_type,
	broad_bean_origin,
	'The company ' +  company + ' uses beans of type "' + bean_type + '", originating from ' + broad_bean_origin + '.'
FROM ratings
WHERE
    -- The 'broad_bean_origin' should not be unknown
	broad_bean_origin NOT LIKE '%unknown%';

```

### Comparing values
- `=` -> equality
- `<>` -> inequality
- `BETWEEN` -> between a range

***
## Aggregation data

### Aggregation functions
- [[SQL - Aggregating data#Aggregating data]]
- [[1. Summarizing Data]]
	- `SUM`
	- `COUNT` -> combined with `DINSTINCT`
		- `COUNT(*)`
		- `COUNT_BIG()`
		- `COUNT(DISTINCT)`
	- `MIN`
	- `MAX`
	- For stings => [[SQL - Aggregating data#Strings]]:
		- `LEN`
		- `LEFT` -> Extract number of characters
		- `RIGTH`
		- `CHARINDEX`
		- `SUBSTRING`
		- `REPLACE`
	- Statistics => 
		- [[1. Summarizing Data#More math functions]]
		- [[6. Aggregation functions for time#Statistical Aggregate functions]]
			- `AVG()`
			- `STDEV`
			- `STDEVP`
			- `VAR`
			- `VARP`
			- Median:
				- `PERCENTILE_COUNT` => [[6. Aggregation functions for time]]

### Counts and totals
- [[1. Summarizing Data#Counts and Totals SUM and COUNT]]

### Aggregation for time series
- [[6. Aggregation functions for time]]
- [[SQL - Aggregating data]]

### Sampling and upsampling data
- [[6. Aggregation functions for time#Sownsampling and upsampling data]]

### Grouping by `ROLLUP`, `CUBE`
- [[6. Aggregation functions for time#Grouping by ROLLUP CUBE and GRUPING SETS]]
	- `ROLLUP` => For summary hierarchical data
	- `CUBE` => Cartesian aggregation of the columns
	- `GROUPING SETS` 

***
## Dealing with missing data
- [[1. Summarizing Data#Dealing with missing data]]
	- `NULL`
	- `IS NOT NULL`
	- `ISNULL`
	- `COALESCE` -> [[1. Summarizing Data#COALESCE]]

***
## Grouping and Having
- [[SQL - Aggregating data#Grouping and Having]]
	- `GROUP BY`
	- `HAVING`

***
## CRUD operations
- [[SQL - CRUD basics#CRUD operations]]
	- `CREATE TABLE`
	- `INSERT INTO`
		- `VALUES`
		- `SELECT`
	- `UPDATE`
		- `SET`
	- `DELETE`
	- `TRUNCATE`

### Variables
- [[SQL - CRUD basics#Use variables]]
	- `DECLARE`
		- `SET`

***
## Table operations

### Temporary tables
- [[SQL - CRUD basics#Temporary tables]]
	-	`SELECT`
		-	`INTO #my_temp_table`
		
### Derived tables
- [[2. Complex Queries#Derived Tables]]

### Common table Expressions
#CommonTableExpression 
- [[2. Complex Queries#Common Table Expressions]]
	- `WITH`

### Rounding values
- [[1. Summarizing Data#Rounding and Truncating numbers]]


***
## Working with dates
- [[1. Dates and times]]
- [[2. Functions for date and time]]
	- `IsDate`
	- `DateFormat`
	- `Set Language {language}`
- [[1. Summarizing Data#Math Dates DATEPART]]
	- `DATEPART`
	- `DATENAME`
	- `DATEADD`
	- `DATEDIFF`


### Formatting dates
- [[2. Formatting dates for reporting]]
	- `CAST`
	- `CONVERT`
	- `FORMAT`

### Building dates
- [[4. Building Dates]]
	- `DateFromParts`
	- `TimeFromParts`
	- `DateTimeFromParts`
	- `DateTimeOffsetFromParts`
	- Dates from **Strings**
		- `CAST()`
		- `CONVERT()`
		- `PARSE()`

### Working with time zones
- [[4. Building Dates#Working with offsets TimeZones]]
- [[2. Functions for date and time#Time zones in SQL Server]]
	- `DateTimeOffset`
	- `SwitchOffset`
	- `ToDateTimeOffset`

### System date
- [[2. Functions for date and time#Time zones in SQL Server]]
	- `SysDateTime`
	- `SysUtcDateTime`
	- `SysDateTimeOffset`
	- `GetDate`
	- `GetUtcDate`
	- `Current_timeStamp`

### Handling invalid dates
- [[5. Handling invalid dates]]
	- `TRY_CAST`
	- `TRY_CONVERT`
	- `TRY_PARSE`

### Calendar tables
- [[3. Calendar tables]]
	- Building calendar tables

### Aggregation functions for time series
- [[SQL Glossary#Aggregation for time series]]

***
## Joining tables
- [[SQL - Joining tables]]
	- Primary keys
	- Foreing key

### Inner Join
- [[SQL - Joining tables#INNER JOIN]]

### Left and right join
- [[SQL - Joining tables#LEFT and RIGHT JOIN]]

### Union and Union all
- [[SQL - Joining tables#UNION UNION ALL]]

### Except and Intercept

- `EXCEPT` returns distinct rows from the left input query that aren't output by the right input query.
```sql
SELECT
    t.year,
    t.avg_critic_score
FROM top_critic_years as t
EXCEPT
SELECT f.year, f.avg_critic_score
FROM top_critic_years_more_than_four_games as f
ORDER BY avg_critic_score DESC;
```

- `INTERSECT` returns distinct rows that are output by both the left and right input queries operator.


***
## CASE statements
- [[1. CASE statement]]
- [[1. Summarizing Data#Binning Data with CASE]]
- [[6. Aggregation functions for time#Filtering aggregates with CASE]]
- Contains a `WHEN`, `THEN`, and `ELSE` statement, finished with `END`

***
## Flow

### While loops
- [[2. Complex Queries#WHILE loops]]
	- `WHILE`
		- `IF`
		- `CONTINUE`
		- `BREAK`

***
## Windowing functions
- [[3. Window Fucntions]]
	- `FIRST_VALUE()`
	- `LAST_VALUE`
	- `LEAD` and `LAG`
		- [[3. Window Fucntions#Getting the next and prev values with LEAD and LAG]]
		- [[7. Window agg functions#Working with LAG and LEAD]]
	- `ROW_NUMBER`
		- [[3. Window Fucntions#Adding row numbers]]

- [[7. Window agg functions]]
	- `RANK_NUMBER()`
	- `RANK()`
	- `DENSE_RANK()`
	- `PARITITON BY` 
	- Calculating Running totals -> [[7. Window agg functions#Calculating running totals]]
	- Finding maximum level of overlap ->
		- [[7. Window agg functions#Finding maximum levels of overlap]]