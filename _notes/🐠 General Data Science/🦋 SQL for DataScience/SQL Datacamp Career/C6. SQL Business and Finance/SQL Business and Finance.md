# Curso 1: Analyzing business data in SQL

[Revenue | SQL](https://campus.datacamp.com/courses/analyzing-business-data-in-sql)

# Content

1. Revenue, cost and profit
2. User-centric metrics
3. Unit economicas and distributions
4. Generating an executive report

# Chapter 1

### Use CTEs to calculate Revenue, Cost, and Profit

- El Revenue viene de la tabla de ventas
    - The money a company makes
- El Costo viene de la tabla de inventarios, además de los costos logísticos
    - The money a company spends
- El Profit
    - Profit = Revenue - Cost
- KPI:
    - A metric with some value that a company use to measure its performance

### Profit

- Profit per user: Identify the best users
- Profit per SKU/product: Identify the most profitable meals
- Profit per month: Tracks profit over time.

## User Centric KPIs

- MEasure perfomance well in B2Cs
- Used by investors to assess revenue and profit startups

### Registrations

- When a suser first signs up for an account on Delivr through its app
- Registrations KPIs ⇒ Counts registrations over time, usually per month.
    - Good at measuring a company’s success in attracting new users
    - Requires a table containing the registration date or the invoice table indicating the first purchase date

**Registrations per month**

```sql
/* Get the number of new buyers per month
*/
with reg_dates as (
-- Get the first user date
	select user_id, 
	min(order_date) as reg_date
from orders group by user_id
)
select 
-- count the distinct users by month
	date_truc(distinct user_id) :: DATE as deliver_month,
	count(distinct user_id) as regs
from reg_date
group by delivr_month
order by delivr_month asc
limit 3
```

### Active Users

- Counts the active users of a company’s appa over a time perido
    - by day ⇒ **DAU** → Daily Active Users
    - by month ⇒ **MAU** → Monthly Active Users
    - by week ⇒ WAU → Weekly Active Users
- **STICKINESS = DAU / MAU Ratio**

[](https://www.geckoboard.com/best-practice/kpi-examples/dau-mau-ratio/#:~:text=What%20is%20DAU%2FMAU%20Ratio,in%20a%20one%20day%20window)[https://www.geckoboard.com/best-practice/kpi-examples/dau-mau-ratio/#:~:text=What](https://www.geckoboard.com/best-practice/kpi-examples/dau-mau-ratio/#:~:text=What) is DAU%2FMAU Ratio,in a one day window.

- Mide que tan seguido los usuarios usan o consumen en promedio
    - Es la proporción de usuarios activos en un mes que interactuaron con el producto en un ventana de tiempo de un día
- How often people engage with your product

**Active Users by month**

```sql
select
	date_trunc('month', order_date) :: date as delivr_month,
	count(distinct user_id) as mau
from orders
group by delivr_month,
order by delivr_month asc
```

## Window functions

- Perform an operation across a set of rows related to the current row

### Registrations running total per month

- A cumulative sum of a variable’s previous values

```sql
-- Registrations running total
with reg_dates as (
-- Get the first user date
	select user_id, 
	min(order_date) as reg_date
from orders group by user_id
),
registrations as (
		select 
	-- count the distinct users by month
		date_truc(distinct user_id) :: DATE as deliver_month,
		count(distinct user_id) as regs
	from reg_dates
	group by delivr_month
)

select 
	delivr_month,
	regs,
	sum(regs) over (order by delivr_month asc) as running_total
from registrations
order by delivr_month asc
```

### Lagged values - Tracking changes over time

- Compare a previous value against a current value.

```sql
-- Monthly Active USERS (current and previous month)
with maus as (
	select 
		date_trunc('month', order_date) :: date as delivr_month,
		count(distinct user_id) as mau
	from orders
	group by delivr_month
)
-- Calculate previous mau
select 
	delivr_month,
	mau,
	coalesce(
		LAG(mau) over (order by delivr_month asc), 1
	) as last_mau
from maus
order by delivr_month

```

### Growth and Deltas

- **Deltas**

```sql
with maus as (...),
maus_lag as (...)

select 
	delivr_month,
	mau,
	mau - last_mau as mau_delta
from maus_lag
order by delivr_month
```

- **Growth rate**
    
    - A percentage that sow the change in a variable over time relative to that variable’s initial value
    
    $$ Growth-rate = \frac{current - previous}{previous} $$
    
- **Month-on-month growth rate**
    

```sql
with maus as (...),
 maus_lag as (...)

select
	delivr_month,
	mau,
	round(
		(mau - prev_mau) :: numeric / prev_mau,
		2
	)
from maus_lag

```

### Retention

- MAU does not show the breakdown of active buyers by tenure
    
- MAU does not distinguish between different patterns of user activity
    
- **Active Users Breakdown:**
    
    - **New Users** ⇒ Users joined in the current month
    - **Retained Users** ⇒ Actives in the previous month, and stayed active this month
    - **Resurrected users ⇒** weren’t active in the previous month, but they returned to activity this month
- **Retention Rate:**
    
    - A percentage measuring how many users who were active in a previous month are still in the current month
    
    $$ RetentionRate = \frac{Current}{Previous} $$
    

— Retention Rate Query

```sql
-- User Activity
with user_activity as (
-- List of unique values of month and user ids
	select distinct 
		month(order_date) as month,
		user_id
	from orders
)

select
	previous.month,
	round(
		count(distinct current.user_id) :: numeric /
			greatest(count distinct previous.user_id), 1),
		2
	) as retention
from user_activity as previous
left join user_activity as current
	on previous.user_id = current.user_id
	and previous.month = (current.month - INTERVAL '1 month')
group by previous.month
order by previous.month asc

```

# Unit Economics

- **Unit Ecomomics:** Masures performance per user

### Average Revenue per User (ARPU)

$$ ARPU = \frac{Revenue}{Num.Customers} $$

- Useful because it measures a company’s success at scaling its business model.

```sql
with kpis as (
	select 
		date_trunc('month', order_date)::date as month,
		sum(meal_price * order_quantity) as revenue,
		count(distinct user_id) as users
	from meals m
	join orders o on m.meal_id = o.meal_id
	group by month
)
select
	month,
	round(revenue::numeric / greatest(users, 1), 2) as arpu
from kpis
order by month asc
```

## Histograms

### Frequency table query

- Query to obtain the frequency table of the num of customers per number of orders

![[Pasted image 20231119122815.png]]

```sql
-- Frequency table for the number of orders
with user_orders as (
	select
		user_id,
		count(distinct order_id) as orders
	from meals
	join orders on meals.meal_id = orders.order_id
	group by user_id
)

select
	orders,
	count(distinct user_id) as users
from user_orders
group by orders
order by orders asc;
```

```sql
-- Fequency table for Rounded REVENUE
with urser_revenues as (
	select
		user_id,
		sum(revenue) as revenue
	from sales
	group by user_id
)

select
-- round to miles (K)
	round(revenue :: numeric, -3) as revenue_1000,
	count(distinct user_id) as users
from user_revenues
group by revenue_1000
order by revenue_1000
```

### Bucketing to customize the histogram (define the number of bins)

![[Pasted image 20231119122906.png]]

```sql
-- Count the number of SKUs that fall into a price category
select
	case
		when price < 4 then 'low-price',
		when price < 6 then 'medium-price',
		else 'high-price'
	end as price_category,
	count(distinct sku_id)
from product_prices
group by price_category
```

## Percentiles and Quantiles

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/25501544-b3f8-481f-a21c-8e68b7a2115c/Untitled.png)

```sql
with user_orders as (
	select
		user_id, count(distinct order_id) as orders
	from orders
	group by user_id
)

select
	round(
		percentile_count(0.25) within group (order by orders asc) :: numeric, 
		2) as orders_p25,
	round(
		percentile_cont(0.50) within group (order by orders asc) :: numeric,
		2) as orders_p50,
	round(
		percentile_cont(0.75) within group (order by orders asc) :: numeric,
		2) as orders_p75,
	round(avg(orders) :: numeric, 2) as avg_orders
from user_orders;
```

# Useful functions

## Dates

```sql
TO_CHAR('2018-08-13', 'FMDay DD, FMMonth YYYY')

-- Another pattern
TO_CHAR('Dy DD/Mon/YYYY')
```

- `‘Dy’` ⇒ Abbrevation of the day
- `‘DD’` ⇒ Day in number

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d5f6e5a4-69d5-45c1-b08f-a9946b1db6e4/Untitled.png)

## Window functions

### Rank

- Assing a rank to each row baed on that row’s prosition in a sorted order
    - `RANK() OVER (ORDER BY revenue DESC)`

## Pivoting

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24f5e6c8-0b2d-4e74-8911-142368624bd2/Untitled.png)

```sql
-- Import tablefunc
CREATE EXTENSION IF NOT EXISTS tablefunc;

SELECT * FROM CROSSTAB(
$$
	SELECT 
		user_id,
		DATE_TRUNC('month', order_date) :: DATE AS delivr_month,
		SUM(meal_price * order_quantity) :: FLAOT AS revenue
	FROM meals
	JOIN orders ON meals.meal_id = orders.meal_id
WHARE user_id IN (0, 1, 2, 3, 4)
GROUP BY user_id, delivr_month
ORDER BY user_id, delivr_month
$$
)
--- Select user ID and the months from june to august 
AS ct (user_id INT,
       "2018-06-01" FLOAT,
       "2018-07-01" FLOAT,
       "2018-08-01" FLOAT)
ORDER BY user_id ASC;
```

# Producing Executive Reports

## Readability

- Use readable date formats
- Round numbers to the second decimal at most
- Table Shaepe ⇒ Reshape long tables into wide ones
- Order ⇒ don’t forget to sort