---
---

# Intermediate data visualization with Seaborn

#DataCamp 


***
## Highlights



***


![[Captura de Pantalla 2021-04-23 a la(s) 22.19.43.png]]

 ```python
import seaborn as sns

sns.distplot(df['col'])
```

### Using the distribution plot
- Distplot has multiple arguments

 ```python
# Create a distplot
sns.distplot(df['Award_Amount'],
             kde=False,
             bins=20)

# Display the plot
plt.show()
```

```python
# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})

# Plot the results
plt.show()
```

### Regression plots in Seaborn
==Bivariate analysis==
- regplot
- lmplot

```python
sns.regplot(x='var1', y='var2', data='df')
```

```python
sns.lmplot(x='var1', y='var2', data='df')
```

#### Faceting
![[Captura de Pantalla 2021-04-24 a la(s) 5.54.11.png]]

```python
# Create a regression plot with multiple rows
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           row="Region")

# Show the plot
plt.show()
```

## Customizing Seaborn Plots

- Seaborn has default configuratiosn that can be applied with `sns.set()`

- `sns,set_style()`:
	- `white`
	- `dark.
	- `whitegrid`
	- `darkgrid`
	- `ticks`

- Removing axes with `despine()`:
	- `sns.despine(left=True)`

### Color in seaborn

```python
sns.set(color_codes = True)
sns.distplot('col', color='g')
```

#### Seaborn palettes
- Seaborn has six color palettes:
	- deep, muted, pastel, bright, dark, colorblind

```python
for p in sns.palettes.SEABORN_PALETTES:
	sns.set_palette(p)
	sns.displot(df['col'])
```

![[Captura de Pantalla 2021-04-24 a la(s) 6.12.06.png]]

- Working with palettes
	- `sns.palplot()`: function displays a palette
	- `sns.color_palette()`: returns the current palette

```python
sns.palplot(
    sns.color_palette('Purples', 8)
)

plt.show()

```

- Defining color palettes:
	- Circular colors => Data not ordered
	- Sequential colors => when the data has a consistent range from high to low
	- Diverging colors = when both the low and high values are interesting
	
![[Captura de Pantalla 2021-04-24 a la(s) 6.16.55.png]]

#### Customizing with matplotlib

##### Matplotlib Axes
- Most customization available through matplotlib Axes objects
- Axes can be passes to seaborn functions

```python
fig, ax = plt.subplots()
sns.distplot(df['col'], ax = ax)
ax.set(xlabel = 'My x label')
```

```python
# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.distplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=df['fmr_1'].median(), color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=df['fmr_1'].mean(), color='b', label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()
```

![[descarga.svg]]


## Additional plot types

### Categorical plot types
#### Show each observation
- `stripplot()`
- `swarmplot()`
![[Captura de Pantalla 2021-04-24 a la(s) 6.31.43.png]]

#### Abstract representations
- `boxplot()`: Shows median, and quartiles
- `violinplot()`: A combination of kde and boxplots
- `lvplot()`: Hybrid between boxplot and violinplot -> Scales better with larger datasets -> quick to render
![[Captura de Pantalla 2021-04-24 a la(s) 6.32.17.png]]

#### Statistical Estimates 
- `barplot()`: Shows and estimate value and its confidence interval
- `poinplot()`: Similar to barplot -> useful to observe changes across categorical values
- `countplot()`: Number of instances of each variable

![[Captura de Pantalla 2021-04-24 a la(s) 6.33.26.png]]

## Regression Plots
#### `regplot()`
- regplot()
```python
sns.regplot(
	data = df, x = 'temp', y = 'total_rentals',
	marker = '+'
)
```
![[Captura de Pantalla 2022-05-16 a la(s) 9.34.39.png]]

- Seaborn supports polynomial regression using the `order` parameter

```python
sns.regplot(
	data = df, x='temp', y='total_rentals',
	order = 2
)
```
![[Captura de Pantalla 2022-05-16 a la(s) 9.38.05.png]]

#### `residplot()`
- Evaluating regression with `residplot()`
- Seaborn supports through `residplot` function
- `residplot()`: To plot ==residuals==.
	- *How random distributed are the residuals?* => [[Linear Regression]]
	- It also accepts the `order` parameter
```python
sns.residplot(data = df, x = 'temp', y = 'total_rentals')
```

![[Captura de Pantalla 2022-05-16 a la(s) 9.36.38.png]]


### Regression with categorical values

```python
sns.regplot(
	data = df, x = 'month', y = 'total_rentals',
	x_jitter = .1, order = 2
)
```
![[Captura de Pantalla 2022-05-16 a la(s) 9.39.46.png]]

### Estimators
- Estimators can be useful for highlighting trends
- Use `x_estimator` to aggregate data

```python
sns.regplot(
	data = df,
	x = 'month', y = 'total_rents',
	x_estimator = np.mean, order = 2
)
```
![[Captura de Pantalla 2022-05-16 a la(s) 9.43.59.png]]

### Binning the data
- `x_bins` can be used to divide the data into discrete bins
- The regression line is still fit against all the data

```python
sns.regplot(
	data = df, x = 'temp', y = 'total_rentals',
	x_bins = 4
)
```

![[Captura de Pantalla 2022-05-16 a la(s) 9.45.33.png]]

## Matrix plots => *Heatmaps*

- Seaborn's `heatmap()` function requires data to be in a grid format
- Pandas `crosstab()` is frequently used to manipulate the data
	- This builds a table to summarize the data by the columns selected

```python
sns.heatmap(
	data = pd.crosstab(df['col1'], df['col2']),
	values  = df['total_rentals'], 
	aggfunc = 'mean'
)
```

![[Captura de Pantalla 2022-05-16 a la(s) 9.51.50.png]]

#### Customize a heatmap
- `cbar = False`
- `cmap = "YlGnBy"`
- `linewidths = 0.5`
- `annot = True`
- `fmt = 'd'`

#### Centering a heatmap
- Seaborn support centering the heatmap colors on an specific value

```python
sns.heatmap
```

![[Captura de Pantalla 2022-05-16 a la(s) 9.53.44.png]]

## FacetGrid, factorplot and lmplot

- Useful to analyze data with many variables
	- This is known as faceting
	- Data must be in tidy format
		- `rows` => observations
		- `columns` => variables

### FacetGrid Categorical Example

- `FacetGrid`
	- Requires a `.map` method where a plotting function is defined
```python
g = sns.FacetGrid(df, col = 'HIGHDEG')
g.map(sns.boxplot, 'Tuition',
	  # Define the order of the `Tuition` valeus
	 order = ['1', '2', '3', '4'])
```

```python
# Create FacetGrid with Degree_Type and specify the order of the rows using row_order
g2 = sns.FacetGrid(
             data = df, 
             row = "Degree_Type",
             row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

# Map a pointplot of SAT_AVG_ALL onto the grid
g2.map(sns.pointplot, 'SAT_AVG_ALL')

# Show the plot
plt.show()
plt.clf()
```

- `sns.factorplot`
	- Combines the facetting and mapping process into one function

```python
sns.factorplot(
	x = 'Tuition', data = df,
	col ='HIGHDEG', king = 'box'
)
```

```python
# Create a factor plot that contains boxplots of Tuition values
sns.factorplot(data=df,
         x='Tuition',
         kind='box',
         row='Degree_Type')

plt.show()
plt.clf()
```

![[Pasted image 20220517214443.png]]

```python
# Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type 
sns.factorplot(data=df,
        x='SAT_AVG_ALL',
        kind='point',
        row='Degree_Type',
        row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

plt.show()
plt.clf()
```

![[Pasted image 20220517214557.png]]

- `sns.lmplot`
	- Used for continue values
	- Plots scatter and regression plots on a `FacetGrid`

```python
sns.lmplot(data = df, 
		   x = 'Tuition', 
		   y = 'SAT_AVG_ALL',
		   col = 'HIGHDEG', 
		   fit_reg = False)
```

```python
# Re-create the previous plot as an lmplot
sns.lmplot(data=df,
        x='UG',
        y='PCTPELL',
        col="Degree_Type",
        col_order=degree_ord)

plt.show()
plt.clf()
```

![[Pasted image 20220517214714.png]]

```python
# Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
sns.lmplot(data=df,
        x='SAT_AVG_ALL',
        y='Tuition',
        col="Ownership",
        row='Degree_Type',
        row_order=['Graduate', 'Bachelors'],
        hue='WOMENONLY',
        col_order=inst_ord)

plt.show()
plt.clf()
```

![[Pasted image 20220517214748.png]]

## PairGrid and pairplot

- Similar to `FacetGrid`
- Allows to see interactions across different columns of data
- However, with these functions we only define the columns of data we want to compare
	- `PairGrid` => shows pairwise relationships between data elements

```python
# Create the same PairGrid but map a histogram on the diag
g = sns.PairGrid(df, vars =["fatal_collisions", "premiums"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)

plt.show()
plt.clf()
```


- `pairplot` is a shortcut for `PairGrid`

```python
# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=df,
        vars = ["fatal_collisions", "premiums"],
        kind = 'scatter',
        hue='Region',
        palette='RdBu',
        diag_kws={'alpha':.5})

plt.show()
plt.clf()
```

```python
# Build a pairplot with different x and y variables
sns.pairplot(data=df,
        x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],
        y_vars=['premiums', 'insurance_losses'],
        kind='scatter',
        hue='Region',
        palette='husl')

plt.show()
plt.clf()
```

![[Pasted image 20220517215425.png]]

```python
# plot relationships between insurance_losses and premiums
sns.pairplot(data=df,
             vars=["insurance_losses", "premiums"],
             kind='reg',
             palette='BrBG',
             diag_kind = 'kde',
             hue='Region')

plt.show()
plt.clf()
```

![[Pasted image 20220517215600.png]]

## JoinGrid and jpinplot

- Allows us to compare the distribution of data between two variables
- Uses:
	- Scatter plots
	- Regression lines
	- Histograms
	- kernel density estimates

![[Captura de Pantalla 2022-05-17 a la(s) 21.57.39.png]]

### Advanced JointGrid
![[Captura de Pantalla 2022-05-17 a la(s) 21.58.49.png]]

### `jointplot()`
- `jointplot()`

![[Captura de Pantalla 2022-05-17 a la(s) 21.59.33.png]]

#### Customizing a joinplot

![[Captura de Pantalla 2022-05-17 a la(s) 22.00.15.png]]

### Examples

```python
# Build a JointGrid comparing humidity and total_rentals
sns.set_style("whitegrid")
g = sns.JointGrid(x="hum",
                  y="total_rentals",
                  data=df,
                  xlim=(0.1, 1.0))

g.plot(sns.regplot, sns.distplot)

plt.show()
plt.clf()
```

![[Pasted image 20220517220217.png]]

```python
# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp",
         y="total_rentals",
         kind='reg',
         data=df,
         order=2,
         xlim=(0, 1))

plt.show()
plt.clf()
```

![[Pasted image 20220517220312.png]]

```python
# Plot a jointplot showing the residuals
sns.jointplot(x="temp",
        y="total_rentals",
        kind='resid',
        data=df,
        order=2)

plt.show()
plt.clf()
```

![[Pasted image 20220517220337.png]]

```python
# Create a jointplot of temp vs. casual riders
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="temp",
             y="casual",
             kind='scatter',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
    
plt.show()
plt.clf()
```

![[Pasted image 20220517220411.png]]