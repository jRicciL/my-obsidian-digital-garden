---
---

# Intermediate data visualization with Seaborn

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

### Regression Plots

- regplot()
```python

```

- residplot(): To plot residuals.
	- ==How random distributed are the residuals?==
```python

```

- Using estimatos:
```python

```

- Binning the data
```python

```