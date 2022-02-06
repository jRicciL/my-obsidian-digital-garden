---
---

# Time Series Analysis in Python

## Chapter 1: Introduction
- [[Time Series]] deals with data that is ordered in time.

### Goals
- Learn about time series models
- fit data to a times series model
- Use the models to make forecasts of the future
- Learn how to use the relevant statistical packages in Python

#### Pandas Tools for time series
- Changing an index to datatime:
```python
df.index = pd.to_datetime(df.idex)
```

- Join two DataFrames:
```python
df1.join(df2)
```

- Computing percent changes and differences of a time series:
```python
df['col'].pct_change()
df['col'].diff()
```

- Plotting data:
```python
df.plot()
```