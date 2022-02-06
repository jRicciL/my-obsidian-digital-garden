---
---

# Linear Regression

> *Essentially, all models are wrong, but some are useful* $--$ George Box

## Assessing Model Fit

- How well does our model works?
- Sum of squares deviation.

```R
library(broom)

mod <- lm(y ~ x, data=data)
mod %>%
	augment() %>%
	summarize(SSE = sum(.resid^2),
			  SSE_also = (n() -1 * var(.resid))
```

### MAE - Mean Absolute Error
#MAE : It is the norm *L1* or #Lasso
	- $MAE_{(X,\theta)}= \frac{1}{m} \sum^m_{i=1}|\theta^T \cdot x_i - y_i|$
- Robust to outliers
- Easy to interpret

---
### SSE - Sum of Squared Error:
Related to Norm 2 or #Ridge
- $RSS(\theta)$ or [[Least Squares Method]]
	- $SSE = \sum_{i=1}^m(y_i - \hat{y})^2$
- [[SSE]] is equal to the variance times m - 1, where $m$ is the number of observations:
	- $SSE = (m -1) \cdot var(\textbf{residuals})$

---
### MSE - Mean Squared Error
Related to norm 2 or #Ridge
- In practice it is easier to minimize than the RMSE
	- $MSE = E[(\hat\theta - \theta)^2]$
	- $MSE(X, \theta) = \frac{1}{m}\sum_{i=1}^m (\theta^T \cdot x_i - y_i)^2$
- It is equivalent to compute the variance of the residuals:
	- $S^2_{res} = \frac{1}{m-1}\sum_{i=1}^m(res_i - \bar{res})^2$
	- This is because the mean of the residuals is approximately 0 (with an intercept in the model) => $(y_i - \hat{y_i}) = (res_i - 0) = res_i$ 
	

---
### RMSE - Root Mean Squared Error
It is equivalent to the standard deviation of the residuals (==**Residual Standard Error**==)
	- $RMSE=\sqrt{ \sum_i e^2_i /m - 2} = \sqrt{ SSE/m - 2}$
- it is divided by the number of degrees of freedom $n -2$

---
### R-squared
#R2: Measures the variability in the response variable explained by the predicted variables.
- The idea is to compare the performance of our model with respect to a benchmark.
- But if we do not have a **Test set** we still can use the ==average== as a reference.

#### The Null (average) model
#Null-model : Without a fitted model, the average is the best prediction for new observations:
- $\hat{y} = \bar{y}$
- It is like a dummy model that always predict the mean: $\bar y$

```R
# Null model
null_mod <- lm(y ~ 1, data = data)
```

#### R-squared => Coefficient of determination
==*How much variability is explained by the model?*==
$$R^2 = 1 - \frac{SSE}{SST} = 1 - \frac{Var(e)}{Var(y)} = 1 - \frac{\sum(y_i - \hat y_i)}{\sum(y_i - \bar y_i)}$$

where $SST$ is the $SSE$ of the null model, i.e. with respect to always predict the mean. $e$ is the vector of residuals.

Properties:
- Its value is bounded at +1: $R \in (-\infty, 1]$
- High $R^2$ value not necessarily guaranties a good model:
	- The model could be overfitted
	- Could violate some assumptions

#### R^2 and Correlation
- For a simple linear regression, with **only one predictive variable**:
	- $r^2_{x,y} = R^2$

## Unusual Points
- #Outliers: *How individual observations affect the slope of the model?*

- [[Leverage]] and [[Influence]] help us to quantify the intuition.

### Leverage
- It is computed for each observation $i$ as:
$$h_i = \frac{1}{n} + (x_i - \bar x)^2 / \sum(x_i - \bar x)^2$$
- Represents the distance between the instance  $i$ between the value of the explanatory variable ($x_i$) and the mean of the explanatory variable ($\bar x$)
- It only takes into account the explanatory variable (x).

### Influence
- The influence of an observation depends not only on its leverage, but also on the magnitude of its residual.
- depends on the response variable (y) and the fitted value (y^).

#### Dealing with outliers

1. What is the justification to remove some points?
2. How does the scope of inference change?
