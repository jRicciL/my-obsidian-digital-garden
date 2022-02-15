---
---

# Linear Regression Concepts

## The absolute trick
#AbsolutTrick

1. Start with a **point** $(p,q)$ an a **line** $f$
2. Get the line close to the point
3. $y=w_1x + w_2$
4. Add $p$ to $w_1$ 
	- Add if $q > q'$, subtract otherwise
5. $y = (w_1 + p)x + (w_2 + 1)$
	- Here, we add to $w_1$ an step of size $p$
	- However, in #MachineLearning, we want to take little steps
		- We will call this step size => #learning-rate 
- ==Learning rate==: $\alpha = 0.05$
- $y = (w_1 + \alpha \cdot p)x + (w_2 + \alpha)$

#### Even simpler
- Basically, **add or subtract**  $\alpha \times p$ to $w_1$ and $\alpha$ to $w_2$
	- Add if $q > q'$, subtract otherwise

> $p$ affects only $x$ through $w_1$

```python
def absolute_trick(w_1: float, 
		   w_2: float, 
		   point: tuple, 
		   alpha: float = 0.01):
    p, q = point
    # Change the signs if the line is above the point
    q_prime = w_1 * p + w_2 # compute q_p (or y)
    alpha = - alpha if q_prime > q else alpha
    w1 = w_1 + p * alpha
    w2 = w_2 + alpha
    return w1, w2


# test for y = -0.6 + 4
absolute_trick(w_1 = -0.6, w_2 = 4, 
			   point = (-5, 3), alpha = 0.1)
```

```
(-0.09999999999999998, 3.9)
```

![[Captura de Pantalla 2022-02-09 a la(s) 19.01.42.png]]


## The square trick
#SquareTrick

Similar to [[#The absolute trick]], however, here we will *add* the current value $q'$ to $q$, $q'$ comes from line $f$ at $f=(p, q')$:
- $y = (w_1 + p[q - q']\alpha)x + (w_2 + [q - q']\alpha)$
- The idea is to ==add the vertical distance== into the formula by considering $q$
- Different form the **Absolute trick** it is not necessary to verify if the *point* is below or above the line (whether $q < q'$), since the term $(q - q')$ evaluates this and determines if the value is added or subtracted

![[Captura de Pantalla 2022-02-09 a la(s) 19.01.50.png]]

```python
def square_trick(w_1: float, 
                 w_2: float, 
                 point: tuple, 
                 alpha: float = 0.01):
    p, q = point
    q_prime = w_1 * p + w_2
	# NO NEED to verify if q < q'
    w1 = w_1 + p * (q - q_prime) * alpha
    w2 = w_2 + (q - q_prime) * alpha
    return w1, w2

square_trick(w_1 = -0.6, w_2 = 4, point = (-5, 3), alpha = 0.01)
```
```
(-0.39999999999999997, 3.96)
```

## Gradient Descent
- Develop an **algorithm** that *best fits* a set of points
	- Reduce the `error` of the model 
	- ==Gradient Descent==^[[[Optimizers]]] is used to reduce the error
		- Find the gradient => *the direction to follow to minimize the error*
			- The parameters $w$ are updated using this gradient
			- The gradient contains the partial derivatives of the Error function in relation with each parameter $w_i$
		- We need an ==Error function==


### Gradient descent formula
> Gradient of **Error Function**

$$w_i \leftarrow w_i + \alpha \frac{\partial}{\partial w_i}\mathbf{Error Function}$$

## Error Function

### Mean Absolute Error
- The error is ==always== positive
- The ==error== at $x_i$ => 
	- $error = \|y - \hat y\|$
- The ==TOTAL== error (norm 1) => **The Total Absolute Error**
	- $\mathbf{Error} = \sum^m_i \|y - \hat y\|$
- The ==Mean Absolute Error==:
	- $\frac{1}{m}\mathbf{Error} = \sum^m_i \|y - \hat y\|$

```python
# Mean Absolute Error
np.abs(y_pred - y).mean()
```

### Mean Squared Error
 
- We use the second norm -> squaring the error
	- $\frac{1}{2m}\mathbf{Error} = \sum^m_i (y - \hat y)^2$
	- Divide between $1/2$ simplifies the computation of the derivative
		- This does not affect the Gradient descent process but simplifies computation

```python
# Mean Squared Error
((y_pred - y)**2).mean()
```

- #MSE is a quadratic function that has a minimum at the point in the middle.
- 🟠 #MSE penalizes more **higher errors**

***

## Minimizing Error Function

The error is defined to be 
- $\mathbf{Error} = \frac{1}{2}(y - \hat y)^2$

Also, the prediction $\hat y$ is defined as:
- $\hat y = w_1 x + w_2$

So, because $\hat y$ is a function of $x$ we use the **chain rule** to calculate the *derivative* of the Error with respect to $w$ ($w_1$ for this case):

$$\frac{\partial}{\partial w_1}\mathbf{Error} = \frac{\partial \mathbf{Error}}{\partial \hat y} \frac{\partial \hat y}{\partial w_1}$$

- The ==first factor== of the right hand side is the derivative of the Error with respect to the prediction $\hat y$ which is $-(y - \hat y)$

- The ==second factor== is the derivative of the prediction with respect to $w_1$ , which is $x$
- Therefor the **derivative** is:

$$\frac{\partial}{\partial w_1}\mathbf{Error}  = -(y - \hat y) \cdot x$$

***

## Using Gradient Descent instead the Normal Equation
- For a problem with $n$ features/variables ->
	- We end with a system of $n$ ==Equations== with $n$ ==unknowns==
	- Solving for systems with a $n$ large -> 
		- <mark style='background-color: #FFA793 !important'>Is very computationally expensive</mark>

## Mean vs Total Error

- Since derivatives are linear functions, the gradient of the total error is just a multiple of the mean error.
	- $M = mT$
- The gradient step consists of subtracting the gradient of the error times the ==learning rate== $\alpha$.
- Therefore choosing between the mean squared error and the total squared error really just mounts to picking a different #learning-rate.

## Gradient Descent Flavors
> Related notes
> -> [[C4 Gradient descent]] and [[Optimizers]]
> --> [[Optimizers#Batch and Stochastic Gradient Descent]]

🔴 <mark style="background-color: #FFA793 !important">There are two ways of do linear regression</mark>:

### Stochastic Gradient Descent
1. Applying the squared (or absolute trick) at every point in our data ==one by one==, and repeat this process many times.
	- Compute the gradient and update the $weights$ **for each data point**.

### Batch Gradient Descent
2. By applying the squared (or absolute) trick at every point in our data ==all at the same time==, and repeating the process many times.
	- Compute the gradient and update the $weights$ using **all data** points at the **same time**

![[Pasted image 20220211192730.png]]

### Mini-batch
- Is the **most** used in practice.
	- We used *batches* of size $b$, with $b << m$, where $m$ is the number of data points.
	- All data points belonging to <mark style='background-color: #9CE684 !important'>each batch</mark> are used to perform a iteration of gradient descent and backpropagation

Visit: [[Optimizers]]

#### Quiz: Mini-Batch Gradient Descent 

- Write a function that executes mini-batch gradient descent to find a best-fitting regression line.

##### Gradient Descent Steps

```python
import numpy as np
np.random.seed(42)

def MSEStep(X, y, W, b, learning_rate = 0.005):
	"""
	This function implements the gradient descent step 
	for squared error as a performance metric.
	
	Parameters
	X : array of predictior features
	y : array of outcome values
	W : predictor feature coeffients
	b : regression function intercept
	
	Returns:
	W_new : predictor feature coefficients following gradient descent step
	b_new : intercept following gradient descent step
	"""
	# Make the predictions
	y_pred = X.dot(W) + b
	# Compute the Error
	error  = y - y_pred
	# Get the partial derivatives 
	dW     = - error.dot(X) 
	db     = - error.sum()
	# Update the weights using the given `learning_rate`
	W_new  = W - learning_rate * dW
	b_new  = b - learning_rate * db
	
	return W_new, b_new
```

###### Steps
1. $\hat y = W^TX + b$
2. $\mathbf{error} = (y - \hat y)$
3. $\frac{\partial}{\partial W}\mathbf{Error}  = -(y - \hat y) \cdot X = \mathbf{error} \cdot X$
4. $\frac{\partial}{\partial b}\mathbf{Error}  = -(y - \hat y)$
5. $W = W - \alpha \cdot \frac{\partial}{\partial w_1}\mathbf{Error}$
6. $b = b - \alpha \cdot \frac{\partial}{\partial b}\mathbf{Error}$

##### Mini-batch implementation

```python
def miniBatchGD(X, y, 
				batch_size = 20, 
				learn_rate = 0.005, 
				num_iter = 25):
    """
    This function performs mini-batch gradient descent on a given dataset.

    Parameters
    X : array of predictor features
    y : array of outcome values
    batch_size : how many data points will be sampled for each iteration
    learn_rate : learning rate
    num_iter : number of batches used

    Returns
    regression_coef : array of slopes and intercepts generated by gradient
      descent procedure
    """
    n_points = X.shape[0]
    W = np.zeros(X.shape[1]) # coefficients
    b = 0 # intercept
    
    # run iterations
    regression_coef = [np.hstack((W,b))]
    for _ in range(num_iter):
        batch = np.random.choice(range(n_points), batch_size)
        X_batch = X[batch,:]
        y_batch = y[batch]
		# Use gradient descent
        W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
        regression_coef.append(np.hstack((W,b)))
    
    return regression_coef
```

***

## Linear Regression with `SkLearn`

```python
# TODO: Add import stateBMIts
import pandas as pd
from sklearn.linear_model import LinearRegression


# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv') 
X = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)

```

## Multiple Linear Regression
- $X$ is $n$ dimensional: $X \in \mathbf{R}^n$ -> ==Predictive variable== or ==Independent==
- $y$ is one dimension: $y \in \mathbf{R}^1$ -> ==Response variable== or ==Dependent==
- The model is an $n-1$ hyperplane

## Polynomial Regression

- Considers ==high degree polynomials== instead of using a simple linear model.

***

# Linear Regression Warnings

![[Pasted image 20220212161857.png]]

1. **Linear regression work best with the data is Linear**
	- If data **is not** lineal:
		- 1️⃣ Transform your data
		- 2️⃣ Add features
		- 3️⃣ Try other models

2. **Linear regression is sensitive to outliers**
	- If there are outliers
		- Use techniques to identify them
			- plotting
		- Use a #Zscore
		- Use the Inter Quartile Range => #IQR

***

# Regularization
Related notes: 
> [[Regularization]], [[ML_with_Spark#Regularization]], [[Optimization Formulation]]

### Model complexity

![[Captura de Pantalla 2022-02-12 a la(s) 16.27.04.png]]
### #Regularization 
- Reduces the ==variance== of the model by reducing the model complexity
	- Reduce the **combined error** 
	- <mark style="background-color: #9CE684 !important">Simpler models</mark> have a tendency to **generalize better**.

## Lasso and Ridge Regularization => $L1$ and $L2$
- $L1$ => `Lasso`
- $L2$ => `Ridge`
- $L1$ and $L2$ regularization:
	- Penalizes a model by considering the number and size of their parameters/$weights$
- Uses a $\lambda$ (`lambda`) hyperparameter that determines *how much the model will be penalized.*

![[Captura de Pantalla 2022-02-12 a la(s) 16.38.34.png]]

### L1 Regularization
- Take all of the coefficients (parameters or $weights$) of the model.
- Get their **absolute values**.
- Sum all values => ==combined error==

### L2 Regularization
- Similar to $L1$ Regularization:
	- It's a #Weight-decay method
	- Add the squares of all of the parameters of the model

## L1 vs L2 Regularization

| L1 (Lasso)                     | L2 (Ridge)                     |
| ------------------------------ | ------------------------------ |
| ❌  Computationally Inefficient | ✅Computationally efficient    |
| ✅ Sparse outputs              | ✅ Non-Sparse Outputs          |
| ✅ Used for Feature Selection     | ❌ Not used for Feature Selection |


#### Exercise

```python
# TODO: Add import statements
from sklearn.linear_model import Lasso
import pandas as pd

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:, :6]
y = train_data.iloc[:, -1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
```

# Feature Scaling
==Feature Scaling== => is a way of transforming the features into a common range of values

#### 🔥 When should I use Feature Scaling?
1. When the algorithm uses a ==distance-based metric== to predict.
	1. **Support Vector Machines** -> [[SVM]]
	2. **k-Nearest Neighbors** 
2. When #Regularization  is incorporated:
	- Because the penalty on particular parameters depend of the ==scale== of the feature they are associated with.
3. Makes the training process **less sensitive** to the scale of features.
4. Makes optimization well-conditioned:
	- Improves the behavior of [[C4 Gradient descent]] algorithms

### Standardizing
- To standardize a vector $x$ we subtract its mean ($\bar x$) to all values (*centralization*), and then divide them by the standard deviation of $x$; $\mathbf{std}(x)$ (*scaling*).

```python
def standardize(x: np.array):
	mean = x.mean()
	std  = x.std()
	x_st = (x - mean) / std
	return x_st
```

### Min-max normalization
- Considers the `min` and `max` values of the vector $x$ to scale its values between $0$ and $1$.

```python
def min_max_normalization(x: np.array):
	min_val = x.min()
	max_val = x.max()
	x_norm  = (x - min_val) / (max_val - min_val)
	return x_norm
```

# Recap

In this lesson, you were introduced to linear models. Specifically, you saw:

-   **Gradient descent** as a method to optimize your linear models.
-   **Multiple Linear Regression** as a technique for when you are comparing more than two variables.
-   **Polynomial Regression** for relationships between variables that aren't linear.
-   **Regularization** as a technique to assure that your models will not only fit to the data available, but also extend to new situations.

### Outro

In this lesson, you were predicting quantitative values. Predicting quantitative values is often just considered a `Regression` problem. In the next lesson, you will be introduced to predicting a category, which is called a `Classification` problem.

## Related Notes
- [[Linear Regression]]
- [[Local Optima]]
- [[Gradient_checking]]
- [[C4 Gradient descent]]