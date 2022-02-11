---
---

# Linear Regression Concepts

## The absolute trick
#AbsolutTrick

1. Start with a **point** $(p,q)$ an a **line** $f$
2. Get the line close to the point
3. $y=w_1x + w_2$
4. Add $p$ to $w_1$ 
5. $y = (w_1 + p)x + (w_2 + 1)$
	- Here, we add to $w_1$ an step of size $p$, which is quite great.
	- In #MachineLearning, we want to take little steps
		- We will call this step size => #learning-rate 
- ==Learning rate==: $\alpha = 0.05$
- $y = (w_1 + \alpha \cdot p)x + (w_2 + \alpha)$

> Basically, sum $\alpha \times p$ to $w1$ and $\alpha$ to $w_2$
> - $p$ affects only $x$ through $w_1$
![[Captura de Pantalla 2022-02-09 a la(s) 19.01.42.png]]


## The square trick
#SquareTrick

Similar to [[#The square trick]], however, here we will subtract to $q$ the current value $q'$ of line $f$ at $f=(p, q')$:
- $y = (w_1 + p[q - q']\alpha)x + (w_2 + [q - q']\alpha)$

y = (-0.6 + (-5)(-4)(0.01)) +  (4 - (-4)(0.01))

![[Captura de Pantalla 2022-02-09 a la(s) 19.01.50.png]]

## Gradient Descent
- Develop an algorithm that best fits a set of points
	- Reduce the error of the model 
	- ==Gradient Descent==^[[[Optimizers]]] is used to reduce the error
		- Find the gradient => the direction to follow to minimize the error
			- The parameters $w$ are updated using this gradient
			- The gradient contains the partial derivatives of the Error function in relation with each parameter $w_i$
		- We need an ==Error function==
		-
### Gradient descent formula
> Gradient of **Error Function**
$$w_i \leftarrow w_i + \alpha \frac{\partial}{\partial w_i}\mathbf{Error Function}$$

## Error Function

### Mean Absolute Error
> The error is ==always== positive

- The ==error== at $x_i$ => $error = |y - \hat y|$
- The ==TOTAL== error (norm 1) => **The Total Absolute Error**
	- $\mathbf{Error} = \sum^m_i |y - \hat y|$
- The ==Mean Absolute Error==:
	- $\frac{1}{m}\mathbf{Error} = \sum^m_i |y - \hat y|$

```python
np.abs(y_pred - y).mean()
```

### Mean Squared Error

- We use the second norm -> squaring the error
	- $\frac{1}{2m}\mathbf{Error} = \sum^m_i (y - \hat y)^2$
	- Divide between $1/2$ simplifies the computation of the derivative
		- This does not affect the Gradient descent process but simplifies computation

```python
((y_pred - y)**2).mean()
```

## Minimizing Error Function

The error defined to be 

$\mathbf{Error} = \frac{1}{2}(y - \hat y)^2$

Also, the prediction $\hat y$
$\hat y = w_1 x + w_2$

So, because $\hat y$ is a function of $x$ we use the chain rule to calculate the derivative of the Error with respect to $w_1$

$$\frac{\partial}{\partial w_1}\mathbf{Error} = \frac{\partial \mathbf{Error}}{\partial \hat y} \frac{\partial \hat y}{\partial w_1}$$

- The first factor of the right hand side is the derivative of the Error with respect to the prediction $\hat y$ which is $-(y - \hat y)$

- The second factor is the derivative of the prediction with respect to $w_1$ , which is $x$