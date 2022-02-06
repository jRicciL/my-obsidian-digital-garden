---
---

# Optimization Formulation
[[Mathematical Theories of Machine Learning]]

The problems can be solve from two angles:

- [[Expectation-maximization]] or #EM 

### Expectation-maximization
- It is used to find ==local maximum likelihood parameters== of a statistical model in scenarios wherein the equations cannot be solved directly.
- This models used latent variables along with unknown parameters and known data observations.


### Markov Chain Monte Carlo
- MCMC methods are primarily used for calculation of the numerical approximations of multidimensional integrals, as used in Bayesian statisticas.

## Optimization techniques needed for machine Learning

#### [[Linear Regression]]
- Considered as one of the standard models in statistics.
- Find the #maximum-likelihood is equivalent to the expression:

$$argmax = - argmin \left[ \frac{1}{2} \sum^n (x_i - y_i)^2 \right]$$

#### [[Ridge Regression]] or L2
- Its advantage is stability
Finding the maximum posterior estimate:

$$argmax = - argmin \left[ \frac{1}{2} \sum^n (x_i - y_i)^2 \right] + \frac{1}{\sigma_0^2}\cdot |y|]$$

#### [[Lasso Regression]] or L1
- Its advantage is sparsity

## Local minima and non-convex functions
![[Captura de Pantalla 2021-01-24 a la(s) 21.15.33.png]]

- Many statistics models are finally transformed to solve an optimization problem:
	- simple convex optimization
	- complex non-convex optimization

#### Optimization algorithms

- The optimization algorithms are based on the information form the objective function
- ==Assumption for smoothness of a function==:
	- **Zero-order** oracle assumption => returns $f(x)$
	- **First-order** oracle assumption => returns $f(x)$ and the gradient $\nabla_xf(x)$
	- **Second-order** oracle assumption => returns $f(x)$, the gradient $\nabla_x^2f(x)$, and the Hessian 


##### Zero-order
- Not popular in practice due to its higher iteration complexity
- Two main types:
	1. Kernel-based bandit algorithms.
	2. Algorithms of single point gradient estimation.

##### Second-order 
- Difficult to compute due to the need to calculate the Hessian matrices.
- Have been studied widespread for the last two decades.
- Based on classical [[Newton iteration]], #Newton-methods, [[Cholesky's method]], cubic-regularization, and trust region method.

##### First order (Most used)
- Used widespread.
- Only need to compute gradient which takes $O(d)$ time complexity, where $d$ is the size of the dimensions.


### Gradient Descent
- By far, one of the major strategies used in machine learning and deep learning.
- It is an algorithm that works towards minimizing the given function.

##### Foundation
Given a $C^1$ and $C^2$ function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ with unconstrained variable $x \in \mathbb{R}^n$, GD uses the following update rule:
- $x^{t=1}=x^{t} - \alpha \nabla f(x^t),$

where $\alpha$ is the step size, and can be either fixed or vary across iterations $t$.

For a non-convex function, GD will stop when $\nablaf(x)=0$, and thus will be no update: $x^{t=1}=x^{t}$.