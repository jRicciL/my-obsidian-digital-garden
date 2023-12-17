---
---

# Hyperparameter Tuning

## Neural Networks hyperparameters
- `learning rate` => $\alpha$ is the most important
- Mini-batch size
- Number of layers: **Architecture**
- Number of hidden units: **Architecture**
- Momentum ($\eta, \beta_1$) and decay ($\beta_2$) parameters

#### Recommendations
1. Don't use a grid. Try random values.

2. ==Coarse to => fine== implementation. #CoarseToFine
![[Captura de Pantalla 2021-08-31 a la(s) 17.41.36.png]]

#### Using an Appropiate Scale to pick Hyperparameters
- Don't sample uniformly random:
	- Particularly for some hyperparameters
	- Use $log$ scale for `learning_rate` and other hyperparameters
		- `r = -4 * np.random.rand()`
		- `alpha = 10 ** r`
	- Do the same for **Hyperparameters for exponentially weighted averages**

## Practice: Pandas vs. Caviar
##### Two approaches
- ==Panda==: Babysitting one model
- ==Caviar==: Training many models in parallel
![[Captura de Pantalla 2021-08-31 a la(s) 20.20.20.png]]