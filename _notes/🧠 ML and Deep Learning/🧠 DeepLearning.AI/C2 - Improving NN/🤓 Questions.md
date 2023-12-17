---
---

# Quizzes

# W3:
1. *If searching among a large number of hyperparameters, you should try values in a grid rather than random values, so that you can carry out the search more systematically and not rely on chance. True or False?* 
	- => ==False==
2. *Every hyperparameter, if set poorly, can have a huge negative impact on training, and so all hyperparameters are about equally important to tune well. True or False?* 
	- => ==False==
1. *During hyperparameter search, whether you try to babysit one model (“Panda” strategy) or train a lot of models in parallel (“Caviar”) is largely determined by:* ==> The amount of computational power you can access
2. *If you think β (hyperparameter for momentum) is between 0.9 and 0.99, which of the following is the recommended way to sample a value for beta?* ==>
```python
r = np.random.rand() 
beta = 1-10**(- r - 1)
```
5. *Finding good hyperparameter values is very time-consuming. So typically you should do it once at the start of the project, and try to find very good hyperparameters so that you don’t ever have to revisit tuning them again. True or false?* ==> ==False==
6. *In batch normalization as presented in the videos, if you apply it on the llth layer of your neural network, what are you normalizing?* ==> $Z$
7. *Which of the following statements about γ and β in Batch Norm are true?* ==> 
	- They set the mean and variance of the linear variable $z^[l]$ of a given layer.
	- They can be learned using Adam, Gradient descent with momentum, or RMSprop, not just with gradient descent.