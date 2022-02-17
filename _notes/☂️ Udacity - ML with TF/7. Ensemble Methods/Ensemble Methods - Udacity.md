---
---

# Ensemble Models

This note is about two types of #EnsembleLearning models:
- 👜 Bootstrap Aggregation -> #Bagging
	- Parallelize models
- 🚀 Boosting => #Boosting
	- Serialize models

***

## Ensembles

- **Weak** learners =>
	- #DecisionTrees are commonly used
	- #DecisionTrees are the default base estimator of `sklearn`

### Why would we want to Ensemble Learners Together?

- To **improve the trade-off** between #Bias and #variance 
	- Simple `linear` models tend to have ==high bias==
	- Other models, like #DecisionTrees, tend to have ==High Variance==
		- Specially those with *no early stopping parameters*
	- A #DecisionTrees is a highly variance algorithm	
- By combining algorithms in an ==Ensemble==:
	- We can often build models that perform better by meeting in the middle in terms of `bias` and `variance`.

#### Introducing randomness into Ensembles
- Introduce `Randomness` into ensembles is a method to improve the performance:
	- Randomness combats the tendency of individual algorithms to ==memorize== data -> 🚨 Overfit 🚨
- There are two ways of introduce `randomness`:
	- 1️⃣ ==Bootstrap the data==: 
		- That is sampling the data with replacement
	- 2️⃣ ==Subset the features randomly==:
		- In **each split** of a decision tree or with each algorithm used in an ensemble only use a subset of the total possible features.

***

## Random Forest

- Combines ==Bootstraping== the *data* and ==Random== subsetting of the *features*.
	- #RandomForest are a type of #Bagging 

### Bagging

- ==Weak== learners => A decision tree of one node
- #RandomForest use ==Voting== and ==Soft-voting== to combine the individual predictions of the #WeakLearners

***

## AdaBoost

- Proposed by Freund and Schapire in 1996
- Is one of the most used algorithms for Boosting
- #AdaBoost uses a ==set of sequential learners==
	- At each new learner the missclassified points have *higher* $weights$
		- This forces the *new model* to focus more on those  points missclassified by the *previous learner*
- The **individual models** are ==combined== on how well they perform individually
	- #AdaBoost provides a $weight$ to each of the individual models/learners

![[Captura de Pantalla 2022-02-13 a la(s) 20.31.54.png]]

##### How to compute the $weight$ of each model?

![[Captura de Pantalla 2022-02-13 a la(s) 20.32.29.png]]

### Exercise

> Calculate the weight of the first model, with 2 significant digits.
![[Pasted image 20220213203317.png]]

- acc Model 1 => 4 + 3 / 8 = 7 / 8 = 0.875
- acc Model 2 => 2 + 2 / 8 = 4 / 8 = 0.5
- acc Model 3 => 1 + 1 / 8 = 2 / 8 = 0.25

**Solution:**

```python
def get_ADAboost_estimator_weight(acc: float):
    estimator_weight = np.log(acc / (1 - acc))
    return estimator_weight

# First model
get_ADAboost_estimator_weight(0.875)
# > 1.9459101490553132

# second model
get_ADAboost_estimator_weight(0.5)
# > 0.0

# second model
get_ADAboost_estimator_weight(0.25)
# > -1.0986122886681098
```

#### Why if a single `WeakLearner` makes has a perfect performance

![[Captura de Pantalla 2022-02-13 a la(s) 20.41.48.png]]

### Combining the models
- Sum the $weights$ of each individual model
	- If they overlap sum their values
	- Use this values to create the ==decision boundary==
- Values with positive values belong to areas from $Positive$ class

### AdaBoost in `sklearn`

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

#### Hyperparameters

- `base_estimator` => The model utilized for the weak learner
- `n_estimators` => The maximum number of weak learners used

```python
from sklearn.tree import DecisionTreeClassifier

model = AdaBootClassifier(
	base_estimator = DecisionTreeClassifier(max_depth = 2),
	n_estimators = 4
)
```

# Recap

In this lesson, you learned about a number of techniques used in ensemble methods. Before looking at the techniques, you saw that there are two variables with tradeoffs **Bias** and **Variance**.

-   **High Bias, Low Variance** models tend to underfit data, as they are not flexible. **Linear models** fall into this category of models.
    
-   **High Variance, Low Bias** models tend to overfit data, as they are too flexible. **Decision trees** fall into this category of models.

### Techniques

You saw a number of ensemble methods in this lesson including:

-   [BaggingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier)
-   [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
-   [AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)

Another really useful guide for ensemble methods can be found [in the documentation here](http://scikit-learn.org/stable/modules/ensemble.html). These methods can also all be extended to regression problems, not just classification.

### Additional Resources

Additionally, here are some great resources on AdaBoost if you'd like to learn some more!

-   Here is the original [paper](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf) from Freund and Schapire.
-   A follow-up [paper](https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf) from the same authors regarding several experiments with Adaboost.
-   A great [tutorial](http://rob.schapire.net/papers/explaining-adaboost.pdf) by Schapire.
