---
---

# Gradient Boosted Decision Trees

- Use an ensemble of decision trees =>
	- For classification and regression
- The key idea is to build a series of trees:
	- Trees are trained sequentially.
	- Trees are small => ==weak learners==
		- `max_depth = 2` or `3`
	- Each tree attempts to correct errors from the previous stage.
- Once trained, predictions are fast and does not take too much memory
- The ==learning rate== controls how hard each new tree tries to correct remaining mistakes from previous round.
	- **High learning rate**: more complex
		- => More complex overall model
	- **Low learning rate**: simpler trees
		- => Could be used for regularization


#### Python code
```python
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor
```

### Pros and Cons

#### Pros
- Often best off-the-shelf accuracy on many problems
- Prediction requires modest memory and is fast
- Does not require Feature scaling => careful normalization of features to perform well
- Handles a mixture of feature types.


#### Cons
- Like [[W4 - Random Forest]], the models are often difficult for humans to interpret.
- Requires careful tuning of learning rate and other parameters.
- Training requires significant computation.
- Not recommended for problems with very high dimensional sparse features.

### Key parameters

Typically tuned together:
- `n_estimators` =>
	- Number of small decision trees to use
	- 🔴  Unlike random forest, ==**increasing the number of trees** -> Overfitting!==
- `learning_rate` =>
	- Controls emphasis on fixing error from previous iteration
- `max_depth` =>
	- Can also have an effect on model complexity
	- GBT assumes each tree is a weak learner, therefore, depths usually have low values