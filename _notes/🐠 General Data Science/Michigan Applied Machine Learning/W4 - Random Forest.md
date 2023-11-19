---
---

# Random Forest

#EnsembleLearning
#RandomForest

## Ensemble Learning
- An ensemble takes multiple individual learning models and combines them to produce an aggregate model that is more powerful than any individual model.
- Why they work?
	- Because individual models tend to make mistakes due to overfitting on different parts of data
	- The ensemble model averages the individual mistakes reducing them.

## Random Forest
- An ensemble of trees -> A case of boostrap aggregation => Bagging
- Widely used -> Very good results on many forest

### Random variation
- Randomness occurs during training => While the trees are created
 ==Bootstrapping== => 
 Random selection with replacement
	1. *Bagging*=> The ==data== used to build each tree is selected randomly
	2. *Random subspace method* => The ==features== are also selected randomly.
- The randomness guaranties that all of the trees will be different.
	- With different spliting rules

#### Selection of the features
- During the tree construction -> the best split is selected by using a subset of features randomly chosen.
	- This process is performed at each node of the trees
- The number of these features are controlled by `max_features` 
- 🔴  Learning is quite sensitive to `max_features`:
	- With `max_features = 1` => 
		- Splits are limited to one feature (the one chosen randomly)
		- The forest will be very diverse with complex trees.
		- Each tree will possibly have many labels in order to fit the target value 
			- Only if the `deep` allows it
	- With `max_features = n` (close to the $n$ number of features) => Homogenius forest with simpler trees
		- Will probably require few levels

#### Prediction
The prediction process consists of:
- Make a prediction for every tree in the forest.
- Combine the individual predictions:
	- For Regression: 
		- Compute the mean
	- For Classification -> Weighted vote 
		- Get the probability per class of each tree
		- Average the probabilities across trees
		- Predict the class with the highest probability average
	

## Pros and Cons
### Pros
- Widely used
- Excellent prediction performance on many problems
- Does not require careful preprocessing steps:
	- Normalization of features
- Does not require extensive parameter tuning
- Handles a mixture of feature types
- Easily parallelized across multiple CPUs

### Cons
- The resulting models are often difficult for humans to interpret
- Like decision trees, random forest may not be a good choice for very high dimensional data

## Key parameters

- `n_estimators`: number of trees to use in ensemble => default is `10`
	- Should be larger for larger datasets to reduce overfitting
	- If there are few trees:
		- But the number of observations is large => some observations will be predicted only once or even not at all
		- But the number of features is large =>Some features (theoretically) be missed in all subspaces used -> They could not be selected.
			- This is an extreme case because the selection of features (subspace) occurs at each node of the trees
- `max_features`: 
	- has strong effect on performance.
	- Influences the diversity of trees in the forest
	- The default values are:
		- For regression => $log_2 n$
		- For classification => $\sqrt{n}$
- `max_depth`:
	- controls the depth of each tree:
	- default is `None` => Splits until all leaves are pure
	- Could be used for regularization