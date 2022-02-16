# Trainig and Tuning

## Types of errors

### Underfitting => Oversiplify
- Does not well in the ==Training set== (neither in the `validation`/`test` set)
	- Error due to ==bias==


![[Captura de Pantalla 2022-02-12 a la(s) 22.03.55.png]]


### Overfitting => Overcomplicate
- Does well in the `training` set (because of memorization) and bad in the `validation`/`test` set.
	- Does not ==generalize== well
	- Fails with the development/validation set
	- **Error due to variance**


![[Captura de Pantalla 2022-02-12 a la(s) 22.04.54.png]]

***

## Trade-off and Model complexity


![[Captura de Pantalla 2022-02-12 a la(s) 22.07.40.png]]

- Use the results from the `training` and the `test`/`validation` to identify ==Bias== or ==Variance==.
![[Captura de Pantalla 2022-02-12 a la(s) 22.08.42.png]]

==High Bias==
- **Low** `training` and testing `performance`

==High Variance==
- **High** performance on `trianing`, but **bad** performance on `testing`

From the point of view of the amount of error:
![[Pasted image 20220212221243.png]]

***

# Cross-Validation

- **Training set** => Data used to train the model
- **Validation set** => Data used to make decisions on the models
- **Test set** => Data used to test the model

## K-Fold Cross Validation
![[Captura de Pantalla 2022-02-12 a la(s) 22.18.18.png]]

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits = 4
		  # Use Shuffle to randomize the data before the split
		   shuffle = True
		  )

for train_indices, test_indices in kf:
	pass
```

## Learning Curves

- A learning curve shows the **validation** and **training score** of an estimator for **varying numbers of training samples**.
	- Determines how much data is required to reduce the error (improve the model performance).
- It is a tool to find out *how much we benefit from adding more training data* and whether the estimator suffers more from a variance error or a bias error.

#Learning-curves display the Model Performance (or Amount of Error) in the $y$ axis, as a function of the ==size== of the `training` set

![[Captura de Pantalla 2022-02-12 a la(s) 22.25.41.png]]

-> `Sklearn` ==training curves==

![[Pasted image 20220212222629.png]]

### Code for `LearningCourves`
#Learning-curves 

```python
X2, y2 = randomize(X, y)

def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")


    plt.legend(loc="best")

    plt.show()
```

***

# Hyperparameter Tuning
Related notes:
- [[Hyperparameter Tuning]]
- [[Hyperparameter_tunning_in_python]]
- [[9. W3 - Adjusting the learning rate dynamically]]

## Grid Search
1. Import `GridSearchCV`
2. Select the parameters
3. Create a Scorer
4. Create a `GridSearch` object with the parameters and the scorer
5. Get the best estimator

```python
from sklearn.model_selection import GridSearchCV

# Select the set of parameters
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# Create a scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)

# Get the best estimator
best_clf = grid_fit.best_estimator_

```

#### Exercise

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier(random_state=42)

# TODO: Create the parameters list you wish to tune.
parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]}

def calculate_F1_Score(parameters):
    # TODO: Make an fbeta_score scoring object.
    scorer = make_scorer(f1_score)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method.
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    # TODO: Fit the grid search object to the training data and find the optimal parameters.
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator.
    best_clf = grid_fit.best_estimator_

    # Fit the new model.
    best_clf.fit(X_train, y_train)

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(X_train)
    best_test_predictions = best_clf.predict(X_test)

    # Calculate the f1_score of the new model.
    print('The training F1 Score is', f1_score(best_train_predictions, y_train))
    print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

    # Plot the new model.
    plot_model(X, y, best_clf)
    
    # Let's also explore what parameters ended up being used in the new model.
    print(best_clf)

#----------------------------------------------#

# Call the function
calculate_F1_Score(parameters)

```