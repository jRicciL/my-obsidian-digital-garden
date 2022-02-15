# Model Evaluation Metrics

## `train_test_split`
- Testing the models

```python
# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Import the train test split
# http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html
from sklearn.cross_validation import train_test_split

# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# Use train test split to split your data 
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                random_state = 42)

# Instantiate your decision tree model
model = DecisionTreeClassifier()

# TODO: Fit the model to the training data.
model.fit(X_train, y_train)

# TODO: Make predictions on the test data
y_pred = model.predict(X_test)

# TODO: Calculate the accuracy and assign it to the variable acc on the test data.
acc = accuracy_score(y_test, y_pred)
```

## Confusion Matrix

![[Captura de Pantalla 2022-02-09 a la(s) 20.30.19.png]]

- The usual layout of a #ConfusionMatrix

|          | Guessed Positive | Guessed Negative |
| -------- | ---------------- | ---------------- |
| Positive | TP               | FN               |
| Negative | FP               | TN               |


### Type 1 and Type 2 Errors
The classic example to understand Type 1 and Type 2 Errors

<mark style='background-color: #9CE684 !important'>False Positives</mark> =>
- **Type Error 1**: In the medical example this is when we misdiagnose a healthy patient as sick.
<mark style='background-color: #FFA793 !important'>False Negatives</mark>
- **Type Error 2**: Labeling as Negative a Positive case -> Misdiagnose as sick a healthy patient

#### Exercise - Confusion Matrix
![[Captura de Pantalla 2022-02-09 a la(s) 20.33.15.png]]

> How many True Positives, True Negatives, False Positives, and False Negatives, are in the model above? Please enter your answer in that order, as four numbers separated by a comma and a space. For example, if your answers are 1, 2, 3, and 4, enter the string `1, 2, 3, 4`. Remember, in the image above the blue points are considered positives and the red points are considered negatives.
> ==Answer== => `6, 5, 2,,1`

## Accuracy
- *How many samples did we classify correctly?*
	- Consider only $TP$ and $TN$
	- Ratio between the number of correctly classified points and the total number of paoints
$$\mathbf{Accuracy} = \frac{TP + TN }{TP + TN + FP + FN} = \frac{TP + TN}{N}$$

```python
from sklearn.metrics import accuracy_score
```

#### When accuracy wont work

- For ==imbalanced== datasets

## Precision and Recall

#### Precision
<mark style='background-color: #FFA793 !important'>Recall</mark>
- Try to identify as `TRUE POSITIVES` all of the positives in the dataset
	- Find **all** the sick people
- In a medical example -> A `False Negative` is worse
	- So a higher ==Recall== is needed

> *How many of the POSITIVES PREDICTED are really POSITIVES?*

$$\mathbf{Precision} = \frac{TP}{TP + FP}$$

##### Exercise

![[Pasted image 20220209211841.png]]

> What is the precision of the linear model above? Please write the number as a decimal, like **0.45** or as a fraction, like **3/5**.
> ==Answer== -> 6 /8 = 0.75

#### Recall
<mark style='background-color: #93EBFF !important'>Precision</mark>
- Try to maximize the number of `TRUE POSITIVES` inside the samples predicted as POSITIVES
	- In the spam example -> a `False Positive` is worse
		- So a higher precision is needed
		- It is not necessary to find all positives

> *How many of POSITIVES PREDICTED did I found from all possible POSITIVES inside the dataset?*
> 
> *Out of all POSITIVES, how many did we correctly classify as POSITIVES?*

$$\mathbf{Precision} = \frac{TP}{TP + FP}$$


##### Exercise

![[Pasted image 20220209212131.png]]

> What is the recall of the linear model above? Please write your number as a decimal, like **0.45** or as a fraction, like **3/5**.
> ==Answer== -> 0.857

## F1-score
- Combination ==Recall== and ==Precisiton==
- `F1-score` is the <mark style='background-color: #FFA793 !important'>Harmonic Mean</mark>
	- It is always lower than the arithmetic mean
	- Penalizes more lower values


$$\mathbf{F1-score} = 2 \cdot \frac{\mathbf{Precision} * \mathbf{Recall}}{\mathbf{Precision} + \mathbf{Recall}}$$

#### Exercise

> If the Precision of the medical model is **55.6%**, and the Recall is **83.3%**, what is the F1 Score? (Please write your answer as a percentage, and round it to 1 decimal point.)
> ==Answer== -> 66.7

```python
def F1_score(P, R):
	return 2 * (P*R)/(P+R)
```

## F$\beta$-score
- A more general version is `Fb-score` -> $F\beta - score$
	- $\beta$ determines how much the metric is skewed towards ==Precision==
	- <mark style='background-color: #FFA793 !important'>The smaller is</mark> `beta` the more towards **precision**

$$\mathbf{F\beta-score} = (1 + \beta^2) \cdot \frac{\mathbf{Precision} * \mathbf{Recall}}{\beta^2 \times \mathbf{Precision} + \mathbf{Recall}}$$

#### Some highlights
- Because `beta` is squared, the minimum value is zero.
- There is not upper limit for `beta`
- When $\beta = 0$ the `F-score` is equal to **Precision**.
- When $\beta \rightarrow \infty$ the `F-score` tends to **Recall**


#### Exercise

![[Pasted image 20220209214259.png]]

> Out of the following three models, which one should have an F-beta score of 2, 1, and 0.5? Match each model with its corresponding score.
> -   Detecting malfunctioning parts in a spaceship
>     - *High recall model* => `beta = 2`
> -   Sending phone notifications about videos a user may like
> -   Sending promotional material in the mail to potential clients
>     - *High Precision model* => `beta = 0.1`

## ROC - Receiver Operating Characteristic

The ==ROC curve== takes into account the `True Positive Ratio` and the `False Postive Ration`

![[Captura de Pantalla 2022-02-09 a la(s) 22.09.03.png]]

##### True Positive Ratio
- *How many true positives among all positives?*

$$TPR = \frac{TP}{TP + FP} = TP / P$$

##### False Positive Ratio
- *How many false positives among all negatives?*

$$FPR = \frac{TN}{TN + FN} = TP / P$$

![[Captura de Pantalla 2022-02-09 a la(s) 22.16.32.png]]

#### ROC curve

![[Captura de Pantalla 2022-02-09 a la(s) 22.20.42.png]]o

## Regression Metrics
-> [[Linear Regression]]

### Mean Absolute Error
- 🔴  #MAE is not differentiable -> <mark style='background-color: #FFA793 !important'>Not used as an function to optimize</mark> with [[C4 Gradient descent]]

```python
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)
guesses = regressor.predict(X)
error = mean_absolute_error(y, guesses)
```

### Mean Square Error
- Used during gradient descent

### R2 score

-> Compares the **error** of the ==linear regression model== with the **error** for the simplest model ==Compute the mean== of $y$.

==*How much variability is explained by the model?*==
$$R^2 = 1 - \frac{SSE}{SST} = 1 - \frac{Var(e)}{Var(y)} = 1 - \frac{\sum(y_i - \hat y_i)}{\sum(y_i - \bar y_i)}$$

-> ![[Linear Regression#R-squared]]

## Exercise

| Classification | Regression |
| -------------- | ---------- |
| `Precision`     |  `R-Squared`          |
| `Recall`       |      `MAE`      |
| `F-Score`      |        `MSE`    |
| `Accuracy`     |          `RME`  |
| `ROC-AUC`      |            |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, auc, accuracy_score, fbeta_score
```

## Related Notes
- [[4. W1 - Performance Metrics]]