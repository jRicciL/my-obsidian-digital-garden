# Support Vector Machines

### Margin Maximization
- Consider the number of miss-classifications 
- While increase the wide of the margin

#### Goal
1. Maximize the wide of the margin
2. Minimize the number of miss-classified samples

## SVM ERROR
- **Error** => `Classification Error` + `Marging Error`
![[Captura de Pantalla 2022-02-13 a la(s) 19.33.16.png]]


### 1. ==Classification Error==
- Compared to a [[Logistic Regression]] or to a [[Perceptron Algorithms - Notes Udacity]]
	- The `SVM` **draws a margin** instead of a single hyperplane.
	- Penalizes those samples that fall inside the margin
		- Thus, tries to keep the margin empty
	- Thus #SVM is more focused in **samples that are closer to the decision boundary**

### 2. ==Margin Error==

![[Captura de Pantalla 2022-02-13 a la(s) 19.21.59.png]]
- **Marging Error**
	- Is the norm of the vector $W$ squared
	- It is equivalent to the $L2$ regularization term -> [[Regularization]]

## The $C$ parameter

- The `C` *hyperparameter*
	- Is related to the `Classification Error`
	- Provides flexibility to the model by **regularizing the size of the margin**
	- $C$:
		- 🔴  ==Large C== => Focus on classifying points
		- 🔵  ==Small C== => Focus on a large margin
		
![[Captura de Pantalla 2022-02-13 a la(s) 19.36.34.png]]


## Polynomial Kernel
![[Captura de Pantalla 2022-02-13 a la(s) 19.45.19.png]]

- ==Add new dimensions== by combining the features and use a linear model to create a margin in that high dimension space.
- The `degree` of the Polinomial Kernel is a **hyperparameter**

![[Pasted image 20220213193835.png]]
> `x^2 + y^2`


## Radial basis function Kernel

- Use radial basis function.
- Uses the `gamma` hyperparameter:
	- **Large** `gamma` => Closer boundaries / Narrow curve
		- Large variance
		- ==Overfit==
	- **Small** `gamma` => Small boundaries / Wider curve
		- Large bias
		- ==Underfit==
- `gamma` is inversally related to `sigma` from a gaussian curve.
	- $\gamma = \frac{1}{2\sigma ^ 2}$
	
![[Captura de Pantalla 2022-02-13 a la(s) 19.52.25.png]]
![[Captura de Pantalla 2022-02-13 a la(s) 19.53.03.png]]

## SVM with `sklearn`

```python
from sklearn.svm import SVC
model = SVC()
model.fit(X, y)
```

#### Hyperparameters

- `C` => The $C$ parameter.
	- tells the SVM optimization how much you want to avoid misclassifying each training example
	- Large `C` => smaller-margin (less missclassification)
		- Overfit is possible
	- Small `C` => larger margin (more misclassification allowed)
		- Underfit is possible
- `kernel` => 'linear', 'poly', or 'rbf'
- `degree` => If the kernel is polynomial, it is the maximum degree of the monomials in the kernel
- `gamma` => Gamma parameter if kernel is `rbf`
	- Large `gamma` => Overfitting
	- Small `gamma` => Underfit

### Exercise

```python
# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
model = SVC(kernel='rbf', gamma=27)

# TODO: Fit the model.
model.fit(X,y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
```

## Related Notes
- [[SVM Recap - Udacity]]