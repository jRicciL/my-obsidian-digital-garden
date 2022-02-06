---
---

# Multiclass Classification

## Softmax
#softmax => ==*Normalized exponential*==
- It is a generalization of the  logistic regression.
- $C$ is the number of classes.
- The output layer $L$ has $C$ units, each for each class.
	- Each unit represents the Bayesian probability of been of class $C_j$ given the input $x_i$
	- The output value is a vector of dimension $C$
	- The predicted value $\hat y$ is computed as the class with max probability

#### Softmax function

The ==*softmax*== function:
$$\mathbf{a}^{[L]} = \frac{e^{Z^L}}{\sum_{i=1}^{C} s_i}$$

for each $\mathbf{a}^{[L]}_i:$
$$\mathbf{a}_i^{[L]} = \frac{s_i}{\sum_{i=1}^{C} s_i}$$

- where $s_i = \mathbf{exp}(Z_i)$ 

From:
$$\hat p_c = \sigma\left(\mathbf{s}(\mathbf{x})\right)_c = \frac{\mathbf{exp}(\mathbf{s}_c(\mathbf{x}))}{\sum_{i=1}^{C} \mathbf{exp}(\mathbf{s}_i(\mathbf{x}))}$$

- $C$ is the number of classes.
- $\mathbf{s}(\mathbf{x})$ is a vector containing the scores of each class for the intance $\mathbf{x}$.
- $\sigma\left(\mathbf{s}(\mathbf{x})\right)_c$ is the estimated probability that the instance $x$ blongs to class $c$ given the scores of each class for thath instance.

with **python**:
```python
from numpy import exp

def softmax(x):
	s_x = exp(a)
	sum_s_x = s_x.sum()
	softmax = s_x / sum_s_x
	# Softmax
	return softmax
```

#### Softmax *Regression*

==Softmax *Regression*== classifier prediction
$$
\begin{aligned}
\hat y = \mathop{\arg \max}\limits_{c} \ \ \sigma\left(\mathbf{s}(\mathbf{x})\right)_c \\ 
\hat y = \mathop{\arg \max} \left((\theta^c)^T \cdot \mathbf{x} \right)
\end{aligned}
$$


## Cross-entropy
### Training a softmax classifier - Loss Function

##### Understanding `softmax`

![[Captura de Pantalla 2021-09-02 a la(s) 15.51.35.png]]

##### *Loss function*
The ==*loss function*== per each class $c$ will be:
- $$\mathcal{L}(\hat y, y) = - \sum_{c=1}^{C}y_c\mathbf{log}(\hat y_c)$$

##### *Cost function*
The ==*Cost function*== takes into account the $\mathcal{L}$ of all classes, thus it will be:
- $$J(\theta) = \frac{1}{m}\sum_{i=1}^m \mathcal{L}(\hat y^{(i)}, p^{(i)})$$
- $$J(\theta) = \frac{1}{m} \sum_{i=1}^m \sum_{j=1}^C y_c^{(i)}\mathbf{log}(\hat p_j)$$


![[Captura de Pantalla 2021-09-02 a la(s) 16.16.43.png]]