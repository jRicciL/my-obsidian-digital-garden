---
---

# Decision Trees

> Related notes: 
> - [[ML_with_Spark#Decision Tree]]

***

## Entropy

### Intuition

![[Captura de Pantalla 2022-02-13 a la(s) 11.52.41.png]]

- 🔴  *High knowledge* => **Low entropy**
- 🟠  *Medium knowledge *
- 🔵  *Low knowledge* => **High entropy**

![[Captura de Pantalla 2022-02-13 a la(s) 11.57.00.png]]


### Entropy formula
#Entropy ($H$) Formula:

$$H(X) = - \sum_{i=1}^n P(x_i) \cdot \mathbf{log_2}P(x_i)$$

#### Python implementation

- Formula to compute entropy given an array of probabilities.
```python
import numpy as np

def get_entropy(probs: np.array):
    entropy = - np.array([p * np.log2(p) for p in probs]).sum()
    return entropy
```

- Aux formula to compute probabilities
```python
def get_probabilities(array: np.array):
    values, counts = np.unique(array, return_counts = True)
    n = len(array)
    probs = counts / n
    return probs
```

#### Dissecting the formula

1. Turn the products into sums by using logarithms

$\mathbf{log_2}(ab) = \mathbf{log_2}(a) + \mathbf{log_2}(b)$


#### Exercises

**What is the entropy for a bucket with a ratio of four red balls to ten blue balls?**
> ==Answer==: `0.8631205`


**If we have a bucket with eight red balls, three blue balls, and two yellow balls, what is the entropy of the set of balls?**
> ==Answer==: 

```python
# Solution
a = ['R']*8 + ['B']*3 + ['Y']*2
# Compute the probabilities
probs = get_probabilities(a)
# Get the entropy
get_entropy(probs)
```

## Information Gain

![[Pasted image 20220213125148.png]]

> **Where did we gain more information? Where did we gain less?**
> 1 => `Smallest`, 2 => `Medium`, 3 => `Largest`

### Information Gain Formula
#InformationGain is calculated for a split by subtracting the *weighted entropies* of each branch ($Children$) from the original entropy ($Parent$)
- When training a #DecisionTrees using these metrics, **the best split** is chosen by ==Maximazing Information Gain==

$\mathbf{InformationGain}= \mathbf{Entropy}(Parent)− \left[ \frac{m}{m + n}\mathbf{Entropy}(Child_1) + \frac{n}{m + n}\mathbf{Entropy}(Child_2) \right]$

##### Python implementation

```python
def two_group_ent(first, tot):                        
    return -(first/tot*np.log2(first/tot) +           
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)                       
g17_ent = 15/24 * two_group_ent(11,15) +              
           9/24 * two_group_ent(6,9)                  

answer = tot_ent - g17_ent  
```

***

## Hyperparameters

### Maximum Depth
- It's the **largest possible length** between the root to a leaf.
- ⚠️ A tree of maximum length $k$ can have at most $2^k$ leafs.
![[Pasted image 20220213130433.png]]

### Minimum number of samples to split
- `min_samples_split` => If a node has less samples than this parameters it will not be split and the splitting process stops
- 🔴 However, it **does not control** the minimum samples per leaf
![[Pasted image 20220213130449.png]]

### Minimum number of samples per leaf

- `min_samples_leaf` => Can be specified as a `number` or as a `float`.
- If it's an `integer`, it's the minimum number of samples allowed in a leaf.
- If it's a float, it's the minimum percentage of samples allowed in a leaf.
- If a threshold on a feature results in a leaf that has fewer samples than `min_samples_leaf`, the algorithm will not allow _that_ split, but it may perform a split on the same feature at a _different threshold_, that _does_ satisfy `min_samples_leaf`.
![[Pasted image 20220213130906.png]]

## Hyperparameters and Overfitting

![[Captura de Pantalla 2022-02-13 a la(s) 13.12.19.png]]

-   ==Large depth== very often causes **overfitting**.
	-   since a tree that is too deep, can *memorize* the data. 
	-   **Small depth** can result in a very *simple model*, which may cause **underfitting**.
-   ==Small minimum samples== per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, **overfit**. 
-   ==Large minimum samples== may result in the tree not having enough flexibility to get built, and may result in **underfitting**.

***

## Decision Trees in `sklearn`

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)
```

#### Hyperparameters
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`

### Exercise

```python
# TODO: Train the model
model = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 5)
model.fit(X_train, y_train)
# TODO: Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# TODO: Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
```