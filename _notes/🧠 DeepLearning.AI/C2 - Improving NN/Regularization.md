---
---

# Regularization

## DeepLearning.AI => Regularization

### L2-Regularization
- It is a #Weight-decay method -> Weights end up smaller
- It is added to the cost function and to the backpropagation function.
- It is the standard way to avoid overfitting is called **L2 regularization**
- It makes the decision boundary smoother. If `lambda` is too large => _**High bias**_

- ==Regularized **cost function** (*cross-entropy cost*)==:
$$J = - \frac{1}{m}\left(y^i log(a^{L(i)}) + (1 - y^i)log(1 - a^{L(i)})\right) + \frac{1}{m}\frac{\lambda}{m}\sum_l \sum_k \sum_j W_{k,j}^{[l]2}$$

**Forward ==>**:
- For each layer $l$, we compute:

$$\sum_k \sum_j W_{k,j}^{[l]2}$$

- With python: 
```python
np.sum(np.square(Wl))
```

**Backward <== **:
$$\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$$

- With python (for `W3`, layer `3`): 
```python
dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd * W3) / m
```

### Dropout
- Specific to *Deep learning* => It randomly shuts down some neurons in each **iteration**.
- It is ==only used during training==.
- Apply dropout both during forward and backward propagation.
- The dropped units do not contribute to the training in both the forward and backward propagations of the iteration.
- ==Normalization== => The remaining neurons need to be divided by `keep_prob` to assure that the reuslt of the cost will still have the same expected value as without drop-out.
- It is not applied to the `input` neither the `output` layers.
- The idea behind:
	- Train a different model, at each iteration, that uses only a subset of the neurons/units.
	- Thus, neurons become less sensitive to the activation of one other specific neuron.

Dropout ==implementation==:

**Forward ==>**:
$$A^{[l]} * D^{[l]},$$
- where $D^{[l]}$ is a mask of `0`s and `1`s with the same dimensions as $A^{[l]}$
- Then divide $A^{[l]}$ by `keep_prob`

```python
Z1 = np.dot(W1, X) + b1
A1 = relu(Z1)
D1 = np.random.rand(A1.shape[0], A1.shape[1])
D1 = (D1 < keep_prob).astype(int)
A1 = A1 * D1
A1 = A1 / keep_prob
```

**Backward <== **:
- The same neurons turned off out in Forward must be turned off in Backward.
	- Use the same $D^{[l]}$ mask over $\partial A^{[l]}$
	- Divide $\partial A^{[l]}$ by `keep_prob`