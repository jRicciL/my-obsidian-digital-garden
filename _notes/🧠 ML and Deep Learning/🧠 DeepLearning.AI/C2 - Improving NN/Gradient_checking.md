---
---

# Gradient Checking

### Highlights
- Implement gradient checking to verify the accuracy of your backprop implementation
- Instead of using the analytical derivative we will use the numerical definition.
	- Then we compare two results (Analytical and Numeric) expecting than the difference ---> 0

#### How does Gradient Checking work?
- #Backpropagation computes the gradients $\partial J / \partial \theta$, where $\theta$ denotes the parameters of the model.
- Derivative (gradient) definition => `Grad_approx`
$$\frac{\partial J}{\partial \theta} = 
\underset{\epsilon \rightarrow 0}{lim} \frac{J(\theta + epsilon) - J(\theta - epsilon)}{2 \epsilon}$$

  
![[1Dgrad_kiank.png]]

![[NDgrad_kiank.png]]
- Key computation steps.

## Implementation
### Gradient with escalar values

- First compute "gradapprox" using the formula above (1) and a small value of $\varepsilon$. Here are the Steps to follow:
    1. $\theta^{+} = \theta + \varepsilon$
    2. $\theta^{-} = \theta - \varepsilon$
    3. $J^{+} = J(\theta^{+})$
    4. $J^{-} = J(\theta^{-})$
    5. $gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}$
- Then compute the gradient using backward propagation, and store the result in a variable "grad
- Finally, compute the relative difference between "gradapprox" and the "grad" using the following formula:
$$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} \tag{2}$$
You will need 3 Steps to compute this formula:
   - 1'. compute the numerator using `np.linalg.norm(...)`
   - 2'. compute the denominator. You will need to call `np.linalg.norm(...)` twice.
   - 3'. divide them.
- If this difference is small (say less than $10^{-7}$), you can be quite confident that you have computed your gradient correctly. Otherwise, there may be a mistake in the gradient computation. 

### Gradient check for gradient vectors
For each `i` in `num_parameters`:
- To compute `J_plus[i]`:
    1. Set $\theta^{+}$ to `np.copy(parameters_values)`
    2. Set $\theta^{+}_i$ to $\theta^{+}_i + \varepsilon$
    3. Calculate $J^{+}_i$ using to `forward_propagation_n(x, y, vector_to_dictionary(`$\theta^{+}$ `))`.     
- To compute `J_minus[i]`: do the same thing with $\theta^{-}$
- Compute $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$

Thus, you get a vector gradapprox, where gradapprox[i] is an approximation of the gradient with respect to `parameter_values[i]`. You can now compare this gradapprox vector to the gradients vector from backpropagation. Just like for the 1D case (Steps 1', 2', 3'), compute: 
$$ difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } \tag{3}$$

**Note**: Use `np.linalg.norm` to get the norms


### Python implementation
```python
# Numerical computation of the gradient
theta_plus = theta + epsilon
theta_minus = theta - epsilon
J_plus = forward_propagation(x, theta_plus)
J_minus = forward_propagation(x, theta_minus)
# Compute the gradient approx
gradapprox = (J_plus - J_minus) / (2 * epsilon)

# Gradient computed analitically
grad = backward_propagation(x, theta)

# Compute the difference between two values
numerator = np.linalg.norm(grad - gradapprox)
denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
difference = numerator / denominator
```