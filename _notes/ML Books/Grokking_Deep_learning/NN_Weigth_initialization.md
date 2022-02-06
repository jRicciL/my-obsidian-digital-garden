---
---

# Weight initialization
#Xavier_initialization

### Highlights 

##### Zero initialization
- Initialize to zero all the weights results in the network failing to ==**break symmetry**==.
	- --> Each neuron will learn the same thing,

##### Random Initialization
- The weights $W^l$ should be initialized randomly to break symmetry.
- However, it's is ok to initialize the biases $b^l$ to zeros. Symmetry is still broken so long as $W^l$ is initialized randomly.
- Use a ==normal distribution== helps to avoid extreme values which will converge much more slowly to the solution.
- When ==Using large values==:
	- Does not work well
	- The cost starts very high => Because in the last activation (`sigmoid`) the values are very close to 0 or one, and therefore the cost of an error is very high.
	- Can lead to **vanishing/exploding** gradients
	
	#### He initialization
	- Named for *He et al, 2015*.
	- Similar to ==Xavier initialization== but the scaling factor (of the variance) is `sqrt(2. / layers_dims[l - 1])`:
		- $$= \sqrt{2 / D^{l-1}}$$

- He initialization implementation
```python
# For each layer
for l in range(1, L + 1):
	he_scaling_factor = np.sqrt(2 / layers_dims[l - 1])
	parameters['W' + str(l)] = np.random.randn(
		layers_dims[l], 
		layers_dims[l - 1]) * he_scaling_factor
	parameters['b' + str(l)] = np.zeros(
		(layers_dims[l], 1)
)
```