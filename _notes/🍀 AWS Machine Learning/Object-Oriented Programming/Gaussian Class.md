---
---

# Gaussian Class

Build a python package to analyze Gaussian distributions:
- Able to add two gaussians together

### Gaussian distribution formulas
#### Probability density function
- $\mu$ => Mean
- $\sigma^2$ => Variance

$$
f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e \frac{- (x - \mu)^2}{2\sigma^2}
$$

#### Binomial distribution formulas
##### Mean
$$\mu = n * p$$

- *In other words, if a fair coin has a probability of a positive outcome p=0.5, then the mean of `n = 20` events of flipping the coin will be:*
- $\mu = n * p = 20 * 0.5 = 10$

##### Variance
$$\sigma^2 = np(1-p)$$
- *continuing with the example above =>* `sigma^2 = 20*0.5(1-0.5) = 0.25`

#### Probability density function
$$
f(k, n, p) = \frac{n!}{k!(n - k)!}p^k (1 - p)^{(n-k)}
$$

- When finding the probabilities using a continuous distribution, ==the probability of obtaining an exact value is zero.==

# How the Gaussian Class works
- See the Jupyter notebook inside the folder

# Magic Methods

#MagicMethods

<div class="rich-link-card-container"><a class="rich-link-card" href="https://www.tutorialsteacher.com/python/magic-methods-in-python" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://www.tutorialsteacher.com/assets/images/fbshare.jpg')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h3 class="rich-link-card-title">Magic or Dunder Methods in Python</h1>
		<p class="rich-link-card-description">
		Learn what is magic methods in Python and how to implement magic methods in your custom classes.
		</p>
		<p class="rich-link-href">
		https://www.tutorialsteacher.com/python/magic-methods-in-python
		</p>
	</div>
</a></div>



-> ==Magic methods== => used for overwritting default python behavior

### The `__init__` method
- It is a magic method that is used as a constructor

### The `__add__` method 
- `__add__` is called which gets called when we add two numbers using the `+` operator.

```python
num = 10
num + 5

# The same as
num.__add_(5)
```

- The implementation with our `Gaussian()` class:
```python
def __add__(self, other):
	
	'''
	Function to add together two Gaussian distributions
	Args:
		other (Gaussian): Gaussian instance
	Returns:
		Gaussian: Gaussian distribution
	'''
	
	result = Gaussian()
	# Creates a new Gaussian object
	result.mean = self.mean + other.mean
	result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
	
	return result
```

### The `__repr__` method
- To get called by built-int repr() method to return a machine readable representation of a type
```python
def __repr__(self):
	return f"Mean {self.mean}, stdev {self.stdev}"
```

- Used when the object is called
```python

```

### The `__str__` method
- It is overridden to return a printable string representation of any user defined class.
```python
num = 12
str(num)
> '12'

# the same as
int.__str__(num)
> '12'
```