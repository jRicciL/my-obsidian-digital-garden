---
---

# DeepLearning Frameworks

### Criteria for choosing
- Ease of programming
- Running speed
- Truly open

## Tensorflow

#### Optimize a cost function $J(w)$
- The basic structure of a *Tensorflow* program:

```python
import numpy as np
import tensorflow as tf

# Define a Variable
w = tf.Variable(0, dtype = tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# Define a training function
def train_step():
	# Only perform the forward prop step with TF
 	with tf.GradientTape() as tape:
    	cost = w ** 2 - 10 * w + 25
	  # Perform the optimization step
	trainable_variables = [w]
	grads = tape.gradient(cost, trainable_variables)
	optimizer.apply_gradients(
		zip(grads, trainable_variables))
```

- Perform the optimization step:
```python
# RUN the train step process
for i in range(1000):
	train_step()
print(w)
```
```python
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.000001>
```

- The function:
```python
cost = lambda w: w ** 2 - 10 * w + 25
ws = np.arange(-10, 15)
costs = cost(ws)

plt.plot(ws, costs)
```

![[Captura de Pantalla 2021-09-02 a la(s) 16.45.16.png]]

- #Tensorflow takes charge of the *backward propagation* step.
	- #Tensorflow automatically will compute the derivatives from the *cost function*
- `GradiaentTape` records the order of the sequence of operations needed to compute the cost function (==Forward process==)

#### Optimize a cost function $J(x, w)$
- The cost function depending of the data $X$
- We will also see another implementation of the optimization:

```python
# The cost function depending on the data

# Parameters
w = tf.Variable(0, dtype = tf.float32)
# The data X
x = np.array([1.0, -10, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

# Define the cost function J(x, w)

def cost_fn():
  return x[0] * w ** 2 + x[1] * w + x[2]

print(w)
# Perform one step of optimization
optimizer.minimize(cost_fn, [w])
print(w)
```

- For the whole training process:
```python
# Parameters
w = tf.Variable(0, dtype = tf.float32)
# The data X
x = np.array([1.0, -10, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

print(w)
# Define the cost function J(x, w)
def training(x, w, optimizer, n_steps = 1000):
  def cost_fn():
    return x[0] * w ** 2 + x[1] * w + x[2]
  for i in range(n_steps):
    optimizer.minimize(cost_fn, [w])

# Perform one step of optimization
training(x, w, optimizer, n_steps = 1000)
print(w)
```
```python
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.000001>
```