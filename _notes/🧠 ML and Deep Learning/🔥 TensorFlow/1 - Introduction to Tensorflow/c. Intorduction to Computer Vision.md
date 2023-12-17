---
---

# Introduction to Computer Vision
- Multiclass image classification

***

- Working with #FashionMNIST
- use lots of pictures to train a model to identify between items of clothing

## The *fashion MNIST* dataset

- 70k images
- 10 categories
- images are `28x28x1` (grey scale)
	- -> 784 features (pixeles)

## Coding a computer vision Neural Network

- Import the **dataset**

```python
# Improt the dataset

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = \
	mnist.load_data()
```

- We can **plot** or inspect the data using `matplotlib` or `numpy`

```python
import numpy as np
import matplotlib.pyplot as plt

plt.imshow(training_images[0])

print(training_labels[0])
```

- ==Normalize== the values because it is easier if we treat all values as between `0` and `1`

```python
training_images = training_images / 255.0
test_images     = test_images / 255.0
```

### Create the first model
A feedforward multilayer perceptron with `Dense` layers.

##### Define the model
- `Sequential` => Defines a Sequence of layers in the neural network
- `Flatten` => Flats the matrix to a flat vector
- `Dense` => Adds a layer of neurons

```python
# The first model to evaluate FashionMNIST
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape = (28, 28)),
	tf.keras.layers.Dense(128, activation = tf.nn.relu),
	tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])
```


##### Compile the model
```python
model.compile(
	optimizer = tf.train.AdamOptimizer(),
	loss = 'sparse_categorical_crossentropy'
)
```

##### Train the model
```python
model.fit(
	training_images, training_labels, epochs = 5
)
```

##### Evaluate over the test set
```python
# Test using the test set
model.evaluate(
	test_images,
	test_labels
)
```

## Exploration Exercises

### Exercise one

-   Create a set of classifications for each of the test images, and then prints the first entry in the classifications.

```python
classifications = model.predict(test_images)
# Returns an array of probabilities for each
# of the ten labels
```

-   Represents the probability that this item is each of the 10 classes

### Exercise two

-   Implement the same model using more hidden units

### Exercise three

-   What would happen if we remove the Flatten() layer
    -   You get an error about the input shpae

### Exercise four

-   Consider the final output layers ⇒
    -   Why are ten of them?
    -   What would happen if you had a different amount than 10?
-   **Alert!!**
    -   ==The number of neurons in the last layer should match the number of classes== you are classifying for.

### Exercise five

-   Consider the effect of **additional layers** in the network.
	-   For far more complex data extra layers are often necessary.

### Exercise six

-   Consider the effect of adding more or less epochs.
    -   Be careful about overffiting.
    -   There is no point in wasting your time trianing if you aren't improving your loss.