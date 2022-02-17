---
---

# Introduction to Computer Vision

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

- ==Normalize== the values because it is easier if we trat all values as between `0` and `1`

```python
training_images = training_images / 255.0
test_images     = test_images / 255.0
```

### Create the first model
- `Sequential` => Defines a Sequence of layers in the neural network
- `Flatten` => Flats the matrix to a flat vector
- `Dense` => Adds a layer of neurons
