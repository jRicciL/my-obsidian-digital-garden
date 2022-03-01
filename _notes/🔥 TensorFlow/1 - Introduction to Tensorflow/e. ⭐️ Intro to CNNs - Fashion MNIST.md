# Introduction to Convolutional Neural Networks

***

## Convolutions and Pooling
- Pooling is a method for image comprehension
- ==Convolutions== are filters applied over an array of values
	- 🚨 The initial filters are not random =>
		- They start with a set of know good filters 

## Default CNN model

- Import #TensorFlow  and clear the `backend`

```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

- Import the `dataset`

```python
# Load the dataset
mnist = tf.keras.datasets.fashion_mnist

# Get the training and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Create the text labels
y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
			'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.
```

- 🚨 ==Important== 🚨: 
	- Add an **extra dimension** to add an empty color channel

```python
X_train = np.expand_dims(X_train, -1)
X_test  = np.expand_dims(X_test, -1)
```

#### Define the model
- A CNN for multiclass classification:
	- 3 convolutional layers
		- Each layer is followed by a `MaxPool2D` layer
		- ⚠️ The first layer has a `padding = 'same'`
	- A `Flatten` layer
	- A `Dense` layer with `128` units ( #ReLU activation )
	- A `Dense` layer for classification (with #softmax)

```python
N_CLASSES = 10

# NOTE: mnist fashion has only one channel, 
# therefore that axis is not used
WIDTH, HEIGHT, N_CHANNELS =  X_train[0].shape

model = tf.keras.models.Sequential(
	name = 'Basic_CNN_model',
	layers = [
	# Define the Input layer
	tf.keras.layers.Input(shape = (WIDTH, HEIGHT)),
	# Define the first Conv layer
	tf.keras.layers.Conv2D(
		filters     = 64, 
		kernel_size = (3, 3),
		activation  = 'relu',
    padding = 'same'
	),
	tf.keras.layers.MaxPool2D(
		pool_size = (2, 2),
		strides   = None,
		padding   = 'valid',
		data_format = None
	),
	tf.keras.layers.Conv2D(64, 
						   (3, 3), 
						   activation = 'relu'),
	tf.keras.layers.MaxPool2D(2, 2),
	tf.keras.layers.Conv2D(32, 
						   (3, 3),
						   activation = 'relu'),
	tf.keras.layers.MaxPool2D(2, 2),
	# Flatten
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(units = 128, activation = 'relu'),
	tf.keras.layers.Dense(N_CLASSES, 
						  activation = 'softmax')
])
```

#### Summary of the model

```python
Model: "Basic_CNN_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 64)        640       
                                                                 
 max_pooling2d (MaxPooling   (None, 14, 14, 64)        0         
 2D)                                                             
                                                                 
 conv2d_1 (Conv2D)           (None, 12, 12, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling (None, 6, 6, 64)          0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 32)          18464     
                                                                 
 max_pooling2d_2 (MaxPooling (None, 2, 2, 32)          0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense   (Dense)             (None, 128)               16512     
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 73,834
Trainable params: 73,834
Non-trainable params: 0
_________________________________________________________________
```

#### Details about the model

- The `conv2d` layer (the first `Con2D`) outputs `64` processed images from a given input image.
	- The shape of each output image is => `(26, 26)`
	- This is done through the use of `64` kernels of `shape = (3,3)`
- The `max_pooling2d` takes the output of `conv2d` and returns a new set of `64` feature "==images==", with a `13x13` dimension each.
    -   This is mainly done for image comprehension
    -   Overall, `MaxPool2D(2,2)` **quarters** the size of the image
- The last dimension of the images is `2x2` (the output of layer `max_pooling2d_3`)
	- The `Flatten` layer outputs `128` values ⇒ `3x3x32`
	- The `Dense` layers are used for the ==classification== task
		- There are $2*2*32*128 + 128 = 16,512$ parameters associated
	- The second `Dense` is used for the ==output layer== ⇒ with ten units
		- The number of units should match the number of classes inside the target variable $y$

#### Visualize the model using `visualkeras`
```python
import visualkeras

visualkeras.layered_view(model = model, scale_xy = 4)
```

![[Pasted image 20220223131651.png]]


#### Compile the model
- We will use `'sparse_categorical_crossentropy'` for #MulticlassClassification

```python
model.compile(
	optimizer = 'adam',
	loss      = 'sparse_categorical_crossentropy',
	metrics   = ['accuracy']
)
```

#### Fit the model

```python
# Convert to float32
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)

# Train the model
model.fit(
    X_train, 
    y_train, 
    validation_data = (X_test, y_test),
    epochs = 20, 
    batch_size = 32
)
```

#### Make an individual prediction
- 🔥 We need to adapt the shape of the individual image using `np.newaxis`

```python
X_test[0][np.newaxis].shape
```
```
(1, 28, 28)
```

- Also, ==remember== that we expanded the dimensions of the images

```python
# Test the model
model.predict(X_test_dims[0][np.newaxis])
```

### Continue visualizing convolutions
- [[f. ⭐️ Visualizing Convolutions]]

## Related notes
- [[1. W1 - CNN]]
- [[6. Using a CNN]]
- [[d. Using callbacks to control trainig]]
- [[Fully Convolutional Networks]]
- [[Advance Keras for Convolutional NN - Datacamp]]