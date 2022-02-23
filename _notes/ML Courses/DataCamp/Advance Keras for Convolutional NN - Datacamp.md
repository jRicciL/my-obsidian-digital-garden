---
---

# Convolutional Neural Networks with Keras

#keras
#convolutional_networks
#pooling

## Image Processing with Neural Networks

### Classifying images

#### One-hot encoding 
- #One-hot-encoding
- Multi-class classification

### Classification with #Keras

```python
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

# Add a first hidden layaer
model.add(Dense(
	10,
	activation='relu',
	input_shape=(784,)
))
# Add a second hidden layaer
model.add(Dense(10, activation='relu'))
# Add the output layer
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

# Fit the model
# first reshape the images
train_data = train_images.reshape((50, 784))

model.fit(
	training_data,
	training_labels,
	validation_split=0.2,
	epochs=3
)

# Evaluate the model
model.evaluate(
	x=test_data, 
	y=test_labels,
	batch_size=128
)
```


## Using Convolutions

### Using convolutions

#### Using ==correlations== in images
- Pixels in most images are not independent in images

#### The convolution
- Uses a `kernel`
- Uses a `window` -> `stride`
- Multiplies the kernel rolling the window over the image

Application of a convolution -> ==Feature map==

```python
# one dimensional convolution
# The  array to be transformed
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) 
kernel = np.array([1, -1, 0]) # The kernel to be used

# The result is the convolution
conv = np.zelos_like(array)

# Output array
for ii in range(8):
	conv[ii] = (kernel * array[ii: ii + len(conv)])
```

- The following kernel finds horizontal lines in images:

```python
kernel = np.array([[-1,-1, -1], 
				   			   [  1,  1,  1],
				               [-1,-1, -1]])
```

### Implementing convolutions in Keras

- ==Covolutional== layer in #Keras

```python
from keras.layers import Conv2D

# Here we have a kernel with 3x3 cells
Conv2D(10, kernel_size=3, activation='relu')
```

- It resembles the "Dense" layer, but instead of having every unit in the layer connected to the previous layer.
- It connects the previous layer with the convolutional kernel
- ==The output== -> is a convolution of a kernel over the input.
- Convolutional layer has one weight for each cell in the kernel.

##### Integrate the convolutional layer into a network

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(
	Conv2D(
		   filters = 10, # Number of units 
		   kernel_size = 3, 
		   activation  = 'relu',
		   input_shape = (img_rows, img_cols, 1) 
		  ))

# The flatten layer serves as a conector between
# the convolution and the next dense layer
model.add(Flatten())

model.add(Dense(3, activation='softmax'))
```

![[Captura de Pantalla 2021-02-24 a la(s) 19.21.31.png]]

#### Fitting the CNN

```python
model.compile(
	          optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy']
			 )

model.fit(
	     train_data, train_labels,
		 validation_split=0.2,
		 n_split=0.2,
		 batch_size=10,
		 epoch=3)

model.evaluate(
	test_data, 
	test_labels, 
	batch_size = 10
)
```

### Tweaking your convolutions

##### Zero padding

![[Captura de Pantalla 2021-02-24 a la(s) 19.38.50.png]]

When we use **zero padding**:
- Output feature map has the same size than the input layer.

```python
model.add(
	Conv2D(10, kernel_size=3,
		 activation='relu',
		 input_shape=(nrows, ncols, 1)),
	# Padding
	padding='valid')
```

- `padding`
	- `valid`: No zero padding
	- `same`: Zero padding applied so the output and the input has the same size.

##### Strides
- Step size of the window to apply the kernel
- It determines whether the kernel will skip over some of the pixels oas it slides along the image.
	- it affects the size of the output

```python
model.add(
	Conv2D(10, kernel_size=3,
		 activation='relu',
		 input_shape=(nrows, ncols, 1)),
	# Strides (default == 1)
	strides=2)
```

##### Dilated convolutions

![[Captura de Pantalla 2021-02-25 a la(s) 7.22.59.png]]

- Useful in cases where you want to aggregate information across multiple scales

```python
model.add(
	Conv2D(10, kernel_size=3,
		 activation='relu',
		 input_shape=(nrows, ncols, 1)),
	# Dilation
	dilation_rate=2)
```

#### Calculate the size of the output

$$O = ((I - K + 2P) / S) ) + 1$$

where:
- $I$: the size of the input
- $K$: the size of the kernel
- $P$: size of the zero padding
- $S$: strides

## Going Deeper

> Networks with more convolution layers are called *deep* networks -> 
> -> More power to fit more complex data
> ---> Because their ability to create hierarchical representations of the data.

==Why do we want deep networks?==
- Different layers and kernels tend to respond to a different set of ==features==
- Intermediate layers tend to respond to **more complex features**

==How deep?==
- Depth comes at a computational cost
- May require more data

### How many parameters?

The number of parameters to be fitted by the CNN. 
- Convolutional layers don't necessarily reduces the number of parameters


##### Counting the number of parameters

Assuming the images has the following dimensions: `28x28`

```python
model.add(Conv2D(
	10,
	kernel_size=3,
	activation='relu',
	input_shape=(28, 28, 1),
	padding='same' # zero padding
))
```

==Convolution 1:==
> parameters = $((3*3) * 10^{kernel}) + 10^{bias}$
> = 100

```python
model.add(Conv2D(
	10,
	kernel_size=3,
	activation='relu',
	input_shape=(28, 28, 1),
	padding='same' # zero padding
))
```

==Convolution 2:==
> parameters = $10^{prev} * ((3*3) * 10^{kernel}) + 10^{bias}$
> = 910

```python
model.add(Flatten())
```

==Flatten:==
> No parameters = 0

```python
model.add(Dense(
	3, activation='softmax'
))
```

==Output==
> parameters = $10^{prev} * (28*28)^{input} * 3^{units} + 3^{bias}$

### Pooling operations

- One of the challenges is the large number of parameters

==Pooling operations==:
- Aggregation operations that work on small grid regions.
- Often added between convolutional layers.
- Are applied at each feature map.
- It allows to reduce the size of the feature map --> Produces another feature map.

##### Implementing max pooling

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPool2D

model.add(Conv2D(...))

# Apply the Pooling
# Specify the size of the pooling window 2x2 
model.add(MaxPool2D(2))

model.add(Conv2D(...))
```


## Understanding and improving Deep CNN

#### [[Learning Curves]]
- #Overfitting

```python
import matplotlib.pytplot as plt

training = model.fit(
	train_data, train_labels, epochs=3,
	validations_split=0.2
)

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()
```

![[Captura de Pantalla 2021-02-25 a la(s) 19.45.10.png]]

#### Storing the optimal parameters

- A keras #callback **checkpoint** that stores the best parameters (those that minimize the ==validation loss==) at each epoch.
- -> It overwrites the weights whenever the validation loss decreases

```python
from keras.callbacks import ModelCheckpoint

# This checkpoint will store the model parameters in a file
checkpoint = modelCheckpoint(
	'weights_file.hdf5',
	monitor='val_loss',
	save_best_only=True
)

# Store in a list to use it during training
callbacks_list = [checkpoint]

# Fit the model and use the callback
model.fit(
		train_data, train_labes,
		validation_split=0.2,
		epochs=3,
		callbacks=callbacks_list
)
```

#### Loading stored weights
- Instantiate a model with the same architecture, configuration, and hyperparameters.

```python
model.load_weights('weights_file.hdf5')
```


### Regularization
#Regularization

#### Dropout
1. #Dropout (nitish Srivastava, 2014):
	- At each learning step:
	- --> Select a ==random== subset of the units (neurons) on a layer
	- --> Ignore it in the **forward** pass
	- --> Ignore it in the **back-propagation** error

![[Captura de Pantalla 2021-02-26 a la(s) 11.09.31.png]]

- It allow us to  ==train many different networks== on different parts of the data.
- Each time the network trained is **randomly choose.**
- --> Therefore, if some part of the networks fit noise, the other parts will help to regularize that overfit 
- ---> Also prevents that some neurons become overly correlated in their activity.

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D /
Flatten
from keras.layers import Dropout

model = Sequential()
model.add(
	Conv2D(5, kernel_size=3,
		  	activation='relu',
		    input_shape=(x, y, 1)
		  )
)
# Drop 25% of neurons in the first layer
model.add(Dropout(0.25))
```

#### Batch Normalization

- Proposed by Sergey Loffe and Christian Szegedy in 2015.
- It ==rescales== the outputs of a particular layer:
	- Normalize the outputs -> $\mu = 0; \sigma = 1$
- **Solves the problem** when ==different batches== have ==wildly distributions of outputs== in a given layer.
- ---> Which could affect the training process
- ---> #Batch-normalization tends to make learning faster.

```python
...
from keras.layers import BatchNormalization

model = ...
model.add(BatchNormalization())
```

##### Warning ‼️

🚨 Sometimes #Dropout  and #Batch-normalization ==do not work well together==

- Dropout slows down learning.
- Batch normalization makes learning faster.

> *"Disharmorny of batch normalization and dropout"*
	
	
### Interpreting the Model 🧠

Many efforts are being made to improve the interpretability of deep neural networks.
- one way to interpret models is to examine the properties of the kernels in the convolutional layers.

##### **How to take apart a trained Neural Network?**

- In `keras`, the layers of the model can be accessed by the `model.layers` attribute.
- The weights of each layer are accessible through: `model.layers[0].get_weights()`
- Access to the kernel by: `kernel = model.layers[0].get_weights()[0]`
- That will return an array -> `[3,3, 0, 5]`, where:
	- the first two dimension represent the kernel size.
	- the second represent the channel.
	- the last dimension (`5`) indicate the number of kernels.
- We can use the kernel and apply a `convolution` function to the training images to see what patters the kernel has learn to identify.

![[Captura de Pantalla 2021-02-26 a la(s) 11.39.55.png]]


## Next Steps

- [[Residual Networks]]
- [[Transfer Learning]]
- [[Fully Convolutional Networks]]
- [[Generative Adversarial Networks]]