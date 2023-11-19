# Visualizing Convolutions

#VisualizeConvolutions
- The model was defined in the previous notebook -> [[e. ⭐️ Intro to CNNs - Fashion MNIST]]

```python
# Visualing the Convolutions and Pooling

import matplotlib.pyplot as plt
from tensorflow.keras import models

FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IAMGE = 26
CONVOLUTION_NUMBER = 3

# Get the outputs of the model's layers in a list
layer_outputs = [layer.output 
					for layer in model.layers]
# Create an activation model,
# Which recives the model.input
# And outputs the layer_outputs
activation_model = tf.keras.models.Model(
		inputs = model.input, 
		outputs = layer_outputs
)

#
figure, axarr = plt.subplots(nrows=3, ncols = 4)
for x in range(0,4):
	  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
	  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
	  axarr[0,x].grid(False)
	  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
	  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
	  axarr[1,x].grid(False)
	  f3 = activation_model.predict(test_images[THIRD_IAMGE].reshape(1, 28, 28, 1))[x]
	  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
	  axarr[2,x].grid(False)
```

![[Pasted image 20220223172858.png]]