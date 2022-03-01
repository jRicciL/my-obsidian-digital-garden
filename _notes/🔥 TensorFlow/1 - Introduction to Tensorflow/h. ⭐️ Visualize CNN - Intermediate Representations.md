# Visualize CNN - Intermediate Representations

```python
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output 
					  for layer in model.layers[1:]]

# Visualization of the model # Define a new visualization model getting the 
# fitted layers of the trained model
visualization_model = tf.keras.models.Model(
						inputs = model.input, 
						outputs = successive_outputs)

# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) 
				   for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) 
				   for f in train_human_names]

# Load an image using keras
img_path = 'path/to/image'
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name 
			   for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  	if len(feature_map.shape) == 4:
		# Just do this for the conv / maxpool layers, not the fully-connected layers
		n_features = feature_map.shape[-1]  # number of features in feature map
		# The feature map has shape (1, size, size, n_features)
		size = feature_map.shape[1]
		
		# We will tile our images in this matrix
		display_grid = np.zeros((size, size * n_features))
		
		# Postprocess the feature to be visually palatable
		for i in range(n_features):
			  # Postprocess the feature to make it visually palatable
			  x = feature_map[0, :, :, i]
			  # Standardize
			  x -= x.mean()
			  x /= x.std()
			  x *= 64
			  x += 128
			  x = np.clip(x, 0, 255).astype('uint8')
			  # We'll tile each filter into this big horizontal grid
			  display_grid[:, i * size : (i + 1) * size] = x
			
		#----------------- 
		# Display the grid
		#-----------------
		scale = 20. / n_features
		plt.figure(figsize = (scale * n_features, 
							  scale))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, 
				   aspect = 'auto', 
				   cmap = 'viridis')
```

## Plot accuracies and loss

```python
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')
```