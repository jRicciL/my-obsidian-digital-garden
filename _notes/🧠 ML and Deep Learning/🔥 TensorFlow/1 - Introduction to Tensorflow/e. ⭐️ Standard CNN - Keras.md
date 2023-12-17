# Standard ConvNN 
## Defined for Fashion MNIST

- This is a ==Standard== #CNN, as defined by #Kearas and #Tensorflow 

### Standard #CNN for multiclass classification
- Uses ==3== #CNN layers
- Uses `BatchNormalization`
- Uses `Dropout` = `0.25`
- Uses #ELU as activation function for the `Conv` layers

```python
def create_model():
	model = tf.keras.models.Sequential()
	# First Conv
	model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
	model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='elu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model.add(tf.keras.layers.Dropout(0.25))
	# Second Conv
	model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
	model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='elu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(tf.keras.layers.Dropout(0.25))
	# Third Conv
	model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
	model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='elu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model.add(tf.keras.layers.Dropout(0.25))
	
	# Classification
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(256))
	model.add(tf.keras.layers.Activation('elu'))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(10))
	model.add(tf.keras.layers.Activation('softmax'))

  return model
```

## Train the model on TPU

- Uses `Adam` optimizer
- Uses `sparse_categorical_crossentropy`
- Uses `sparse_categorical_accuracy`

```python
# Clear session
tf.keras.backend.clear_session()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
	# Instantiate the model
 	model = create_model()
 	
	# Compile the model
	model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, ),
      loss = 'sparse_categorical_crossentropy',
      metrics = ['sparse_categorical_accuracy'])

model.fit(
    x_train.astype(np.float32), y_train.astype(np.float32),
    epochs=17,
    steps_per_epoch=60,
    validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
    validation_freq=17
)
```

#### Save the model

```python
model.save_weights('./fashion_mnist.h5', 
				    overwrite = True)
```

## Function to verify predictions

- Load the weights from the `TPU` model

```python
LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 
			   'dress', 'coat', 'sandal', 'shirt', 
			   'sneaker', 'bag', 'ankle_boots']

# Create a model using the CPU session
cpu_model = create_model()
cpu_model.load_weights('./fashion_mnist.h5')
```

- Function to plot the results

```python
from matplotlib import pyplot
%matplotlib inline

def plot_predictions(images, predictions):
	n = images.shape[0]
	nc = int(np.ceil(n / 4))
	f, axes = pyplot.subplots(nc, 4)
	for i in range(nc * 4):
		y = i // 4
		x = i % 4
		axes[x, y].axis('off')

		label = LABEL_NAMES[np.argmax(predictions[i])]
		confidence = np.max(predictions[i])
		if i > n:
			continue
		axes[x, y].imshow(images[i])
		axes[x, y].text(0.5, 0.5, label + '\n%.3f' % confidence, fontsize=14)

	pyplot.gcf().set_size_inches(8, 8)  
	plot_predictions(np.squeeze(x_test[:16]), cpu_model.predict(x_test[:16]))
```