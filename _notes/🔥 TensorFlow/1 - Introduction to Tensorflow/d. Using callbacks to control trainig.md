# Using Callbacks to control training 

#KerasCallbacks

## Example Callbacks
### First example
- A ==Callback== that ends the training if an specific `loss` is reached.
- Use the `Callback` class from `tf.keras.callbacks.Callback`
	- Use the `on_epoch_end` function => This function is executed every time an **epoch** ends.

```python
# Define a particular callback inherit from `Callback` class
class MyCallback(tf.keras.callbacks.Callback):
	# Use tue `on_epoch_end` function
	def on_epoch_end(self, epoch, logs = {}):
		if logs.get('loss') < 0.4:
			print("\nLoss is low so cancelling trainig")
			self.model.stop_training = True
```

- ==Metrics== are obtained from the `logs.get()` method parsed thought the `on_epoch_end()`
- Training stops with the `self.model.stop_training` method.
	- 🚨 This waits until the epoch ends, not the batch

### Second example
- Stop training after a given `accuracy` is reached.
	- We can provide an specific value when the callback is instantiated
	- We will provide our parameter `loss_value`

```python
class myCallback(tf.keras.callbacks.Callback): 
	
	def __init__(self, loss_value = 0.85):
		self.loss_value = loss_value
		
	
	def on_epoch_end(self, epoch, logs = {}): 
		if(logs.get('val_accuracy') > self.loss_value): 
			print("The model has reached an accuracy of 0.85")
			print("Stoping training") 
			# Stop the training using `stop_training`
			self.model.stop_trianing = True
```

## Callback implementation

```python
# Instatniate the callback
my_callback = MyCallback(loss_value = 0.9)

# Create the model
model = ...

# Fit the model using the callback
model.fit(
	training_images,
	training_labels,
	epochs = 30,
	callbacks = [my_callback]
)
```


## Resources


<div class="rich-link-card-container"><a class="rich-link-card" href="https://keras.io/api/callbacks/" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://keras.io/img/logo-k-keras-wb.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Keras documentation: Callbacks API</h1>
		<p class="rich-link-card-description">
		Keras documentation
		</p>
		<p class="rich-link-href">
		https://keras.io/api/callbacks/
		</p>
	</div>
</a></div>



<div class="rich-link-card-container"><a class="rich-link-card" href="https://keras.io/guides/writing_your_own_callbacks/" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://keras.io/img/logo-k-keras-wb.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Keras documentation: Writing your own callbacks</h1>
		<p class="rich-link-card-description">
		Keras documentation
		</p>
		<p class="rich-link-href">
		https://keras.io/guides/writing_your_own_callbacks/
		</p>
	</div>
</a></div>


