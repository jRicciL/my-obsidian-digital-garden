---
---

# The `hello world` of Neural Networks

## A primer in machine learning

![[Pasted image 20220216224122.png]]

- **Machine Learning** -> Give examples (`Data` and `Answers`) 
	- And let the computer find the rules
	- Lots of examples with their labels (==Supervised==)

## Hello world

A Perceptron^[[[Perceptron Algorithms - Notes Udacity]]]

1. Create the model using the #Sequential API

```python
# A single unit --> Perceptron
model = keras.Sequential([
	keras.layers.Dense(
		units = 1,		  # number of Units (neurons)
		input_shape = [1] # Input shape
	)
])
```

2. **Compile** the model
- Use an #Optimizer
- Use a #LossFunction

```python
model.compile(
	optimizer = 'sdg', # Stochastic Gradient Descent
	loss = 'mean_squared_error'
)
```

3. **Fit** the model using the training set.
	- Train during 100 `epochs`

```python
model.fit(
	X_train, y_train,
	epochs = 100
)
```

4. Make **predictions** using the model

```python
y_pred = model.predict(X_test)
```

## Resources


<div class="rich-link-card-container"><a class="rich-link-card" href="https://aihub.cloud.google.com/u/0/" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('//www.gstatic.com/aihub/aihub_favicon.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">AI Hub</h1>
		<p class="rich-link-card-description">
		AI Hub: The one place for everything AI
		</p>
		<p class="rich-link-href">
		https://aihub.cloud.google.com/u/0/
		</p>
	</div>
</a></div>

