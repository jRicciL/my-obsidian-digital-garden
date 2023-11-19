---
---

# Neural Networks highlights

- Weights = Model coefficients
- Multilayer perceptron
- `ReLu` is the default activation function in sklearn
- Activation function affect the decision bundaries of the model

- NN in sklearn

```python
from sklearn.neural_network import MLPClassifier

# Two hidden layer, 10 and 10 units
nnclf = MLPClassifier(
	hidden_layer_sizes = [10, 10],
	solver = 'lbfgs'
).fit(X_train, y_train)
```

## Deep Learning
- Feature engineering can be part art and part science.
	- Some times is more important that even the model choice
	- Algorithms to extract the features automatically.

##### The key of deep learning
**=> Performs automatic feature extraction.**

- Deep learning architectures combine a sophisticated automatic feature extraction phase with a supervised learning phase
	- ==feature extraction==
	- This process usually requires a hierarchy of multiple feature extraction

## Resources


<div class="rich-link-card-container"><a class="rich-link-card" href="https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://2.bp.blogspot.com/-gSIgoqSR1sY/WLjHshWv0eI/AAAAAAAABnY/jEMd9ybBnawy3yVuc5loR9-Zl01zM_RtQCLcB/s1600/image01.png')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Assisting Pathologists in Detecting Cancer with Deep Learning</h1>
		<p class="rich-link-card-description">
		Posted by Martin Stumpe, Technical Lead, and Lily Peng, Product Manager A pathologist’s report after reviewing a patient’s biological tissue...
		</p>
		<p class="rich-link-href">
		https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html
		</p>
	</div>
</a></div>

