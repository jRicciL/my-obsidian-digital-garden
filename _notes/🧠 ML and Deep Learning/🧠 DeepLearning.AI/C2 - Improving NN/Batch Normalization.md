---
---

# Batch Normalization
#Batch-normalization 
***

## Highlights

- Proposed in 2015 for Sergey Ioffe and Christian Szegedy, to adress:
	- ==Primarily developed== to deal with #vanishing / #exploding gradient problems.
		- Allowing training deeper networks
		- Even allowing more saturating activation function such as $thanh$ or $sigmoid$.
		- The networks are less sensitive to [[Weight Initialization]]
		- Greater `learning_rate` values could be used.
		- Batch Normalization acts as a ==Regularizer==
	- The problem that **the distribution of each's layers changes during training**
	- Makes the choice of hyperparameters more easier
- The technique consists of adding an operation to the model just before the **activation function** =>
	- Zero-centering and standard scaling
		- a. `mu_B`: Mean of $B$ (mini-batch)
		- b. `sigma_B`: Standard deviation of $B$ (mini-batch)
			- In practice the $bias$ term is replaced by the $\beta$ parameter
		- ==During test or inferring==: These values are computed using [[Exponentially Weighted Averages]]
- Each layer includes ==two new parameters== that have to be learned:
	1. `gamma`: Scale
	2. `beta`: Offset
	- ✨ This values are learned because at testing or inferring there are no mini-batch to compute the mean ($\mu$) or the standard deviation ($\sigma$).
- **Mean** and **variance** are ==controlled== by $\gamma$ and $\beta$ to avoid `mean = 0` and `std = 1`, which will affect the activation function.
- ❌ It requires more computation even during predictions.
	- It is slow during the first `epoch`
	- It is slow at predicting

### Intuition
- Can we **normalize the values of a hidden layer** ($a^{[l]}$) to train the weights of the next layer ($W^{[l+1]}$)  faster.
	- In practice $Z^{[l]}$ is normalized instead of a^{[l]}$

#### Normalizing inputs $X$
1. $\mu = \frac{1}{m} \sum_i x_i$ <- Get the mean
2. $X = X - \mu$ <- Center the data
3. $\sigma^2 = \frac{1}{m}\sum_i (x_i - \mu)^2$ <- Get the variance
4. $\hat X = X / \sigma$ <- Scale the data using the $\sigma$

### Implementing Batch Norm

1. $\mu_B = \frac{1}{m_B} \sum_i^{m_B} x_i$ 
3. $\sigma^2_B = \frac{1}{m_B}\sum_i^{m_B} (x_i - \mu_B)^2$ 
4. $\hat x_i =  x_i - \mu_B / \sqrt{\sigma_B^2 + \epsilon}$ 
5. $z_i = \gamma \hat{x}^i + \beta$

- $\mu_B$ is the empirical mean (of the batch), evaluated over the whole mini-batch $B$
- $\sigma_B$ is the empirical standard deviation, also evaluated over the whole mini-batch
- $m_B$ is the number of instances in the mini-batch
- $\hat x_i$ is teh zero-centered and normalized input
- $\gamma$ is the scaling parameter for the layer
- $\beta$ is the shifting parameter (offset) for the layer
- $\epsilon$ is a tiny number to avoid division by zero (*smothing term*)
- $z_i$ is the output of the BN operations: => scaled and centered version of the inputs.
	- This value, $\mathbf{z}^{[l]}$,  is used to compute $\mathbf{a}^{[l]}$
- ==NOTE:== $\gamma$ and $\beta$ are learnable parameters of the model
	- They are also learned along with $W$ and $b$ by using [[Optimizers#Gradient Descent]]
	- They effect is the following:
		- $\gamma => \sqrt{\sigma^2 + \epsilon}$
		- $\beta => \mu$

### Application
##### Application for layer $l$ with mini-batch $\mathbf{B}$:

$$\mathbf{X_B} \xRightarrow{{W^l, b^l}} \mathbf{Z_B} \xRightarrow[\mathbf{BN}]{\gamma^l, \beta^l} \mathbf{\hat Z_B} \xRightarrow{\phi^l} \mathbf{a_B}^{[l]}$$

- $\gamma$: BN parameters
- $\beta$: BN parameters

### Why it works?

#### Covariance Shift problem
- #Covariance-shift =>
	- From the perspective of the hidden layers, their input layers are changing all the time during training.
		- Because the previous parameters change
	- ![[Captura de Pantalla 2021-09-01 a la(s) 11.39.17.png]]
- With #Batch-normalization weights are more robust to changes
	- It reduces the amount of distribution of the previous values due to the normalization step
	- Because their distribution remains (it does not change too much)
	- Thus, each layer learn a little more ==independently== by itself

#### Regularization Effect
- Because the `mean` and `variance` are computed from each mini-batch $B$, it adds a little noise to the network => ==Regularization effect==
	- With bigger batch sizes the regularization effect decreases
- However, its primary use is not **Regularization**

#### Before or after the activation function?

<div class="rich-link-card-container"><a class="rich-link-card" href="https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">r/MachineLearning - [D] Batch Normalization before or after ReLU?</h1>
		<p class="rich-link-card-description">
		93 votes and 30 comments so far on Reddit
		</p>
		<p class="rich-link-href">
		https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
		</p>
	</div>
</a></div>

