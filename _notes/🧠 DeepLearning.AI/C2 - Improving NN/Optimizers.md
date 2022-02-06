---
---

# Faster Optimizers
[[Mathematics for Machine Learning]] and [[Neural Networks and Deep Learning]]

Ways to optimize training:
- Apply a good initialization strategy for the weights
- Using a good activation function
- Using batch normalization
- Reuse parts of pretrained networks

![[Captura de Pantalla 2021-08-29 a la(s) 19.24.10.png#center]]

## Gradient Descent
- Also known as the *steepest=descent method*
	- The gradient of the ==Loos function is used to make parameter updates==
	- It does not always point to the best direction when steps of finite size are used.
	- #Ill_conditioning: When the partial derivatives of the loss are very different with respect to to the different optimization variables.
- It updates the weights $\theta$ by directly subtractiong the gradient of the cost function $J(\theta)$ with regard to the weights ($\nabla_{\theta}J(\theta)$) multiplied by the learning rate $\alpha$.
- $\alpha$ is constant.

#### The Algorithm
$$x_{t+1} = x_t - \nabla_x f(x_t)$$ 

#### DeepLearning.AI version
Implement the gradient descent update rule. The  gradient descent rule is, for $l = 1, ..., L$: 
$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{1}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{2}$$

***

## Mini-batch Gradient Descent
#### Batch and Stochastic Gradient Descent
- Allows to take multiple Gradient Descent steps per epoch.
- `Loss` does not necessarily decreasses with each `iteration`
	- Looks noisy
![[Captura de Pantalla 2021-08-27 a la(s) 19.33.17.png]]
- The size of the *mini-batch* is another hyperparameter.
	1. A mini-batch of size `m` => ==Batch Gradient Descent==
		- Too long for each iteration
	1. If mini-batchs of size =`1` => ==Stochastic Gradient Descent==
		- It is extremely noisy
		- Lose the speed provided by the vectorization

Typical mini-batch sizes:
- `64`, `128`, `256`, `512`
- Shuffling and Partitioning are the two steps required to build mini-batches


***
## Stochastic Gradient Descent
- A special case of Mini-batch were the `batch-size = 1`.
- #SDG requires 3 for-loops in total:
	1. Over the number of iterations
	2. Over the $m$ training examples
	3. Over the layers (to update all parameters, from $(W^{[1]},b^{[1]})$ to $(W^{[L]},b^{[L]})$)


***
## Momentum Optimization
- #MDG: Proposed by Boris Polyak in 1964
- Also known as ==Momentum-Based Learning==
	- Comparing with Vanilla Gradient Descent #MGD does care about what early gradients were.
- Based on computing the [[Exponentially Weighted Averages]] of the gradients to update at the new step.
	- Formally, this will be the exponentially weighted average of the gradient on previous steps.
- Momentum takes into account the past gradients to smooth out the update.
- The larger the momentum 𝛽β is, the smoother the update, because it takes the past gradients into account more. But if 𝛽β is too big, it could also smooth out the updates too much.

#### The Intuition
- 🔶 **==Adapt the direction==**
- To move in an **averaged** direction of the last few steps => the zigzagging will be smother
- The leaning is ==accelerated==:
	- *It is moving in a direction that often points closer to the optimal.*
	- *Larger steps could be used*
- It gives greater preference to *consistent* directions.
- 🚨 Could cause the solution to slightly overshoot the direction of the momentum.
	- ==Overshooting== is desirable to avoid local minima.
	- The *momentum parameter* controls this behavior.
- Due to momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times.

#### The algorithm:
> At each iteration, subtracts the ==local gradient== form the *momentum vector* **m**, multiplied by the learning rate $\alpha$, and updates the weights by adding this momentum vector.
- The gradient is used for **acceleration**, not speed.
- $\eta$ => To simulate ==Friction==, and prevent the momentum from growing too large.
	- Should e between 0 and 1 -> Typically 0.9
	- It avoids the accumulation effect of the previous gradients over the most recent.


#### The formulas

- The ==direction== ($\bar V$ or $p_t$) is computed as:

	1. $$\bar V \leftarrow \eta \bar V - \alpha \frac{\partial J(\bar W)}{\partial \bar W} $$
	2. $$\bar W \leftarrow \bar W + \bar V$$

	which is similarly to:
	1. $$p_t \leftarrow \eta p_t - \nabla f(x)$$
	2. $$x_t \leftarrow x_t + \alpha p_t$$

	and to:
	1. $$m \leftarrow\beta m - \eta \nabla_{\theta}J(\theta)$$
	2. $$\theta \leftarrow \theta + m$$

- $\eta$ is a friction or smoothing parameter (*momentum parameter*): $\eta \in (0,1)$ => a value of 0.9 is the most recommended.

![[Captura de Pantalla 2021-08-28 a la(s) 13.22.52.png]]

##### Python implementation
```python
def MGD(theta, grad, gd_params, f_params):
	 n_iter = gd_params['n_iter']
	 alpha = gd_params['alpha']
	 eta = gd_params['eta']
	 # inizialize the p values at t=0 with zeros
	 p_t = np.zeros(theta.shape)
	 Theta = []

	 for t in range(n_iter):
		 g_t = grad(theta, f_params = f_params)
		 # update the direction
		 p_t = eta * p_t - g_t
		 # update the parameter
		 theta = theta + alpha * p_t
		 Theta.append(theta)
			
	 return np.array(Theta)
```

#### DeepLearning.AI implementation

Momentum
- On each interation $t$:
	- Compute `dW`, `db` on current mini-batch:
	- Update the derivatives by the following:
		- `VdW = b*VdW + (1 - b)*dW`
		- `Vdb = b*Vdb + (1 - b)*db`
		- ==NOTE==: The term `1 - b` is commonly omited in the literature.
	- Update the weights: 
		- `W = W - a*VdW`
		- `b = b - a*Vdb`

🟠 => $\beta$ = `friction`
🟠 => $VdW$ = `Velocity`
🟠 => $(1 - \beta)*dW$ = `Acceleration`

***

## *Nesterov* Momentum

- **Nesterov Accelerated Gradient** #NAG => By Yurii Nesterov 1983
- A modification of the traditional #GDMomentum.
- Works only in [[mini-batch gradient descent]] with modest batch sizes.
- It measures the gradient of the cost function, but not at the local position, but slightly ahead in the direction of the **momentum**
- The only difference with #GDMomentum is that the gradient is measured at $\theta + \beta m$ => $\frac{\partial J(\bar W + \eta \bar V)}{\partial \bar W}$

#### The intuition
- 🔶  **==Adapt the direction==**
- It keeps the momentum but avoids overshooting => The only difference is where the gradient is computed.
- The gradients are computed with respect to  $\bar x_t$ and not with respect to $x_t$:
	- $\bar x_t$ is the poing that would be reached after executing the momentum of the current point $x_t$.
		- The gradient is computed using this point.
	- $x_t$ is the current point.


#### The formulas

- The ==direction== ($\bar V$, $p_t$ or $m$) is computed as:

	1. $$\bar V \leftarrow \eta \bar V - \alpha \frac{\partial J(\bar W + \eta \bar V)}{\partial \bar W} $$
	2. $$\bar W \leftarrow \bar W + \bar V$$

	which is similarly to:
	1. $$p_t \leftarrow \eta p_t - \nabla f(x - \alpha p_t)$$
	2. $$x_t \leftarrow x_t + \alpha p_t$$

	and to:
	1. $$m \leftarrow\beta m - \eta \nabla_{\theta}J(\theta + \beta m)$$
	2. $$\theta \leftarrow \theta + m$$

- $\eta$ is a friction or smoothing parameter (*momentum parameter*): $\eta \in (0,1)$ => a value of 0.9 is the most recommended.


![[Captura de Pantalla 2021-08-28 a la(s) 13.40.20.png]]

***
# Parameter-Specific Learning Rates

### [[Learning Rate]] Decay
- A constant learning rate is not desirable.
	- **Low value**: the algorithm will take too long to converge to the optimal solution
	- **High value**: The algorithm could came close to the optima, but then it will keep oscillating around the optimal solution without reaching it.

#### Decay Functions
There are two main decay functions, *Exponential decay* and *Inverse decay*. ==The value of $\alpha_t$ can be expressed in terms  of the initial decay rate  $\alpha_{0}$ and epoch $t$ as follows:==
**Exponential decay**:
$$\alpha_t = \alpha_0 \ exp(-k\cdot t)$$

**Inverse decay**:
$$\alpha_t = \frac{\alpha_0}{1 + k\cdot t}$$

The parameter $k$ controls the rate of the decay.

**Step decay**: The learning rate is reduced by a particular factor every few [[Epoch]]s.
- E.g. $\alpha$ can be multiplied by 0.5 every 5 epochs.

## Delta-bar-Delta Algorithm
- It is an early heuristic approach to adapt individual learning rates for model parameters during training.
	- ==Tracks the sign of the partial derivatives== of a given parameter $\theta$
	- If the sign is consistent (does not change) -> ==**Increase**== the learning rate for that parameter.
	- If the sign changes -> Decrease the learning rate for that parameter.
- 🚨 Should not be used with [[Stochastic Gradient Descent]]
	- Because the errors can be magnified.

***
## ADAGrad
- **Adagrad** scales down the gradient vector along the steepest dimensions => The change in the #learning-rate is proportional to the gradient.
	- The parameters with the largest $\partial$ have correspondingly rapid decrease in their learning rate.
- As a result => Greater progress in the more gently sloped directions of the parameter space.

#### The intuition
- It individually adapts the learning rates of all model parameters by scaling them ->
	- Inversely proportional to the square root of the sum of all the historical squared values of the gradient.
	- The learning rate will be -> $\alpha / \sqrt{A_i + \epsilon}$
	- Where $\sqrt{A_i + \epsilon}$ is a kind of **=="signal-to-noise" normalization==**
- 🚨 **Drawbacks**:
	- Absolute movements along all components will tend to slow down =>
		- Because $A_i$ will become grater and greater if the sign of the partial does not change.
	- This result in a premature and excessive decrease in the effective learning rate.
		- **Stops to early when training** #ANN -> Not recommended 🥶

#### The formulas
- The aggregation of the squared value of the partial derivatives with respect to the parameter $\theta_i$:
	- $$A_i \leftarrow A_i + (\frac{\partial}{\partial \theta_i}J)^2$$
	the same as:
	- $$G^t_i = \sum^t_{k=0} (\frac{\partial}{\partial x_i}J)^2$$
- The update for the $ith$ parameter ($\theta_i$ or $x_i$) is as follows:
	- $$\theta_i \leftarrow \theta_i - \frac{\alpha}{\sqrt{A_i + \epsilon}}\left( \frac{\partial}{\partial \theta_i}J \right)$$
	the same as:
	- $$x_{t+1} = x_t - \frac{\alpha}{\sqrt{G^t_i + \epsilon}} \cdot p_t$$
	where $p_t$ could be computed by #GD or a momentum variant. And $\epsilon$ is a small constant to avoid ill-conditioning => $10^{-8}$.

#### The algorithm

***
## RMSProp -> Root Mean Square Prop

- *RMSProp* algorithm fixes the **AdaGrad** slowing down by accomulating only the gradients from the most recent iterations.
- It uses [[Exponentially Weighted Averages]] in the first step (AdaGrad does not use this).
- Originally proposed in a Coursera Course for Jeff Hinton
- The ==scaling== effect provided allows to use a larger `learning_rate` value.

#### Tensorflow
```python
optimizer = keras.optimizers.RMSprop(
	lr = 0.001, rho = 0.9
	# rho is the decay rate (beta2)
)
```

#### The intuition
- #RMSProp accumulates only the gradients from the most recent iterations.
	- *Not all the gradients since the beginning of training as #AdaGrad does.*

#### The formulas

- Aurelien Geron version:
	1. $s \leftarrow \beta_2 s + (1 - \beta_2)\nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta)$
	2. $\theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{s + \epsilon}$

#### DeepLearning.AI implementation

Momentum
- On each interation $t$:
	- Compute `dW` ($\partial W$), `db` ($\partial b$) on current mini-batch:
	- Compute $S_{\partial W}$
		- $S_{dW} = \beta S_{dW} + (1 - \beta)(dW \otimes dW)$
		- $S_{db} = \beta S_{db} + (1 - \beta)(db \otimes db)$
		- ==NOTE==: The term `1 - b` is commonly omited in the literature.
		- ==NOTE==: Here, $\beta$ is the `decay rate`
	- Update the weights: 
		- $W = W - \alpha \frac{dW}{\sqrt {S_{dW} + \epsilon}}$
		- $b = b - \alpha \frac{db}{\sqrt {S_{db} + \epsilon}}$ <- *This scales the update by *
		- $\epsilon = 10^{-7}$ => A very small value that provides numerical 



***
## ADAM
#ADAM -> Adaptive Moment Estimation
		-	D. P. Kingma and J. L. Ba., *Adam: a Method for Stochastic Optimization*. In procc. ICLR 2015, 1–13 (2015)	
		
#### Tensorflow
```python
optimizer = keras.optimizers.Adam(
	lr = 0.001, beta_1 = 0.9, beta_2 = 0.999
)
```
		
#### The intuition
- Combines ideas of #MGD and #RMSProp
	- Adapting both the direction and the step size of the gradient.
- Like #MGD : 
	- It keeps track of an exponentially decaying average of past gradients.
- Like #RMSProp :
	- Keeps track of an exponentially decaying average of past squared gradients.
- It isvery useful with combined with #SDG (Stochastic Gradient Descent) and for #Batch-GD strategies.

**Adam algorithm**
1. $m \leftarrow \beta_1 m - (1 - \beta_1)\nabla_{\theta}(\theta)$
2. $s \leftarrow \beta_2 s = (1 - \beta) \nabla J(\theta) \otimes \nabla J(\theta)$
3. $\hat m \leftarrow m / (1 - \beta_1^t)$
4. $\hat s \leftarrow s / (1 - \beta_2^t)$
5. $\theta \leftarrow \theta  + \eta \cdot\hat m \oslash \sqrt{\hat s + \epsilon}$

#### DeepLearning.AI implementation

ADAM
- Initialize:
	- $V_{dW} = 0$, $S_{dW} = 0$
	- $V_{db} = 0$, $S_{db} = 0$ 
- On each interation $t$:
	- Compute $dW$: `dW`,  $db$: `db` on current mini-batch:
	- Update the derivatives by the following:
		- ==Momentum component update==, with $beta_1$ as the `friction`:
			- $V_{dW} = \beta_1 V_{dW} + (1 - \beta_1)dW$
			- $V_{db} = \beta_1 V_{db} + (1 - \beta_1)db$
			- ==NOTE==: The term `1 - b` is commonly omitted in the literature.
		- ==RMSprop component update==, with $beta_2$ as the `decay rate`:
			- $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)(dW \otimes dW)$
			- $S_{db} = \beta_2 S_{db} + (1 - \beta_2)(db \otimes db)$
		- ==Bias correction== (as with [[Exponentially Weighted Averages]]) of $V$ and $S$:
			- Momentum component:
				- $V_{dW}^{corrected} = V_{dW} / (1 - \beta_1^t)$
				- $V_{db}^{corrected} = V_{db} / (1 - \beta_1^t)$
			- Decay component:
				- $S_{dW}^{corrected} = S_{dW} / (1 - \beta_2^t)$
				- $S_{db}^{corrected} = S_{db} / (1 - \beta_2^t)$
	- Update the weights (as with #RMSProp but using the `corrections`): 
		- $W = W - \alpha \frac{V_{dW}^{corrected}}{\sqrt {S_{dW} + \epsilon}}$
		- $b = b - \alpha \frac{V_{db}^{corrected}}{\sqrt {S_{db} + \epsilon}}$
		- 
##### Hyperparametrs choice:
🟠 => $\alpha$ = #learning-rate needs to be tune
🟠 => $\beta_1$ = `friction` = $0.9$
🟠 => $\beta_2$ = `decay rate` = $0.999$

#### Another version of ADAM
**How does Adam work?**
1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction). 
2. It calculates an exponentially weighted average of the squares of the past gradients, and  stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction). 
3. It updates parameters in a direction based on combining information from "1" and "2".

The update rule is, for $l = 1, ..., L$: 

$$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases}$$
where:
- t counts the number of steps taken of Adam 
- L is the number of layers
- $\beta_1$ and $\beta_2$ are hyperparameters that control the two exponentially weighted averages. 
- $\alpha$ is the learning rate
- $\varepsilon$ is a very small number to avoid dividing by zero


# Learning Rate Decay
## Learning Rate Scheduling
- Slowly reduce the learning rate ($\alpha$ or $\eta$)
- Page 359 of `Hands of Machine Learning` book. 1^[Géron, Aurélien. 2019. _Hands_-on _Machine Learning_ with Scikit-Learn, Keras and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Second edition]

![[Captura de Pantalla 2021-08-29 a la(s) 19.31.57.png]]

#### Predetermined piecewise constant learning rate
- Use a constant learning rate for a number of epochs
- Then a smaller learning rate for another number of epochs, and so on.

With #Keras:

```python
# A function than takes an `epoch` as an input
def piecewise_constant_fn(epoch): 
	if epoch < 5:
		return 0.01 
	elif epoch < 15: 
		return 0.005
	else:  
		return 0.001
	
# Create a `LearningRateScheduler` callback
lr_scheduler = keras.callbacks.\
		LearningRateScheduler(piecewise_constant_fn)

# Use the callback during training
history = model.fit(X_train_scaled, 
					y_train, [...], 
					callbacks=[lr_scheduler])
```

#### Performance scheduling
- Measure the validation error every $N$ steps
- Then reduce the learning rate by a factor $\lambda$ when the error stops dropping
	- Monitoring is similar as [[Early Stoping]]
	
With #keras:
- use the `ReduceLROnPlateau` callback.

#### Power scheduling
- Set learning rate to a function of the iteration number ($t$):
	- $\eta(t) = \eta_0 / (1 + t/s)^c$

- **Hyperparameters**:
	- Learning rate $\eta_0$
	- The power $c$ (typically set to 1)
	- The steps $s$.

With #keras:
```python
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
# Decay is the inverse of s, and c = 1
```

#### Exponential scheduling
- Set learning rate to:
	- $\eta(t) = \eta_0\times 0.1^{t/s}$
- Exponential scheduling keeps slashing $\eta$ by a factor of 10 every $s$ steps

With #keras:
```python

# Define a function that takes the current 
# epoch and returns the learning rate
def exponential_decay(lr0, s):  
	def exponential_decay_fn(epoch):
		return lr0 * 0.1**(epoch / s) 
	return exponential_decay_fn

# Get the function with specific lr and s values
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

# Create a `LearningRateScheduler` callback
lr_scheduler = keras.callbacks.\
		LearningRateScheduler(exponential_decay_fn)

# Use the callback during training
history = model.fit(X_train_scaled, 
					y_train, [...], 
					callbacks=[lr_scheduler])
```

- The `LearningRateScheduler` will update the optimizer’s `learning_rate` attribute at the beginning of each epoch.

```python
 

s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32) 
learning_rate = keras.optimizers.\
					  schedules.\
					  ExponentialDecay(0.01, s, 0.1) 
optimizer = keras.optimizers.SGD(learning_rate)
```

#### 1-cycle scheduling
- Introduced in a 2018 paper by Leslie Smith.
- It seems to perform better.
- It **starts by increasing the initial learning rate $\eta_0$
- Then, it grows linearly up to $\eta_1$ halfway through training.
- Then it decreases the learning rate linearly down to $\eta_0$
	- during the second half of training