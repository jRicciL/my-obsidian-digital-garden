---
---

# Exponentially Weighted Averages

- Key component of several Optimization methods
	- #ADAM 
	- #RMSProp 
	
- Compute the local average => Moving average
$$V_t = \beta V_{t-1} + (1 - \beta)\theta_t$$

- with $\theta$ been the new value in the time series.
- $V_t$ is the average of the current time/step $t$ 

![[Captura de Pantalla 2021-08-27 a la(s) 21.16.22.png]]

- $\beta$ modifies the weight of the current $t$ value respect to the previous value.
	- ==Increasing $\beta$==: A larger `beta` gives a lot of weight to the previous values, and small weight to the current value `theta`
	- Decreasing $\beta$ will create more oscillation within plot.
- The larger the momentum 𝛽β is, the smoother the update, because it takes the past gradients into account more. But if 𝛽β is too big, it could also smooth out the updates too much.

### Understanding Exponentially Weighted Averages

```python
V_theta = 0
for i in range(len(theta)):
	V = b*V_theta + (1 - b)*theta[i]
```
- This implementation takes very little memory

#### Bias correction
- When #EWA is employed $V_0 = 0$ which affects the initial values of the averages
- The ==Bias correction==
	$$V_t = \frac{V_t}{1 - \beta^t}$$
	
$\beta^t$  will decrease as $t$ increases

## Highlights
- #ExponentialMovingAverage #EWA ->
	- Puts more weight and significance on the most recent data points.
	- Reacts more significantly to recent changes than a simply #MovingAverage