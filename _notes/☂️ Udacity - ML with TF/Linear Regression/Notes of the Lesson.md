---
---

# Linear Regression Concepts

## The absolute trick
#AbsolutTrick

1. Start with a **point** $(p,q)$ an a **line** $f$
2. Get the line close to the point
3. $y=w_1x + w_2$
4. Add $p$ to $w_1$ 
5. $y = (w_1 + p)x + (w_2 + 1)$
	- Here, we add to $w_1$ and step of size $p$, which is quite great.
	- In #MachineLearning, que want to take little steps
		- We will call this step size => #learning-rate 
- ==Learning rate==: $\alpha = 0.05$
- $y = (w_1 + \alpha \cdot p)x + (w_2 + \alpha)$

## The square trick
#SquareTrick

Similar to [[#The square trick]], however, here we will subtract to $q$ the current value $q'$ of line $f$ at $f=(p, q')$:
- $y = (w_1 + p[q - q']\alpha)x + (w_2 + [q - q']\alpha)$

y = (-0.6 + (-5)(-4)(0.01)) +  (4 - (-4)(0.01))