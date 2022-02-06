---
---

# The problem with Local Optima

- When training a neural network, most points of **zero gradients** are not local optima => But saddle points
	- The point where the derivatives are zero.
- A ==Saddle point== is more probable than a ==local optima== with a multidimensional space.
	1. Unlikely to get stuck in a bad local optima => a saddle point instead

![[Captura de Pantalla 2021-08-29 a la(s) 21.14.06.png]]

- ==Plateaus== can really slow the learning process
	- A region where the derivative is close to 0 for a long time.