---
---

# Naive Bayes

🔥 Check [[W4 - Naive Bayes]]

***

#NaiveBayes is a supervised machine learning algorithm **based on conditional probabilities** of the input features.
- Assigns the probability distributions to each possible classes

## Bayes Theorem

- `Prior` and `Posterior` information
	- Switch from *What we know* -> *What we infer*

Knowing the probability of an event $A$, $P(A)$, and of an event $R$ giving $A$: $P(R | A)$ ->
- ==Infer== the probability of $A$ giving $R$
- $P(A | R)$
	- The probability of $A$ once we know that $R$ has occurred.

### #Bayes 

![[Captura de Pantalla 2022-02-13 a la(s) 14.10.39.png]]

#### Formula

$$P(A | R) = \frac{P(A) P(R | A)}{P(A)P(R | A) + P(B)P(R | B)}$$

### Exercise

Compute the probability of been sick of a $S$ disease given than you got a positive test $+$ => $P(S|+)$
- What we know:
	- $P(S)$: Probability of have the disease $S$
		- `1 / 10 000`
	- $P(+|S)$: Probability of give positive given that you're sick  
		- `0.99`
	- $P(+|H)$: Probability of give positive given that you're **NOT** sick: healthy $H$:
		- `0.01`

$$P(S|+) = P(S)P(+|S) / P(S)P(+|S) + P(H)P(+|H)$$

```python
# Solution using python
PS = 1 /10000
PH = 1 - PS
P_pS = 0.99
P_pH = 0.01

# Compute the P(S|+)
P_Sp = (PS * P_pS) / ((PS * P_pS) + (PH * P_pH))
P_Sp
```
```python
0.00980392156862745
```

![[Captura de Pantalla 2022-02-13 a la(s) 14.27.06.png]]

![[Captura de Pantalla 2022-02-13 a la(s) 14.33.17.png]]

***

## Naive Assumption

**We assume that the probabilities are independent**
- We assumed that ==each event occurs independently== from the rest of the events
	- Thus, we compute the probabilities for each feature **without considering the intersection** with the rest of the features
- Makes the algorithm very *fast*

### Exercise

- Suppose you have a bag with three standard 6-sided dice with face values [1,2,3,4,5,6] and two non-standard 6-sided dice with face values [2,3,3,4,4,5]. Someone draws a dice from the bag, rolls it, and announces it was a **3**. What is the probability that the die that was rolled was a standard die?

```python
N = 8 dices
P(S) = 3 / 5
P(NS) = 2 / 5

P(3 | S) = 1 /6
P(3 | NS) = 2 / 6

Question
P(S | 3) = P(S) * P(3 | S) / ((P(S) * P(3 | S)) + (P(NS) * P(3 | NS)))

= (3/5)*( 1/6) / (((3/5)*(1 /6) ) + (2/5)*(2/6))
```

Answer:
`0.42857142857142855`

## Related Notes
- [[W4 - Naive Bayes]]
- [[3. Improving Model Performance]]