# The Perceptron Algorithm

## Perceptrons as Logical Operators

![[Pasted image 20220212211910.png]]


![[Pasted image 20220212212717.png]]

XOR -> Multilayer perceptron
![[Pasted image 20220212213214.png]]

#### Quiz: Build an XOR Multi-Layer Perceptron

- Now, let's build a multi-layer perceptron from the AND, NOT, and OR perceptrons to create XOR logic!

- The neural network below contains `3` perceptrons, A, B, and C. The last one (`AND`) has been given for you. The input to the neural network is from the first node. The output comes out of the last node.

- The multi-layer perceptron below calculates `XOR`. Each perceptron is a logic operation of `AND`, `OR`, and `NOT`. However, the perceptrons A, B, and C don't indicate their operation. 
- In the following quiz, set the correct operations for the perceptrons to calculate XOR.

And if we introduce the **NAND** operator as the combination of **AND** and **NOT**, then we get the following two-layer perceptron that will model **XOR**. That's our first neural network!

![[Pasted image 20220212214043.png]]

## Perceptron trick

```python
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_ratec
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

```

## Related notes
- [[Neural Networks and Deep Learning]]
- [[W4 - NeuralNetworks Highlights]]
- [[NN_Weigth_initialization]]