import math
import numpy as np

# Neural Network w Matrices

**
X  - Input matrix from training set
Zh - Hidden layer weighted input matrix
Zo - Output layer weighted input matrix
Bh - Hidden layer bias matrix
Bo - Output layer bias matrix
H  - Hidden layer activation matrix
yHat  - Output layer predictions
**

# Initialize Weights
Wh = np.random.randn(inputLayerSize, hiddenLayerSize) * \
            np.sqrt(2.0/inputLayerSize)
Wo = np.random.randn(hiddenLayerSize, outputLayerSize) * \
            np.sqrt(2.0/hiddenLayerSize)

# Initialize Biases
Bh = np.full((1, hiddenLayerSize), 0.1)
Bo = np.full((1, outputLayerSize), 0.1)


def relu(Z):
    return np.maximum(0, Z)

def feed_forward(X):

    # Hidden layer
    Zh = np.dot(X, Wh) + Bh
    H = relu(Zh)

    # Output layer
    Zo = np.dot(H, Wo) + Bo
    yHat = relu(Zo)
    return yHat

def relu_prime(Z):
    **
    Z - weighted input matrix
    Returns the gradient of the
    Z matrix where all negative
    values are switched to 0 and
    all positive values switched to 1
    **
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z

def cost(yHat, y):
    cost = np.sum((yHat - y)**2) / 2.0
    return cost

def cost_prime(yHat, y):
    return yHat - y

def backprop(X, y, lr):

    yHat = feed_forward(X)

    # Layer Error
    Eo = (yHat - y) * relu_prime(Zo)
    Eh = np.dot(Eo, Wo.T) * relu_prime(Zh)

    # Cost derivative for weights
    dWo = np.dot(H.T, Eo)
    dWh = np.dot(X.T, Eh)

    # Cost derivative for bias
    dBo = np.sum(Eo, axis=0, keepdims=True)
    dBh = np.sum(Eh, axis=0, keepdims=True)

    # Update weights
    Wo -= lr * dWo
    Wh -= lr * dWh

    # Update biases
    Bo -= lr * dBo
    Bh -= lr * dBh
