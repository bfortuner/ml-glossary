import math
import numpy as np

# Neural Network w Matrices

INPUT_LAYER_SIZE = 1
HIDDEN_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 2

def init_weights():
    Wh = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * \
                np.sqrt(2.0/INPUT_LAYER_SIZE)
    Wo = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * \
                np.sqrt(2.0/HIDDEN_LAYER_SIZE)


def init_bias():
    Bh = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
    Bo = np.full((1, OUTPUT_LAYER_SIZE), 0.1)
    return Bh, Bo

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    '''
    Z - weighted input matrix

    Returns gradient of Z where all
    negative values are set to 0 and
    all positive values set to 1
    '''
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z

def cost(yHat, y):
    cost = np.sum((yHat - y)**2) / 2.0
    return cost

def cost_prime(yHat, y):
    return yHat - y

def feed_forward(X):
    '''
    X    - input matrix
    Zh   - hidden layer weighted input
    Zo   - output layer weighted input
    H    - hidden layer activation
    yHat - output layer predictions
    '''

    # Hidden layer
    Zh = np.dot(X, Wh) + Bh
    H = relu(Zh)

    # Output layer
    Zo = np.dot(H, Wo) + Bo
    yHat = relu(Zo)
    return yHat

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


