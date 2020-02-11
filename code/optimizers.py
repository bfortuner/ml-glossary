import math
import numpy as np


def Adadelta(data):
    pass


def Adagrad(data):
    gradient_sums = np.zeros(theta.shape[0])
    for t in range(num_iterations):
        gradients = compute_gradients(data, weights)
        gradient_sums += gradients ** 2
        gradient_update = gradients / (np.sqrt(gradient_sums + epsilon))
        weights = weights - lr * gradient_update
    return weights


def Adam(data):
    pass


def LBFGS(data):
    pass


def RMSProp(data):
    pass


def SGD(data, batch_size, lr):
    N = len(data)
    np.random.shuffle(data)
    mini_batches = np.array([data[i:i+batch_size]
     for i in range(0, N, batch_size)])
    for X,y in mini_batches:
        backprop(X, y, lr)


def SGD_Momentum():
    pass

