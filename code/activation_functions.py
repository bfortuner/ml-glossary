import math
import numpy as np

def relu(z):
  return max(0, z)

def relu_prime(z):
  return 1 if z > 0 else 0

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))
