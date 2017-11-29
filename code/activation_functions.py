import math
import numpy as np

### Note ###

# z is weighted input


### Functions ###

def relu(z):
  return max(0, z)

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))



### Derivatives ###

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def relu_prime(z):
  return 1 if z > 0 else 0
