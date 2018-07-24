import math
import numpy as np

### Note ###

# z is weighted input


### Functions ###

def linear(z,m)
	return m*z

def elu(z,alpha)
	return z if z >= 0 else alpha*(e^z -1)

def leakyrelu(z, alpha):
	return max(alpha * z, z)

def relu(z):
  return max(0, z)

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def tanh(z):
	return (np.exp(z) - np.exp(-z)) 
	/ (np.exp(z) + np.exp(-z))




### Derivatives ###

def linear_prime(z,m)
	return m

def elu_prime(z,alpha)
	return 1 if z > 0 else alpha*e^z

def leakyrelu_prime(z, alpha):
	return 1 if z > 0 else alpha

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def relu_prime(z):
  return 1 if z > 0 else 0

def tanh_prime(z):
	return 1 - np.power(tanh(z), 2)

