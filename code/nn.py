import math
import numpy as np

def relu(z):
  return max(0,z)

def feed_forward(x, Wh, Wo):
  # Hidden layer
  Zh = x * Wh
  H = relu(Zh)

  # Output layer
  Zo = H * Wo
  output = relu(Zo)
  return output

# Zh - Hidden layer weighted input
# h = Hidden layer activation
# Zo - Output layer weighted input
