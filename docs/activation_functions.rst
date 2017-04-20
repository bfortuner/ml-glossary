.. _activation_functions:

====================
Activation functions
====================

.. toctree::
  :maxdepth: 1
  :titlesonly:

ELU
===

Be the first to contribute!


LeakyReLU
=========

Be the first to contribute!


ReLU
====

A recent invention which stands for Rectified Linear Units. The formula is deceptively simple: :math:`max(0,z)`. Despite its name and appearance, it’s not linear and provides the same benefits as Sigmoid but with better performance.

.. math::

  R(z) & = max(0,z) \\

::

  def relu(z):
    if z > 0:
        return z
    return 0

**Derivative**

The derivative of relu...

.. math::

  R'(z) & = \begin{Bmatrix}
  1 & z>0 \\
  0 & z<0 \\
  \end{Bmatrix}

::

  def relu_prime(z):
    if z > 0:
      return 1
    return 0


Sigmoid
=======

There are many types of activation functions to choose from, but one of the most popular among textbook-writers is the logistic sigmoid function. Sigmoid takes in a real value and outputs another value between 0 and 1. It’s easy to work with and has all the nice properties above: it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

.. math::

  S(z) = \frac{1} {1 + e^{-z}}

::

  def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


**Derivative**

.. math::

  S'(z) = S(z) * (1 - S(z))

::

  def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


Softmax
=======

Be the first to contribute!


Tanh
====

Be the first to contribute!
