.. _activation_functions:

====================
Activation Functions
====================

.. contents:: :local:

Introduction
============

Activation functions live inside neural network layers and modify the data they receive before passing it to the next layer. Activation functions give neural networks their power — allowing them to model complex non-linear relationships. By modifying inputs with non-linear functions neural networks can model highly complex relationships between features. Popular activation functions include :ref:`relu <activation_relu>` and :ref:`sigmoid <activation_sigmoid>`.

Activation functions typically have the following properties:

  * **Non-linear** - In linear regression we’re limited to a prediction equation that looks like a straight line. This is nice for simple datasets with a one-to-one relationship between inputs and outputs, but what if the patterns in our dataset were non-linear? (e.g. :math:`x^2`, sin, log). To model these relationships we need a non-linear prediction equation.¹ Activation functions provide this non-linearity.

  * **Continuously differentiable** — To improve our model with gradient descent, we need our output to have a nice slope so we can compute error derivatives with respect to weights. If our neuron instead outputted 0 or 1 (perceptron), we wouldn’t know in which direction to update our weights to reduce our error.

  * **Fixed Range** — Activation functions typically squash the input data into a narrow range that makes training the model more stable and efficient.

ELU
===

Be the first to contribute!


LeakyReLU
=========

Be the first to contribute!

.. _activation_relu:

ReLU
====

.. image:: images/relu.png
    :align: center

A recent invention which stands for Rectified Linear Units. The formula is deceptively simple: :math:`max(0,z)`. Despite its name and appearance, it’s not linear and provides the same benefits as Sigmoid but with better performance.

.. math::

  R(z) = \begin{Bmatrix}
  z & z > 0 \\
  0 & otherwise \\
  \end{Bmatrix}\\

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: relu

**Derivative**

The derivative of relu...

.. math::

  R'(z) = \begin{Bmatrix}
  1 & z>0 \\
  0 & z<0 \\
  \end{Bmatrix}

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: relu_prime


.. _activation_sigmoid:

Sigmoid
=======

.. image:: images/sigmoid.png
    :align: center

Sigmoid takes a real value as input and outputs another value between 0 and 1. It’s easy to work with and has all the nice properties of activation functions: it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

.. math::

  S(z) = \frac{1} {1 + e^{-z}}

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: sigmoid

**Derivative**

.. math::

  S'(z) = S(z) * (1 - S(z))

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: sigmoid_prime


Softmax
=======

Be the first to contribute!


Tanh
====

Be the first to contribute!


.. rubric:: References

.. [1] Example
