.. _activation_functions:

====================
Activation Functions
====================

.. contents:: :local:



ELU
===

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


LeakyReLU
=========

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

.. _activation_relu:

ReLU
====

A recent invention which stands for Rectified Linear Units. The formula is deceptively simple: :math:`max(0,z)`. Despite its name and appearance, it’s not linear and provides the same benefits as Sigmoid but with better performance.

+-------------------------------------------------------+------------------------------------------------------+
| Function                                              | Derivative                                           |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z) = \begin{Bmatrix} z & z > 0 \\              |       R'(z) = \begin{Bmatrix} 1 & z>0 \\             |
|       0 & z <= 0 \end{Bmatrix}                        |       0 & z<0 \end{Bmatrix}                          |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/relu.png                            | .. image:: images/relu_prime.png                     |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: relu                                 |      :pyobject: relu_prime                           |
+-------------------------------------------------------+------------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Pros

- Pro 1

.. rubric:: Cons

- Con 1

.. rubric:: Further reading

- `Deep Sparse Rectifier Neural Networks <http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf>`_ Glorot et al., (2011)
- `Yes You Should Understand Backprop <https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b>`_, Karpathy (2016)


.. _activation_sigmoid:

Sigmoid
=======

Sigmoid takes a real value as input and outputs another value between 0 and 1. It’s easy to work with and has all the nice properties of activation functions: it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

+-----------------------------------------------------+-----------------------------------------------------+
| Function                                            | Derivative                                          |
+-----------------------------------------------------+-----------------------------------------------------+
| .. math::                                           | .. math::                                           |
|      S(z) = \frac{1} {1 + e^{-z}}                   |      S'(z) = S(z) \cdot (1 - S(z))                  |
+-----------------------------------------------------+-----------------------------------------------------+
| .. image:: images/sigmoid.png                       | .. image:: images/sigmoid_prime.png                 |
|       :align: center                                |       :align: center                                |
|       :width: 256 px                                |       :width: 256 px                                |
+-----------------------------------------------------+-----------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py | .. literalinclude:: ../code/activation_functions.py |
|       :pyobject: sigmoid                            |       :pyobject: sigmoid_prime                      |
+-----------------------------------------------------+-----------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Pros

- Pro 1

.. rubric:: Cons

- Con 1

.. rubric:: Further reading

- `Yes You Should Understand Backprop <https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b>`_, Karpathy (2016)


Softmax
=======

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Tanh
====

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


.. rubric:: References

.. [1] Example
