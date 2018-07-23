.. _activation_functions:

====================
Activation Functions
====================

.. contents:: :local:



ELU
===

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

- It avoids and rectifies vanishing gradient problem.
- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.

.. rubric:: Cons

- One of its limitation is that it should only be used within Hidden layers of a Neural Network Model.
- Some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. Simply saying that ReLu could result in Dead Neurons.
- In another words, For activations in the region (x<0) of ReLu, gradient will be 0 because of which the weights will not get adjusted during descent. That means, those neurons which go into that state will stop responding to variations in error/ input ( simply because gradient is 0, nothing changes ). This is called dying ReLu problem.
- The range of ReLu is [0, inf). This means it can blow up the activation.

.. rubric:: Further reading

- `Deep Sparse Rectifier Neural Networks <http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf>`_ Glorot et al., (2011)
- `Yes You Should Understand Backprop <https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b>`_, Karpathy (2016)


.. _activation_leakyrelu:

LeakyReLU
=========

LeakyRelu is a variant of ReLU. Instead of being 0 when :math:`z < 0`, a leaky ReLU allows a small, non-zero, constant gradient :math:`\alpha` (Normally, :math:`\alpha = 0.01`). However, the consistency of the benefit across tasks is presently unclear. [1]_

+-------------------------------------------------------+------------------------------------------------------+
| Function                                              | Derivative                                           |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z) = \begin{Bmatrix} z & z > 0 \\              |       R'(z) = \begin{Bmatrix} 1 & z>0 \\             |
|       \alpha z & z <= 0 \end{Bmatrix}                 |       \alpha & z<0 \end{Bmatrix}                     |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/leakyrelu.png                       | .. image:: images/leakyrelu_prime.png                |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: leakyrelu                            |      :pyobject: leakyrelu_prime                      |
+-------------------------------------------------------+------------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/



.. rubric:: Pros

- Pro 1

.. rubric:: Cons

- Con 1

.. rubric:: Further reading

- `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/pdf/1502.01852.pdf>`_, Kaiming He et al. (2015)


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


.. _activation_tanh:

Tanh
====

Tanh squashes a real-valued number to the range [-1, 1]. It's non-linear. But unlike Sigmoid, its output is zero-centered.
Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity. [1]_ 

+-----------------------------------------------------+-----------------------------------------------------+
| Function                                            | Derivative                                          |
+-----------------------------------------------------+-----------------------------------------------------+
| .. math::                                           | .. math::                                           |
|      tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}|      tanh'(z) = 1 - tanh(z)^{2}                     |
+-----------------------------------------------------+-----------------------------------------------------+
| .. image:: images/tanh.png                          | .. image:: images/tanh_prime.png                    |
|       :align: center                                |       :align: center                                |
|       :width: 256 px                                |       :width: 256 px                                |
+-----------------------------------------------------+-----------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py | .. literalinclude:: ../code/activation_functions.py |
|       :pyobject: tanh                               |       :pyobject: tanh_prime                         |
+-----------------------------------------------------+-----------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Pros

- Pro 1

.. rubric:: Cons

- Con 1


Softmax
=======

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


.. rubric:: References

.. [1] http://cs231n.github.io/neural-networks-1/
