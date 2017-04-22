.. _nn_concepts:

.. toctree::
  :maxdepth: 1
  :titlesonly:

=====
Terms
=====

Basic terms and concepts in neural networks


Neuron
======

A neuron takes a group of weighted inputs, applies an activation function, and returns an output.

.. image:: images/neuron.png
    :align: center

Inputs to a neuron can either be features from a training set or outputs from a previous layer’s neurons. Weights are applied to the inputs as they travel along synapses to reach the neuron. The neuron then applies an activation function to the “sum of weighted inputs” from each incoming synapse and passes the result on to all the neurons in the next layer.



Synapse
=======

Synapses are like roads in a neural network. They connect inputs to neurons, neurons to neurons, and neurons to outputs. In order to get from one neuron to another, you have to travel along the synapse paying the “toll” (weight) along the way. Each connection between two neurons has a unique synapse with a unique weight attached to it. When we talk about updating weights in a network, we’re really talking about adjusting the weights on these synapses.


Weights
=======

Explanation of weights (parameters)


Bias
====

Bias terms are additional constants attached to neurons and added to the weighted input before the activation function is applied. Bias terms help models represent patterns that do not necessarily pass through the origin. For example, if all your features were 0, would your output also be zero? Is it possible there is some base value upon which your features have an effect? Bias terms typically accompany weights and must also be learned by your model.


Layers
======

.. image:: images/neural_network_simple.png
    :align: center

.. rubric:: Input Layer

Holds the data your model will train on. Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).

.. rubric:: Hidden Layer

Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network. In traditional networks, hidden layers are typically fully-connected layers — each neuron receives input from all the previous layer’s neurons and sends its output to every neuron in the next layer. This contrasts with how convolutional layers work where the neurons send their output to only some of the neurons in the next layer.

.. rubric:: Output Layer

The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an output representing your model’s prediction.



Weighted Input
==============

A neuron’s input equals the sum of weighted outputs from all neurons in the previous layer. Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron. If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights — one for each each synapse.

**Single Input**

.. math::

  Z &= Input \cdot Weight \\
    &= X W

**Multiple Inputs**

.. math::

  Z &= \sum_{i=1}^{n}x_i w_i \\
    &= x_1 w_1 + x_2 w_2 + x_3 w_3


Notice, it’s exactly the same equation we use with linear regression! In fact, a neural network with a single neuron is the same as linear regression! The only difference is the neural network post-processes the weighted input with an activation function.



Activation Functions
====================

Activation functions live inside neurons and modify the data they receive before passing it to the next layer. Activation functions give neural networks their power — allowing them to model complex non-linear relationships. By modifying inputs with non-linear functions neural networks can model highly complex relationships between features. Popular activation functions include :ref:`relu` and :ref:`sigmoid`.

Activation functions typically have the following properties:

  * **Non-linear** - In linear regression we’re limited to a prediction equation that looks like a straight line. This is nice for simple datasets with a one-to-one relationship between inputs and outputs, but what if the patterns in our dataset were non-linear? (e.g. :math:`x^2`, sin, log). To model these relationships we need a non-linear prediction equation.¹ Activation functions provide this non-linearity.

  * **Continuously differentiable** — To improve our model with gradient descent, we need our output to have a nice slope so we can compute error derivatives with respect to weights. If our neuron instead outputted 0 or 1 (perceptron), we wouldn’t know in which direction to update our weights to reduce our error.

  * **Fixed Range** — Activation functions typically squash the input data into a narrow range that makes training the model more stable and efficient.





.. rubric:: References

- ref 1










