.. _layers:

======
Layers
======

.. contents:: :local:


BatchNorm
---------

BatchNorm accelerates convergence by reducing internal covariate shift inside each batch.
If the individual observations in the batch are widely different, the gradient
updates will be choppy and take longer to converge.

The batch norm layer normalizes the incoming activations and outputs a new batch
where the mean equals 0 and standard deviation equals 1. It subtracts the mean
and divides by the standard deviation of the batch.

.. rubric:: Code

Code example from `Agustinus Kristiadi <https://wiseodd.github.io/techblog/2016/07/04/batchnorm/>`_

.. literalinclude:: ../code/layers.py
      :pyobject: BatchNorm

.. rubric:: Further reading

- `Original Paper <https://arxiv.org/abs/1502.03167>`_
- `Implementing BatchNorm in Neural Net <https://wiseodd.github.io/techblog/2016/07/04/batchnorm/>`_
- `Understanding the backward pass through Batch Norm <https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html>`_


Convolution
-----------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Dropout
-------

A dropout layer takes the output of the previous layer's activations and randomly sets a certain fraction (dropout rate) of the activatons to 0, cancelling or 'dropping' them out.

It is a common regularization technique used to prevent overfitting in Neural Networks.

.. image:: images/dropout_net.png
      :align: center

The dropout rate is the tunable hyperparameter that is adjusted to measure performance with different values. It is typically set between 0.2 and 0.5 (but may be arbitrarily set).

Dropout is only used during training; At test time, no activations are dropped, but scaled down by a factor of dropout rate. This is to account for more units being active during test time than training time.

For example:

 - A layer in a neural net outputs a tensor (matrix) A of shape (batch_size, num_features).
 - The dropout rate of the layer is set to 0.5 (50%).
 - A random 50% of the values in A will be set to 0.
 - These will then be multiplied with the weight matrix to form the inputs to the next layer.

The premise behind dropout is to introduce noise into a layer in order to disrupt any interdependent learning or coincidental patterns that may occur between units in the layer, that aren't significant.

.. rubric:: Code

.. code-block:: python

      # layer_output is a 2D numpy matrix of activations

      layer_output *= np.random.randint(0, high=2, size=layer_output.shape) # dropping out values

      # scaling up by dropout rate during TRAINING time, so no scaling needs to be done at test time
      layer_output /= 0.5 
      # OR
      layer_output *= 0.5 # Scaling down during TEST time.

.. [2]

This results in the following operation.

.. image:: images/dropout.png
      :align: center

All reference, images and code examples, unless mentioned otherwise, are from section 4.4.3 of `Deep Learning for Python <https://www.manning.com/books/deep-learning-with-python>`_ by François Chollet. 

.. [2]


Linear
------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


LSTM
----

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Pooling
-------

Max and average pooling layers.

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


RNN
---

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


.. rubric:: References

.. [1] http://www.deeplearningbook.org/contents/convnets.html
.. [2] “4.4.3, Fundamentals of Machine Learning: Adding Dropout.” `Deep Learning for Python <https://www.manning.com/books/deep-learning-with-python>`_, by Chollet, François. Manning Publications Co., 2018, pp. 109–110.
