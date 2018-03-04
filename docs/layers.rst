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

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


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
