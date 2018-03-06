.. _optimizers:

==========
Optimizers
==========

.. contents:: :local:

Adadelta
--------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Adagrad
-------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Adam
----

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Conjugate Gradients
-------------------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


.. _optimizers_lbfgs:

BFGS
----

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Momentum
--------

Used in conjunction Stochastic Gradient Descent (sgd) or Mini-Batch Gradient Descent, Momentum takes into account
past gradients to smooth out the update. This is seen in variable :math:`v` which is an exponentially weighted average
of the gradient on previous steps. This results in minimizing oscillations and faster convergence

.. math::


    v_{dW} = \beta v_{dW} + (1 - \beta) dW \\
    W = W - \alpha v_{dW}

.. note::

  - :math:`v` - the exponentially weighted average
  - :math:`dW` - cost graident with respect to current layer weight tensor
  - :math:`W` - weight tensor
  - :math:`\beta` - hyperparameter to be tuned
  - :math:`\alpha` - the learning rate

Nesterov Momentum
-----------------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Newton's Method
---------------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


RMSProp
-------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


SGD
---

Stochastic Gradient Descent.

.. literalinclude:: ../code/optimizers.py
    :language: python
    :pyobject: SGD


.. rubric:: References

.. [1] http://sebastianruder.com/optimizing-gradient-descent/
.. [2] http://www.deeplearningbook.org/contents/optimization.html
.. [3] https://arxiv.org/pdf/1502.03167.pdf
