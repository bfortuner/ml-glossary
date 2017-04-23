.. _cost_function:

==============
Loss Functions
==============

.. contents:: :local:

Introduction
============

A loss function, or cost function, is a wrapper around our model's predict function that tells us "how good" the model is at making predictions for a given set of parameters. The loss function has its own curve and its own derivatives. The slope of this curve tells us how to change our parameters to make the model more accurate! We use the model to make predictions. We use the cost function to update our parameters. Our cost function can take a variety of forms as there are many different cost functions available. Popular loss functions include: :ref:`mse` and :ref:`Cross-entropy Loss <loss_cross_entropy>`.


.. _loss_cross_entropy:

Cross-Entropy Loss
==================

Cross-entropy loss, or Log Loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

The graph below shows the range of possible loss values given a true observation (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss increases rapidly. Log loss penalizes both types of errors, but especially those predications that are confident and wrong!

.. image:: images/cross_entropy.png
    :align: center

.. note::

  Cross-entropy and log loss are slightly different depending on context, but in machine learning when calculating error rates between 0 and 1 they resolve to the same thing.

.. rubric:: Code

::

  def cross_entropy(true_label, prediction):
      if true_label == 1:
          return -log(prediction)
      else:
          return -log(1 - prediction)

.. rubric:: Binary classification (M=2)

.. math::

  -{(y\log(p) + (1 - y)\log(1 - p))}

.. note::

  - N - number of observations
  - M - number of possible class labels (dog, cat, fish)
  - log - the natural logarithm
  - y - a binary indicator (0 or 1) of whether class label :math:`c` is the correct classification for observation :math:`o`
  - p - the model's predicted probability that observation :math:`o` is of class :math:`c`


.. rubric:: Multi-class cross-entropy

In multi-class classification (M>2), we take the sum of loss values for each class prediction in the observation.

.. math::

  -\sum_{c=1}^My_{o,c}\log(p_{o,c})

.. note::

  Why the Negative Sign?

  Cross-entropy takes the negative log to provide an easy metric for comparison. It takes this approach because the positive log of numbers < 1 returns negative values, which is confusing to work with when comparing the performance of two models.

  .. image:: images/log_vs_neglog.gif
        :align: center


.. _hinge_loss:

Hinge Loss
==========

Be the first to contribute!


.. _kl_divergence:

Kullback-Leibler divergence
===========================

Be the first to contribute!


.. _l1_loss:

L1 Loss
=======

Be the first to contribute!


.. _l2_loss:

L2 Loss
=======

Be the first to contribute!


.. _mle:

Maximum Likelihood
==================

Be the first to contribute!


.. _mse:

Mean Squared Error
==================

Description of MSE...

.. literalinclude:: ../code/loss_functions.py
    :language: python
    :pyobject: MSE

**Derivative**

.. literalinclude:: ../code/loss_functions.py
    :language: python
    :pyobject: MSE_prime


.. rubric:: References

.. [1] https://en.m.wikipedia.org/wiki/Cross_entropy
.. [2] https://www.kaggle.com/wiki/LogarithmicLoss
.. [3] https://en.wikipedia.org/wiki/Loss_functions_for_classification
.. [4] http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
.. [5] http://neuralnetworksanddeeplearning.com/chap3.html
.. [6] http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/
