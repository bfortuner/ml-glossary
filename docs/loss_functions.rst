.. _cost_function:

==============
Loss Functions
==============

.. contents:: :local:


.. _loss_cross_entropy:

Cross-Entropy
=============

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

.. image:: images/cross_entropy.png
    :align: center

The graph above shows the range of possible loss values given a true observation (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss increases rapidly. Log loss penalizes both types of errors, but especially those predications that are confident and wrong!

.. note::

Cross-entropy and log loss are slightly different depending on context, but in machine learning when calculating error rates between 0 and 1 they resolve to the same thing.

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: CrossEntropy

.. rubric:: Math

In binary classification, where the number of classes :math:`M` equals 2, cross-entropy can be calculated as:

.. math::

  -{(y\log(p) + (1 - y)\log(1 - p))}

If :math:`M > 2` (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.

.. math::

  -\sum_{c=1}^My_{o,c}\log(p_{o,c})

.. note::

  - M - number of classes (dog, cat, fish)
  - log - the natural log
  - y - binary indicator (0 or 1) if class label :math:`c` is the correct classification for observation :math:`o`
  - p - predicted probability observation :math:`o` is of class :math:`c`


.. _hinge_loss:

Hinge
=====

Used for classification. 

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: Hinge


.. _huber_loss:

Huber
=====

Typically used for regression. It's less sensitive to outliers than the MSE.

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: Huber
      

.. _kl_divergence:

Kullback-Leibler
================

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: KLDivergence


.. _l1_loss:

L1
=======

Excellent overview below [6] and [10].

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: L2


.. _l2_loss:

L2
==

Excellent overview below [6] and [10].

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: L2


.. _mle:

Maximum Likelihood
==================

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py   
      :pyobject: MLE


.. _mse:

Mean Squared Error
==================

Description of MSE...

.. literalinclude:: ../code/loss_functions.py
    :language: python
    :pyobject: MSE

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
.. [7] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
.. [8] https://en.wikipedia.org/wiki/Huber_loss
.. [9] https://en.wikipedia.org/wiki/Hinge_loss
.. [10] http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
