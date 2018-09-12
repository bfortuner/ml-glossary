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

The graph above shows the range of possible loss values given a true observation (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss increases rapidly. Log loss penalizes both types of errors, but especially those predictions that are confident and wrong!

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

Typically used for regression. It's less sensitive to outliers than the MSE as it treats error as square only inside an interval.

.. math::

  L_{\delta}=\left\{\begin{matrix}
  \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
  \delta (y - \hat{y}) - \frac1 2 \delta & otherwise
  \end{matrix}\right.

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py
      :pyobject: Huber

Further information can be found at `Huber Loss in Wikipedia`_.  

.. _`Huber Loss in Wikipedia`: https://en.wikipedia.org/wiki/Huber_loss

.. _kl_divergence:

Kullback-Leibler
================

Used in Regression and Classification. The kullback-Liebler Divergence compares similarity between two distributions. It's calculated as follows for discrete datasets:

.. math::

    D_{KL}(P|Q) = \sum_{i=1}^{N} {p_{i} \: log(\frac{p_{i}}{q_i})}

Assuming P as the objective (:math:`y`) and Q as the prediction (:math:`\hat{y}`), the KL-divergence is implemented as:

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py
      :pyobject: KLDivergence

Unfortunatelly, this implementation of Kullback-Leibler divergence (KL-divergence) cannot be applied directly to classification as it diverges on the occurence of 0.

One way to avoid this divergence is to ignore elements with zero as suggested by Matus Telgarsky in `How to compute KL-divergence when PMF contains 0s?`_. This approach may lead to errors, as it may ignore errors in the prediction where the label is 0 (e.g.: logit classification).
Another way to compute the Loss is suggested by user1417648 on `Calculate the Kullback-Leibler Divergence in practice?`_, since the KL-divergence is assimetric (:math:`D_{KL}(P|Q) \neq D_{KL}(Q|P)) and :math:`P + P`

.. math::

    L_{KL} = \frac{D_{KL}(P|\frac{P+Q}{2}) + D_{KL}(Q|\frac{P+Q}{2})}{2}

.. rubric:: Code

    def KLDivergence(yHat,y):
        return np.where(yHat*y == 0, np.zeros(shape=y.shape),(y*np.log(y/((yHat+y)/2))+yHat*np.log(yHat/((y+yHat/2)))/2)

.. _`How to compute KL-divergence when PMF contains 0s?`: https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
.. _`Calculate the Kullback-Leibler Divergence in practice?`: https://stats.stackexchange.com/questions/97938/calculate-the-kullback-leibler-divergence-in-practice

.. _mae:

MAE (L1)
========

Mean Absolute Error, or L1 loss. Excellent overview below [6] and [10].

.. rubric:: Code

.. literalinclude:: ../code/loss_functions.py
      :pyobject: L1


.. _mse:

MSE (L2)
========

Mean Squared Error, or L2 loss. Excellent overview below [6] and [10].

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
