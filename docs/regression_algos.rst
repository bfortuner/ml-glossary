.. _regression_algos:

=====================
Regression Algorithms
=====================

.. contents:: :local:


Ordinary Least Squares
======================

OLS is the method with which linear regression is performed. The square of the difference from the mean is taken for every data point, and the summed loss function is to be minimized.

.. math::
    l = \sum_{i=1}^n (y_i - \bar{y})^2


Polynomial
==========

Polynomial regression is a modification of linear regression where the existing features are mapped to a polynomial form. The problem is still a linear regression problem, but the input vector is now mapped to a higher dimensional vector which serves as a pseudo-input vector of sorts.

.. math::

    \textbf{x} = (x_0, x_1) \rightarrow \textbf{x'} = (x_0, x^2_0, x_1, x^2_1, x_0x_1)


Lasso
=====

Lasso Regression tries to reduce the ordinary least squares error similar to vanilla regression, but adds an extra term. The sum of the :math:`L_1` norm for every data point multiplied by a hyperparameter :math:`\alpha` is used. This reduces model complexity and prevents overfitting. 

.. math::

    l = \sum_{i=1}^n (y_i - \tilde{y})^2 + \alpha \sum_{j=1}^p |w_j|


Ridge
=====

Ridge regression is similar to lasso regression, but the regularization term uses the :math:`L_2` norm instead.

.. math::

    l = \sum_{i=1}^n (y_i - \tilde{y})^2 + \alpha \sum_{j=1}^p w^2_j


Stepwise
========

Stepwise regression or spline regression helps us fit a piece wise function to the data. It is usually used with linear models, but it can be generalized to higher degrees as well. The regression equation takes the form of

.. math::
    y = ax + b(x-\bar{x})H_{\alpha}+c
 
where :math:`H_{\alpha}` is the shifted Heaviside step function, having its discontinuity at :math:`\alpha`.


.. rubric:: References

.. [1] https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/
.. [2] http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/



