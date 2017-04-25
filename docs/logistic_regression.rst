.. _logistic_regression:

===================
Logistic Regression
===================

.. contents:: :local:

Introduction
============

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs number values on a continuous plane, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

Comparison to linear regression
-------------------------------

Given data on time spent studying and exam scores. :doc:`linear_regression` and logistic regression can predict different things:

  - **Linear Regression** could help us predict the student's test score on a scale of 0 - 100. Linear regression predictions are continuous (numbers in a range).

  - **Logistic Regression** could help use predict whether the student passed or failed. Logistic regression predictions are discrete (only specific values or categories are allowed). We can also view probability scores underlying the model's classifications.

Types of logistic regression
----------------------------

  - Binary (Pass/Fail)
  - Multi (Cats, Dogs, Sheep)
  - Ordinal (Low, Medium, High)



Binary logistic regression
==========================

Say we're given `data <http://scilab.io/wp-content/uploads/2016/07/data_classification.csv>`_ on student exam results and our goal is to predict whether a student will pass or fail based on number of hours slept and hours spent studying. We have two features (hours slept, hours studied) and two classes: passed (1) and failed (0).


+--------------+-------------+-------------+
| **Studied**  | **Slept**   | **Passed**  |
+--------------+-------------+-------------+
| 4.85         | 9.63        | 1           |
+--------------+-------------+-------------+
| 8.62         | 3.23        | 0           |
+--------------+-------------+-------------+
| 5.43         | 8.23        | 1           |
+--------------+-------------+-------------+
| 9.21         | 6.34        | 0           |
+--------------+-------------+-------------+

Graphically we could represent our data with a scatter plot.

.. image:: images/logistic_regression_exam_scores_scatter.png
    :align: center


Sigmoid activation
------------------

In order to map predicted values to probabilities, we use the :ref:`sigmoid <activation_sigmoid>` function. The function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.

.. rubric:: Math

.. math::

  S(z) = \frac{1} {1 + e^{-z}}

.. note::

  - :math:`s(z)` = output between 0 and 1 (probability estimate)
  - :math:`z` = input to the function (your algorithm's prediction e.g. mx + b)
  - :math:`e` = base of natural log

.. rubric:: Graph

.. image:: images/sigmoid.png
    :align: center

.. rubric:: Code

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: sigmoid

.. could potentially link to another file.. http://docutils.sourceforge.net/docs/ref/rst/directives.html#include


Decision boundary
-----------------

Our current prediction function returns a probability score between 0 and 1. In order to map this to a discrete class (true/false, cat/dog), we select a threshold value or tipping point above which we will classify values into class 1 and below which we classify values into class 2.

.. math::

  p \geq 0.5, class=1 \\
  p < 0.5, class=0

For example, if our threshold was .5 and our prediction function returned .7, we would classify this observation as positive. If our prediction was .2 we would classify the observation as negative. For logistic regression with multiple classes we could select the class with the highest predicted probability.

.. image:: images/logistic_regression_sigmoid_w_threshold.png
    :align: center


Making predictions
------------------

Using our knowledge of sigmoid functions and decision boundaries, we can now write a prediction function. A prediction function in logistic regression returns the probability of our observation being positive, True, or "Yes". We call this class 1 and its notation is :math:`P(class=1)`. As the probability gets closer to 1, our model is more confident that the observation is in class 1.

.. rubric:: Math

Let's use the same :ref:`multiple linear regression <multiple_linear_regression_predict>` equation from our linear regression tutorial.

.. math::

  z = W_0 + W_1 Studied + W_2 Slept

This time however we will transform the output using the sigmoid function to return a probability value between 0 and 1.

.. math::

  P(class=1) = \frac{1} {1 + e^{-z}}

If the model returns .4 it believes there is only a 40% chance of passing. If our decision boundary was .5, we would categorize this observation as "Fail.""

.. rubric:: Code

We wrap the sigmoid function over the same prediction function we used in :ref:`multiple linear regression <multiple_linear_regression_predict>`

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: predict


Cost function
-------------

Unfortunately we can't (or at least shouldn't) use the same cost function :ref:`mse` as we did for linear regression. Why? There is a great math explanation in chapter 3 of Michael Neilson's deep learning book [5]_, but for now I'll simply say it's because our prediction function is non-linear (due to sigmoid transform). Squaring this prediction as we do in MSE results in a non-convex function with many local minimums. If our cost function has many local minimums, gradient descent may not find the optimal global minimum.

.. rubric:: Math

Instead of Mean Squared Error, we use a cost function called :ref:`loss_cross_entropy`, also known as Log Loss. Cross-entropy loss can be divided into two separate cost functions: one for :math:`y=1` and one for :math:`y=0`.

.. image:: images/ng_cost_function_logistic.png
    :align: center

The benefits of taking the logarithm reveal themselves when you look at the cost function graphs for y=1 and y=0. These smooth monotonic functions [7]_ (always increasing or always decreasing) make it easy to calculate the gradient and minimize cost. Image from Andrew Ng's slides on logistic regression [1]_.

.. image:: images/y1andy2_logistic_function.png
    :align: center

The key thing to note is the cost function penalizes confident and wrong predictions more than it rewards confident and right predictions! The corollary is increasing prediction accuracy (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of our cost function.

..rubric:: Above functions compressed into one

.. image:: images/logistic_cost_function_joined.png
    :align: center

Multiplying by :math:`y` and :math:`(1-y)` in the above equation is a sneaky trick that let's us use the same equation to solve for both y=1 and y=0 cases. If y=0, the first side cancels out. If y=1, the second side cancels out. In both cases we only perform the operation we need to perform.

.. rubric:: Vectorized cost function

.. image:: images/logistic_cost_function_vectorized.png
    :align: center

.. rubric:: Code

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: cost_function


Gradient descent
----------------

To minimize our cost, we use :doc:`gradient_descent` just like before in :doc:`linear_regression`. There are other more sophisticated optimization algorithms out there such as conjugate gradient like :ref:`optimizers_lbfgs`, but you don't have to worry about these. Machine learning libraries like Scikit-learn hide their implementations so you can focus on more interesting things!

.. rubric:: Math

One of the neat properties of the sigmoid function is its derivative is easy to calculate. If you're curious, there is a good walk-through derivation on stack overflow [6]_. Michael Neilson also covers the topic in chapter 3 of his book.

.. math::

  \begin{align}
  s'(z) & = s(z)(1 - s(z))
  \end{align}

Which leads to an equally beautiful and convenient cost function derivative:

.. math::

  C' = x(s(z) - y)

.. note::

  - :math:`C'` is the derivative of cost with respect to weights
  - :math:`y` is the actual class label (0 or 1)
  - :math:`s(z)` is your model's prediction
  - :math:`x` is your feature or feature vector.

Notice how this gradient is the same as the :ref:`mse` gradient, the only difference is the hypothesis function.

.. rubric:: Pseudocode

::

  Repeat {

    1. Calculate gradient average
    2. Multiply by learning rate
    3. Subtract from weights

  }

.. rubric:: Code

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: update_weights


Mapping probabilities to classes
--------------------------------

The final step is assign class labels (0 or 1) to our predicted probabilities.

.. rubric:: Decision boundary

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: decision_boundary

.. rubric:: Convert probabilities to classes


.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: classify

.. rubric:: Example output

::

  Probabilities = [ 0.967, 0.448, 0.015, 0.780, 0.978, 0.004]
  Classifications = [1, 0, 0, 1, 1, 0]


Training
--------

Our training code is the same as we used for :ref:`linear regression <simple_linear_regression_training>`.

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: train


Model evaluation
----------------

If our model is working, we should see our cost decrease after every iteration.

::

  iter: 0 cost: 0.635
  iter: 1000 cost: 0.302
  iter: 2000 cost: 0.264

**Final cost:**  0.2487.  **Final weights:** [-8.197, .921, .738]

.. rubric:: Cost history

.. image:: images/logistic_regression_loss_history.png
    :align: center

.. rubric:: Accuracy

:ref:`Accuracy <glossary_accuracy>` measures how correct our predictions were. In this case we simple compare predicted labels to true labels and divide by the total.

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: accuracy


.. rubric:: Decision boundary

Another helpful technique is to plot the decision boundary on top of our predictions to see how our labels compare to the actual labels. This involves plotting our predicted probabilities and coloring them with their true labels.

.. image:: images/logistic_regression_final_decision_boundary.png
    :align: center

.. rubric:: Code to plot the decision boundary

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: plot_decision_boundary


Multiclass logistic regression
==============================

Instead of :math:`y = {0,1}` we will expand our definition so that :math:`y = {0,1...n}`. Basically we re-run binary classification multiple times, once for each class.

Procedure
---------

  #. Divide the problem into n+1 binary classification problems (+1 because the index starts at 0?).
  #. For each class...
  #. Predict the probability the observations are in that single class.
  #. prediction = <math>max(probability of the classes)

For each sub-problem, we select one class (YES) and lump all the others into a second class (NO). Then we take the class with the highest predicted value.


Softmax activation
------------------

something about softmax here...


Scipy example
-------------

Let's compare our performance to the ``LogisticRegression`` model provided by scikit-learn [8]_.

.. literalinclude:: ../code/logistic_regression_scipy.py


**Scikit score:**  0.88. **Our score:** 0.89


.. rubric:: References

.. [1] http://www.holehouse.org/mlclass/06_Logistic_Regression.html
.. [2] http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning
.. [3] https://scilab.io/machine-learning-logistic-regression-tutorial/
.. [4] https://github.com/perborgen/LogisticRegression/blob/master/logistic.py
.. [5] http://neuralnetworksanddeeplearning.com/chap3.html
.. [6] http://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
.. [7] https://en.wikipedia.org/wiki/Monotoniconotonic_function
.. [8] http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>
