.. _multiclass_logistic_regression:

===================
Multi-Class Logistic Regression
===================

.. contents:: :local:

Introduction
============

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. An example of a logistic regressor used for binary classification can be found at :doc:`logistic_regression`, and here is a slightly more advanced example using numpy's matrix algebra tools and a one-vs-all approach to enable extension to multi-class problems.

One-Versus-All Classification
=============================

The logistic regression algorithm is inherently limited to binary classification. To use the algorithm for multi-class classification then, we implement the one-versus-all approach. Simply put, the procedure is:

Procedure
---------

  #. Divide the problem into a set of binary classification problems, one for each allowable value of y (each class)
  #. Then, for each class:
  #. Use a logistic regression to predict the probability that the observations are in that single class.
  #. prediction = softmax(probability of the classes)

The value of implementing the softmax function instead of simply taking the max of the predictions will be discussed further along in the document.

Problem Initialization
======================

In this case, we will be using the highly-studied Iris dataset as our object of study. The iris dataset is commonly used to demonstrate the necessity of supervised classification in some data, as it is clear from plotting the data that the data is not clearly separable into three distinct clusters.

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: importIris

Although we cannot view the four-dimensional data all at once, we can split up the X data two dimensions at a time and still get a good idea of what the overall data clusters look like:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: plotIris

.. image:: images/iris_scatter.png
    :align: center

Examining at the data, we should expect good performance at identifying the red flower type, as it is fairly easy to visualize a straight line separating the red type from the others on both the plots. However, we can expect more difficulty in separating the yellow and blue/gray flower types.

The y data takes the shape of multi-class y data:
+--------------+
| **Yval**     |
+--------------+
| 0            |
+--------------+
| 1            |
+--------------+
| 2            |
+--------------+
| 1            |
+--------------+

This needs to be converted into three y data sets (one for each flower type). We'll do this quickly using the tools in numpy:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: oneVsAll

This leaves the y data now as three binary y vectors, which we can assemble into a y matrix:

+--------------+-------------+-------------+
| **Y0**       | **Y1**      | **Y2**      |
+--------------+-------------+-------------+
| 1            | 0           | 0           |
+--------------+-------------+-------------+
| 0            | 1           | 0           |
+--------------+-------------+-------------+
| 0            | 0           | 1           |
+--------------+-------------+-------------+
| 1            | 0           | 0           |
+--------------+-------------+-------------+

Using the matrix algebra tools in numpy, we'll be building functions that look very similar to those in a binary logistic regression that can handle all three models simultaneously. In addition, we finish the initialization of the data structures by standardizing the X data, adding the bias feature, and generating an initial random guess for the weights vector for each of the three models.

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: initialize

Multiclass Logistic Regressor
==============================

We'll quickly work through the code for the logistic regressors here, noting the differences between this and the binary logistic regression in :doc:`logistic_regression`

Sigmoid and Prediction
----------------------

To start, we implement code for a sigmoid function, the heart of the logistic regression:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: sigmoid

This is used to modify the linear predictions into a psuedo-probability that can be used as a classification criteria. By properly orienting the X (150x5) and weight (3x5) vectors, this is handled for all three logistic regressors in a single line:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: predict

Multiclass Decision boundary
----------------------------

Our current prediction function returns a probability score between 0 and 1 for each of the three models. To decide which class a data point should be labeled as, we apply a softmax function to the full set of y predictions. The softmax preserves the true max of the data, but also has the effect of the absolute value of the softmax being able to be interpreted as a 'confidence' of the model.

.. image:: images/softmax_math.png
    :align: center

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: softmaxClassifier

Thus we select the Logistic Regressor with the highest probability prediction as the 'True' classification, and use the actual softmax prediction as a way to quantify the confidence of the model's predictions if we so chose.

Cost function
-------------

Just as for the binary logistic regression, we use the cross entropy (a.k.a. log loss) cost function.

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: costFunction


Gradient descent
----------------

To minimize our cost, we use :doc:`gradient_descent` just like before in :doc:`logistic_regression`. 

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: cgStep

Accuracy
--------

The final metric of the success of our model is the accuracy of prediction, defined similarly to the binary problem, but using numpy tools to make the multiclass math easier:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: accuracy

Training
--------

The training code is very similar to the code we used for :ref:`linear regression <simple_linear_regression_training>`.

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: train


Model evaluation
----------------

If our model is working, we should see our cost decrease after every iteration, and should see our accuracy generally increase. Unlike the cost, however, the accuracy is not directly improved by the algorithm and may not necessarily improve with every iteration.

.. image:: images/multiclass_logistic_training.png
    :align: center

The final accuracy of 81% isn't particularly great, but we already suspected that it would be difficult to separate this data with linear decision boundaries.

Using a similar function to plotIris, we can make a custom plot to compare the results of the multiclass logistic regression and confirm our suspicions:

.. image:: images/multiclass_logistic_predict_versus_actual.png
    :align: center

The logistic regression performs well on the red flower type in particular, but struggles to properly capture the somewhat 'fuzzy' boundaries of the yellow and gray/blue flower types.