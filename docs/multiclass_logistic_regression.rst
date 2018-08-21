.. _multiclass_logistic_regression:

===================
Multi-Class Logistic Regression
===================

.. contents:: :local:

Introduction
============

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. An example of a logistic regressor used for binary classification can be found at :doc:`logistic_regression`, and this exmaple contains a slightly more advanced multiclass implementation using numpy's matrix algebra tools and a one-vs-all approach to enable extension to multi-class problems.

One-Versus-All Classification
=============================

The logistic regression algorithm is inherently limited to binary classification. So to enable use of the algorithm for multi-class classification, we implement the one-versus-all approach. Simply put, the procedure is:

  #. Divide the problem into a set of binary classification problems, one for each unique value of y (each class)
  #. Then, for each class:
  #. Use a logistic regression to predict the probability that the observations are in that single class.
  #. prediction = max(softmax(probability of the classes))

The value of implementing the softmax function instead of simply taking the max of the predictions will be discussed further along in the document.

Problem Initialization
======================

In this case, we will be using the highly-studied Iris dataset as our object of study. The iris dataset is commonly used to demonstrate the necessity of supervised classification, as it is clear from plotting the data that it is not clearly separable into three distinct clusters.

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

The y data takes the shape of a multi-class label vector:

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

This needs to be converted into three y data sets (one for each flower type). We'll do this using the tools in numpy:

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

Using the matrix algebra tools in numpy, we'll be building functions that look very similar to those in the binary logistic regression, but can handle all three models simultaneously. Additionally, we finish the initialization of the data structures by standardizing the X data, adding the bias feature, and generating an initial random guess for the weights vector for each of the three models.

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: initialize

Multiclass Logistic Regressor
==============================

We'll quickly work through the code for the multiclass logistic regressors here, but without too much detail as it is assumed that you've already learned the binary logistic regression in :doc:`logistic_regression`.

Sigmoid and Prediction
----------------------

To start, we implement code for a sigmoid function, the heart of the logistic regression:

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: sigmoid

The sigmoid is applied to modify the linear prediction model into a psuedo-probability that can be used as a classification criteria. By properly matching the dimesions of the X (150x5) and weight (3x5) vectors, this is handled for all three binary logistic regressors in a single line:

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

The softmax can be used as a 'confidence' because it will give a smaller answer if the guesses from the different models are of similar magnitude. So, for example, max(softmax([1,10,1]))=0.999, and max(softmax([1,2,1]))=0.576.

Cost function
-------------

Just as for the binary logistic regression, we use the cross entropy (a.k.a. log loss) cost function.

.. literalinclude:: ../code/multiclass_logistic_regression.py
    :language: python
    :pyobject: costFunction


Gradient descent
----------------

To minimize our cost, we use :doc:`gradient_descent` just like before in :doc:`logistic_regression`. The weights are updated in a step-wise fashion, as the negative of the gradient scaled by a learning rate.

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

The training code is very similar to the code we used for :ref:`linear regression <simple_linear_regression_training>`. It iterates over the conjugate gradient steps, updating the weights and a record of the total cost and accuracy of the model as it goes.

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: train


Model evaluation
----------------

If our model is working, we should see our cost decrease after every iteration, and should see our accuracy generally increase. Unlike the cost, however, the accuracy is not directly improved by the algorithm and may not necessarily improve with every iteration.

.. image:: images/multiclass_logistic_training.png
    :align: center

The final accuracy of about 81% isn't particularly great, but we already suspected that it would be difficult to separate this data with linear decision boundaries.

Using a function similar to plotIris(), we can make a custom plot to compare the results of the multiclass logistic regression and confirm our suspicions. Just like the plotIris() function, the color is mapped by the class label, and additionally for the predicted data the marker size directly corresponds to the confidence of the prediction- higher confidence leads to a larger marker size.

.. image:: images/multiclass_logistic_predict_versus_actual.png
    :align: center

The logistic regressor in this case struggles to accurately identify many of the data points, even misclassifying many of the data points that should correspond to the yellow flower type as red, possibly due to the model for the yellow flower type having low predictive values overall. The logistic regression also does a relatively poor job at capturing the 'fuzzy' boundary of the yellow and blue/gray clusters. In general the model is about 50% more confident at classifying the data on the edges compared to the data in the center of the overall dataset, and the origins of the low accuracy are clearly coming from the central region.