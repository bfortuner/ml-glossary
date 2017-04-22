.. _cost_function:

=============
Cost function
=============

A cost function is a wrapper around our model function that tells us "how good" our model is at making predictions for a given set of parameters. The cost function has its own curve and its own derivatives. The slope of this curve tells us how to change our parameters to make the model more accurate! We use the model to make predictions. We use the cost function to update our parameters. Our cost function can take a variety of forms as there are many different cost functions available. Popular cost functions include: Mean Squared Error, Root Mean Squared Error, and [[Log Loss]].

[[File:linear_line_w_cost_function.png]]
[http://www.ken-szulczyk.com/misc/statistical_lecture_10.php source]

Let's take an example from linear regression where our model is :math:`f(x) = mx + b`, where :math:`m` and :math:`b` are the parameters we can tweak.

If we use Mean Squared Error as our cost function, we can calculate total cost of our predictions like this:

Math
----

.. math::

  MSE =  \frac{1}{N} \sum_{i=1}^{n} (y_i - (mx_i + b))^2

* :math:`N` is the total number of observations (data points)
* :math:`frac{1}{N} \sum_{i=1}^{n}` is the mean
* :math:`y_i` is the actual value of an observation
* :math:`mx_i + b` is our prediction

Code
----

::

  def cost_function(x, y, m, b):
      N = len(x)
      total_error = 0.0
      for i in range(N):
          total_error += (y[i] - (m*x[i] + b))**2
      return total_error / N
