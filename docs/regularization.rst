.. _regularization:

==============
Regularization
==============

.. contents:: :local:

What is overfitting?

From Wikipedia `overfitting <https://en.wikipedia.org/wiki/Overfitting>`__ is, 

The production of an analysis that corresponds too closely or exactly to a particular set 
of data, and may therefore fail to fit additional data or predict future observations 
reliably

What is Regularization?

It is a Techniques for combating overfitting and improving training.


Data Augmentation
=================

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Dropout
=======

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Early Stopping
==============

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Ensembling
==========

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Injecting Noise
===============

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

L1 Regularization
=================

A regression model that uses L1 regularization technique is called *Lasso Regression*. 

.. rubric:: Mathematical formula for L1 Regularization. 

Let's define a model to see how L1 Regularization works. For simplicity, We define a simple linear regression model Y with one independent variable. 

In this model, W represent Weight, b represent Bias. 

.. math::

  W = w_1, w_2 . . . w_n
  
  X = x_1, x_2 . . . x_n

and the predicted result is :math:`\widehat{Y}` 

.. math::

  \widehat{Y} =  w_1x_1 +  w_2x_2 + . . . w_nx_n + b
 
Following formula calculates the error without Regularization function
  
.. math::

  Loss = Error(Y , \widehat{Y})
  
Following formula calculates the error With L1 Regularization function
  
.. math::

  Loss = Error(Y - \widehat{Y}) + \lambda \sum_i^n |w_i|
  
.. note:: 
	
	Here, If the value of lambda is Zero then above Loss function becomes Ordinary Least Square whereas very large value will make the coefficients (weights) zero hence it will under-fit. 

One thing to note is that :math:`|w|` is differentiable when w!=0 as shown below, 

.. math::

  \frac{\text{d}|w|}{\text{d}w} = \begin{cases}1 & w > 0\\-1 & w < 0\end{cases}
  
To understand the Note above, 

Let's substitute the formula in finding new weights using Gradient Descent optimizer. 

.. math::

   w_{new} = w - \eta\frac{\partial L1}{\partial w}
   
When we apply the L1 in above formula it will become, 

.. math::

   w_{new} = w - \eta. (Error(Y , \widehat{Y}) + \lambda\frac{\text{d}|w|}{\text{d}w})
           
           = \begin{cases}w - \eta . (Error(Y , \widehat{Y}) +\lambda) & w > 0\\w - \eta . (Error(Y , \widehat{Y}) -\lambda) & w < 0\end{cases}
 
From the above formula, 

- If w is positive, the regularization parameter :math:`\lambda` > 0 will push w to be less positive, by subtracting :math:`\lambda` from w. 
- If w is negative, the regularization parameter :math:`\lambda` < 0 will push w to be less negative, by adding :math:`\lambda` to w.  hence this has the effect of pushing w towards 0. 

Simple python implementation

::
   def update_weights_with_l1_regularization(features, targets, weights, lr,lambda):
    '''
    Features:(200, 3)
    Targets: (200, 1)
    Weights:(3, 1)
    '''
    predictions = predict(features, weights)

    #Extract our features
    x1 = features[:,0]
    x2 = features[:,1]
    x3 = features[:,2]

    # Use matrix cross product (*) to simultaneously
    # calculate the derivative for each weight
    d_w1 = -x1*(targets - predictions)
    d_w2 = -x2*(targets - predictions)
    d_w3 = -x3*(targets - predictions)

    # Multiply the mean derivative by the learning rate
    # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
	
	weights[0][0] = (weights[0][0] - lr * np.mean(d_w1) - lambda) if weights[0][0] > 0 else (weights[0][0] - lr * np.mean(d_w1) + lambda)
    weights[1][0] = (weights[1][0] - lr * np.mean(d_w2) - lambda) if weights[1][0] > 0 else (weights[1][0] - lr * np.mean(d_w2) + lambda)
	weights[2][0] = (weights[2][0] - lr * np.mean(d_w3) - lambda) if weights[2][0] > 0 else (weights[2][0] - lr * np.mean(d_w3) + lambda)
	
    return weights


.. rubric:: Further reading

- `Linear Regression  <https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html>`_


L2 Regularization
=================

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__



.. rubric:: References

.. [1] http://www.deeplearningbook.org/contents/regularization.html
