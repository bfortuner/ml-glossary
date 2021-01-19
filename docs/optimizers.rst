.. _optimizers:

==========
Optimizers
==========

.. rubric:: What is Optimizer ? 

It is very important to tweak the weights of the model during the training process, to make our predictions as correct and optimized as possible. But how exactly do you do that? How do you change the parameters of your model, by how much, and when?

Best answer to all above question is *optimizers*. They tie together the loss function and model parameters by updating the model in response to the output of the loss function. In simpler terms, optimizers shape and mold your model into its most accurate possible form by futzing with the weights. The loss function is the guide to the terrain, telling the optimizer when it’s moving in the right or wrong direction.

Below are list of example optimizers

.. contents:: :local:

.. image:: images/optimizers.gif
      :align: center

Image Credit: `CS231n <https://cs231n.github.io/neural-networks-3/>`_

Adagrad
-------

Adagrad (short for adaptive gradient) adaptively sets the learning rate according to a parameter.

- Parameters that have higher gradients or frequent updates should have slower learning rate so that we do not overshoot the minimum value.
- Parameters that have low gradients or infrequent updates should faster learning rate so that they get trained quickly.
- It divides the learning rate by the sum of squares of all previous gradients of the parameter.
- When the sum of the squared past gradients has a high value, it basically divides the learning rate by a high value, so the learning rate will become less. 
- Similarly, if the sum of the squared past gradients has a low value, it divides the learning rate by a lower value, so the learning rate value will become high. 
- This implies that the learning rate is inversely proportional to the sum of the squares of all the previous gradients of the parameter.

.. math::

    g_{t}^{i} = \frac{\partial \mathcal{J}(w_{t}^{i})}{\partial W} \\
    W = W - \alpha \frac{\partial \mathcal{J}(w_{t}^{i})}{\sqrt{\sum_{r=1}^{t}\left ( g_{r}^{i} \right )^{2} + \varepsilon }}
    
.. note::
  
  - :math:`g_{t}^{i}` - the gradient of a parameter, :math: `\Theta `  at an iteration t.
  - :math:`\alpha` - the learning rate
  - :math:`\epsilon` - very small value to avoid dividing by zero

.. literalinclude:: ../code/optimizers.py
    :language: python
    :pyobject: Adagrad

Adadelta
--------

AdaDelta belongs to the family of stochastic gradient descent algorithms, that provide adaptive techniques for hyperparameter tuning. Adadelta is probably short for ‘adaptive delta’, where delta here refers to the difference between the current weight and the newly updated weight. 

The main disadvantage in Adagrad is its accumulation of the squared gradients. During the training process, the accumulated sum keeps growing. From the above formala we can see that, As the accumulated sum increases learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge.

Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done.

With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the update rule.

Implementation is something like this, 

.. math::

  v_t = \rho v_{t-1} + (1-\rho) \nabla_\theta^2 J( \theta) \\ 
  \Delta\theta &= \dfrac{\sqrt{w_t + \epsilon}}{\sqrt{v_t + \epsilon}} \nabla_\theta J( \theta) \\
  \theta &= \theta - \eta \Delta\theta \\ 
  w_t = \rho w_{t-1} + (1-\rho) \Delta\theta^2

.. literalinclude:: ../code/optimizers.py
    :language: python
    :pyobject: Adadelta

Adam
----

Adaptive Moment Estimation (Adam) combines ideas from both RMSProp and Momentum. It computes adaptive learning rates for each parameter and works as follows.

- First, it computes the exponentially weighted average of past gradients (:math:`v_{dW}`).
- Second, it computes the exponentially weighted average of the squares of past gradients (:math:`s_{dW}`).
- Third, these averages have a bias towards zero and to counteract this a bias correction is applied (:math:`v_{dW}^{corrected}`, :math:`s_{dW}^{corrected}`).
- Lastly, the parameters are updated using the information from the calculated averages.

.. math::


    v_{dW} = \beta_1 v_{dW} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W } \\
    s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W })^2 \\
    v^{corrected}_{dW} = \frac{v_{dW}}{1 - (\beta_1)^t} \\
    s^{corrected}_{dW} = \frac{s_{dW}}{1 - (\beta_1)^t} \\
    W = W - \alpha \frac{v^{corrected}_{dW}}{\sqrt{s^{corrected}_{dW}} + \varepsilon}

.. note::

  - :math:`v_{dW}` - the exponentially weighted average of past gradients
  - :math:`s_{dW}` - the exponentially weighted average of past squares of gradients
  - :math:`\beta_1` - hyperparameter to be tuned
  - :math:`\beta_2` - hyperparameter to be tuned
  - :math:`\frac{\partial \mathcal{J} }{ \partial W }` - cost gradient with respect to current layer
  - :math:`W` - the weight matrix (parameter to be updated)
  - :math:`\alpha` - the learning rate
  - :math:`\epsilon` - very small value to avoid dividing by zero



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
of the gradient on previous steps. This results in minimizing oscillations and faster convergence.

.. math::

    v_{dW} = \beta v_{dW} + (1 - \beta) \frac{\partial \mathcal{J} }{ \partial W } \\
    W = W - \alpha v_{dW}

.. note::

  - :math:`v` - the exponentially weighted average of past gradients
  - :math:`\frac{\partial \mathcal{J} }{ \partial W }` - cost gradient with respect to current layer weight tensor
  - :math:`W` - weight tensor
  - :math:`\beta` - hyperparameter to be tuned
  - :math:`\alpha` - the learning rate

Nesterov Momentum
-----------------

Nesterov momentum optimization is a minor but effective variation of the regular momentum optimization proposed by Yuri Nesterov. 
The key concept is that the gradient of the cost function is measured ahead of the local position in the direction 
of the momentum (at point :math:`W + \beta v_{dW}`). This Works since the momentum vector will be pointing in the correct direction and
is generally faster than regular momentum optimization.

.. math::

    v_{dW} = \beta v_{dW} + (1 - \beta) \frac{\partial \mathcal{J} }{ \partial (W + \beta v_{dW}) } \\
    W = W - \alpha v_{dW}


Newton's Method
---------------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


RMSProp
-------

Another adaptive learning rate optimization algorithm, Root Mean Square Prop (RMSProp) works by keeping an exponentially weighted average of the squares of past gradients.
RMSProp then divides the learning rate by this average to speed up convergence.


.. math::


    s_{dW} = \beta s_{dW} + (1 - \beta) (\frac{\partial \mathcal{J} }{\partial W })^2 \\
    W = W - \alpha \frac{\frac{\partial \mathcal{J} }{\partial W }}{\sqrt{s^{corrected}_{dW}} + \varepsilon}

.. note::

  - :math:`s` - the exponentially weighted average of past squares of gradients
  - :math:`\frac{\partial \mathcal{J} }{\partial W }` - cost gradient with respect to current layer weight tensor
  - :math:`W` - weight tensor
  - :math:`\beta` - hyperparameter to be tuned
  - :math:`\alpha` - the learning rate
  - :math:`\epsilon` - very small value to avoid dividing by zero

SGD
---

SGD stands for Stochastic Gradient Descent.In Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. In Gradient Descent, there is a term called “batch” which denotes the total number of samples from a dataset that is used for calculating the gradient for each iteration. In typical Gradient Descent optimization, like Batch Gradient Descent, the batch is taken to be the whole dataset. Although, using the whole dataset is really useful for getting to the minima in a less noisy or less random manner, but the problem arises when our datasets get really huge.

This problem is solved by Stochastic Gradient Descent. In SGD, it uses only a single sample to perform each iteration. The sample is randomly shuffled and selected for performing the iteration.

Since only one sample from the dataset is chosen at random for each iteration, the path taken by the algorithm to reach the minima is usually noisier than your typical Gradient Descent algorithm. But that doesn’t matter all that much because the path taken by the algorithm does not matter, as long as we reach the minima and with significantly shorter training time.

.. literalinclude:: ../code/optimizers.py
    :language: python
    :pyobject: SGD


.. rubric:: References

.. [1] http://sebastianruder.com/optimizing-gradient-descent/
.. [2] http://www.deeplearningbook.org/contents/optimization.html
.. [3] https://arxiv.org/pdf/1502.03167.pdf
