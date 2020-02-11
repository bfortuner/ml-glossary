.. _optimizers:

==========
Optimizers
==========

.. contents:: :local:

Adadelta
--------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


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

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


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

Stochastic Gradient Descent.

.. literalinclude:: ../code/optimizers.py
    :language: python
    :pyobject: SGD


.. rubric:: References

.. [1] http://sebastianruder.com/optimizing-gradient-descent/
.. [2] http://www.deeplearningbook.org/contents/optimization.html
.. [3] https://arxiv.org/pdf/1502.03167.pdf
