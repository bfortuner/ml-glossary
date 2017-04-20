.. _algorithms:

===============================
Algorithms
===============================

.. toctree::
  :maxdepth: 1

Fundamental machine learning algorithms and concepts


Linear Regression
===================

When a model's predicted output is continuous and has a constant slope.
At its most basic, it takes the form of:

.. math::

  y = ax + b

* **Pros**: fast, no tuning required, highly interpretable, well-understood
* **Cons**: unlikely to produce the best predictive accuracy

(presumes a linear relationship between the features and response)

[[File:Linear_regression_glossary.png]]

A more complex linear equation might look like this:

.. math::

  y = B_0 + B_1 x + B_2 z + B_3 j + B_4 k

A linear regression model would try to "learn" the correct values for
:math:`B_0, B_1, B_2 ..` The independent variables :math:`x, y, j, k`
represent the various attributes of each observation in our sample. For
sales predictions, these attributes might include: day of the week, employee
count, inventory levels, and store location.

.. math::

  y = B_0 + B_1 Day + B_2 Employees + B_3 Inventory + B_4 Location

Best Reads:
* <https://arxiv.org/abs/1511.07122>
* <https://en.wikipedia.org/wiki/Linear_regression>

References:
* <http://people.duke.edu/~rnau/regintro.htm>
* <https://en.wikipedia.org/wiki/Linear_regression>


Logistic Regression
===================

The bread and butter of neural networks is *affine transformations*: a
vector is received as input and is multiplied with a matrix to produce an
output (to which a bias vector is usually added before passing the result
through a nonlinearity). This is applicable to any type of input, be it an
image, a sound clip or an unordered collection of features: whatever their
dimensionality, their representation can always be flattened into a vector
before the transformation.


