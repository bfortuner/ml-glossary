.. _glossary:

========
Glossary
========

Definitions of common machine learning terms

.. _accuracy:

Accuracy
  Percentage of correct predictions made by the model.

.. _algorithm:

Algorithm
  A method, function, or series of instructions used to generate a machine learning model_. Examples include linear regression, decision trees, support vector machines, and neural networks.

.. _attribute:

Attribute
  A quality describing an observation (e.g. color, size, weight). In Excel terms, these are column headers.

.. _bias_metric:

Bias Metric
  What is the average difference between your predictions and the correct value for that observation?

  - **Low bias** could mean every prediction is correct. It could also mean half of your predictions are above their actual values and half are below, in equal proportion, resulting in low average difference.

  - **High bias** (with low variance) suggests your model may be underfitting and you're using the wrong architecture for the job.

.. _bias_term:

Bias Term
  Allow models to represent patterns that do not pass through the origin. For example, if all my features were 0, would my output also be zero? Is it possible there is some base value upon which my features have an effect? Bias terms typically accompany weights and are attached to neurons or filters.

.. _categorical_variables:

Categorical Variables
  Variables with a discrete set of possible values. Can be ordinal (order matters) or nominal (order doesn't matter).

.. _classification:

Classification
  Predicting a categorical output (e.g. yes or no?, blue, green or red?).

.. _classification_threshold:

Classification Threshold
  The lowest probability value at which we're comfortable asserting a positive classification. For example, if the predicted probability of being diabetic is > 50%, return True, otherwise return False.

.. _clustering:

Clustering
  Unsupervised grouping of data into buckets.

.. _confusion_matrix:

Confusion Matrix
  Table that describes the performance of a classification model by grouping predictions into 4 categories.

  - **True Positives**: we *correctly* predicted they do have diabetes
  - **True Negatives**: we *correctly* predicted they don't have diabetes
  - **False Positives**: we *incorrectly* predicted they do have diabetes (Type I error)
  - **False Negatives**: we *incorrectly* predicted they don't have diabetes (Type II error)

.. _continuous_variables:

Continuous Variables
  Variables with a range of possible values defined by a number scale (e.g. sales, lifespan).

.. _deduction:

Deduction
  A top-down approach to answering questions or solving problems. A logic technique that starts with a theory and tests that theory with observations to derive a conclusion. E.g. We suspect X, but we need to test our hypothesis before coming to any conclusions.

.. _deep_learning:

Deep Learning
  Contribute a definition!

.. _dimension:

Dimension
  Contribute a definition!

.. _epoch:

Epoch
  Contribute a definition!

.. _extrapolation:

Extrapolation
  Making predictions outside the range of a dataset. E.g. My dog barks, so all dogs must bark. In machine learning we often run into trouble when we extrapolate outside the range of our training data.

.. _feature:

Feature
  With respect to a dataset, a feature represents an attribute_ and value combination. Color is an attribute. "Color is blue" is a feature. In Excel terms, features are similar to cells. The term feature has other definitions in different contexts.

.. _feature_selection:

Feature Selection
  Contribute a definition!

.. _feature_vector:

Feature Vector
  A list of features describing an observation with multiple attributes. In Excel we call this a row.

.. _induction:

Induction
  A bottoms-up approach to answering questions or solving problems. A logic technique that goes from observations to theory. E.g. We keep observing X, so we <b><i>infer</i></b> that Y must be True.

.. _instance:

Instance
  A data point, row, or sample in a dataset. Another term for observation_.

.. _learning_rate:

Learning Rate
  The size of the update steps to take during optimization loops like :ref:`gradient_descent`_. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

.. _loss:

Loss
  Contribute a definition!

.. _machine_learning:

Machine Learning
  Contribute a definition!

.. _model:

Model
  A data structure that stores a representation of a dataset (weights and biases). Models are created/learned when you train an algorithm on a dataset.

.. _neural_networks:

Neural Networks
  Contribute a definition!

.. _normalization:

Normalization
  Contribute a definition!

.. _null_accuracy:

Null Accuracy
  Baseline accuracy that can be acheived by always predicting the most frequent class ("B has the highest frequency, so lets guess B every time").

.. _observation:

Observation
  A data point, row, or sample in a dataset. Another term for instance_.

.. _overfitting:

Overfitting
  Overfitting occurs when your model learns the training data too well and incorporates details and noise specific to your dataset. You can tell a model is overfitting when it performs great on your training/validation set, but poorly on your test set (or new real-world data).

.. _precision:

Precision
  In the context of binary classification (Yes/No), precision measures the model's performance at classifying positive observations (i.e. "Yes"). In other words, when a positive value is predicted, how often is the prediction correct? We could game this metric by only returning positive for the single observation we are most confident in.

  .. math::

    P = \frac{True Positives}{True Positives + False Positives}

.. _recall:

Recall
  Also called sensitivity. In the context of binary classification (Yes/No), recall measures how "sensitive" the classifier is at detecting positive instances. In other words, for all the true observations in our sample, how many did we "catch." We could game this metric by always classifying observations as positive.

  .. math::

    R = \frac{True Positives}{True Positives + False Negatives}

.. _recall_vs_precision:

Recall vs Precision
  Say we are analyzing Brain scans and trying to predict whether a person has a tumor (True) or not (False). We feed it into our model and our model starts guessing.

  - **Precision** is the % of True guesses that were actually correct! If we guess 1 image is True out of 100 images and that image is actually True, then our precision is 100%! Our results aren't helpful however because we missed 10 brain tumors! We were super precise when we tried, but we didn’t try hard enough.

  - **Recall**, or Sensitivity, provides another lens which with to view how good our model is. Again let’s say there are 100 images, 10 with brain tumors, and we correctly guessed 1 had a brain tumor. Precision is 100%, but recall is 10%. Perfect recall requires that we catch all 10 tumors!

.. _regression:

Regression
  Predicting a continuous output (e.g. price, sales).

.. _regularization:

Regularization
  Contribute a definition!

.. _reinforcement_learning:

Reinforcement Learning
  Training a model to maximize a reward via iterative trial and error.

.. _segmentation:

Segmentation
  Contribute a definition!

.. _specificity:

Specificity
  In the context of binary classification (Yes/No), specificity measures the model's performance at classifying negative observations (i.e. "No"). In other words, when the correct label is negative, how often is the prediction correct? We could game this metric if we predict everything as negative.

  .. math::

    S = \frac{True Negatives}{True Negatives + False Positives}

.. _supervised_learning:

Supervised Learning
  Training a model using a labeled dataset.

.. _test_set:

Test Set
  A set of observations used at the end of model training and validation to assess the predictive power of your model. How generalizable is your model to unseen data?

.. _training_set:

Training Set
  A set of observations used to generate machine learning models.

.. _transfer_learning:

Transfer Learning
  Contribute a definition!

.. _type_1_error:

Type 1 Error
  False Positives. Consider a company optimizing hiring practices to reduce false positives in job offers. A type 1 error occurs when candidate seems good and they hire him, but he is actually bad.

.. _type_2_error:

Type 2 Error
  False Negatives. The candidate was great but the company passed on him.

.. _underfitting:

Underfitting
  Underfitting occurs when your model over-generalizes and fails to incorporate relevant variations in your data that would give your model more predictive power. You can tell a model is underfitting when it performs poorly on both training and test sets.

.. _uat:

Universal Approximation Theorem
  A neural network with one hidden layer can approximate any continuous function but only for inputs in a specific range. If you train a network on inputs between -2 and 2, then it will work well for inputs in the same range, but you can’t expect it to generalize to other inputs without retraining the model or adding more hidden neurons.

.. _unsupervised_learning:

Unsupervised Learning
  Training a model to find patterns in an unlabeled dataset (e.g. clustering).

.. _validation_set:

Validation Set
  A set of observations used during model training to provide feedback on how well the current parameters generalize beyond the training set. If training error decreases but validation error increases, your model is likely overfitting and you should pause training.

.. _variance:

Variance
  How tightly packed are your predictions for a particular observation relative to each other?

  - **Low variance** suggests your model is internally consistent, with predictions varying little from each other after every iteration.

  - **High variance** (with low bias) suggests your model may be overfitting and reading too deeply into the noise found in every training set.


**References**

* http://robotics.stanford.edu/~ronnyk/glossary.html
