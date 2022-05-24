.. _glossary:

========
Glossary
========

Definitions of common machine learning terms.

.. http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-arbitrary-locations

.. _glossary_accuracy:

Accuracy
  Percentage of correct predictions made by the model.

.. _glossary_algorithm:

Algorithm
  A method, function, or series of instructions used to generate a machine learning :ref:`model <glossary_model>`. Examples include linear regression, decision trees, support vector machines, and neural networks.

.. _glossary_attribute:

Attribute
  A quality describing an observation (e.g. color, size, weight). In Excel terms, these are column headers.

.. _glossary_bias_metric:

Bias metric
  What is the average difference between your predictions and the correct value for that observation?

  - **Low bias** could mean every prediction is correct. It could also mean half of your predictions are above their actual values and half are below, in equal proportion, resulting in low average difference.

  - **High bias** (with low variance) suggests your model may be underfitting and you're using the wrong architecture for the job.

.. _glossary_bias_term:

Bias term
  Allow models to represent patterns that do not pass through the origin. For example, if all my features were 0, would my output also be zero? Is it possible there is some base value upon which my features have an effect? Bias terms typically accompany weights and are attached to neurons or filters.

.. _glossary_categorical_variables:

Categorical Variables
  Variables with a discrete set of possible values. Can be ordinal (order matters) or nominal (order doesn't matter).

.. _glossary_classification:

Classification
  Predicting a categorical output.

  - **Binary classification** predicts one of two possible outcomes (e.g. is the email spam or not spam?)

  - **Multi-class classification** predicts one of multiple possible outcomes (e.g. is this a photo of a cat, dog, horse or human?)

.. _glossary_classification_threshold:

Classification Threshold
  The lowest probability value at which we're comfortable asserting a positive classification. For example, if the predicted probability of being diabetic is > 50%, return True, otherwise return False.

.. _glossary_clustering:

Clustering
  Unsupervised grouping of data into buckets.

.. _glossary_confusion_matrix:

Confusion Matrix
  Table that describes the performance of a classification model by grouping predictions into 4 categories.

  - **True Positives**: we *correctly* predicted they do have diabetes
  - **True Negatives**: we *correctly* predicted they don't have diabetes
  - **False Positives**: we *incorrectly* predicted they do have diabetes (Type I error)
  - **False Negatives**: we *incorrectly* predicted they don't have diabetes (Type II error)

.. _glossary_continuous_variables:

Continuous Variables
  Variables with a range of possible values defined by a number scale (e.g. sales, lifespan).

.. _glossary_convergence:

Convergence
  A state reached during the training of a model when the :ref:`loss <glossary_loss>` changes very little between each iteration.

.. _glossary_deduction:

Deduction
  A top-down approach to answering questions or solving problems. A logic technique that starts with a theory and tests that theory with observations to derive a conclusion. E.g. We suspect X, but we need to test our hypothesis before coming to any conclusions.

.. _glossary_deep_learning:

Deep Learning
  Deep Learning is derived from one machine learning algorithm called perceptron or multi layer perceptron that is gaining more and more attention nowadays because of its success in different fields like, computer vision to signal processing and medical diagnosis to self-driving cars. Like all other AI algorithms, deep learning is based on decades of research. Nowadays, we have more and more data and cheap computing power that make this algorithm really powerful to achieve state of the art accuracy. In modern world this algorithm is known as artificial neural network. Deep learning is much more accurate and robust compared to traditional artificial neural network. But it was highly influenced by machine learning's neural network and perceptron network.

.. _glossary_dimension:

Dimension
  Dimension for machine learning and data scientist is differ from physics, here Dimension of data means how much feature you have in you data ocean(data-set). e.g in case of object detection application, flatten image size and color channel(e.g 28*28*3) is a feature of the input set. In case of house price prediction (maybe) house size is the data-set so we call it 1 dimentional data.

.. _glossary_epoch:

Epoch
  An epoch describes the number of times the algorithm sees the entire data set.

.. _glossary_extrapolation:

Extrapolation
  Making predictions outside the range of a dataset. E.g. My dog barks, so all dogs must bark. In machine learning we often run into trouble when we extrapolate outside the range of our training data.

.. _glossary_false_positive_rate:

False Positive Rate
  Defined as

  .. math::

    FPR = 1 - Specificity = \frac{False Positives}{False Positives + True Negatives}

  The False Positive Rate forms the x-axis of the :ref:`ROC curve <glossary_roc_curve>`.

.. _glossary_feature:

Feature
  With respect to a dataset, a feature represents an :ref:`attribute <glossary_attribute>` and value combination. Color is an attribute. "Color is blue" is a feature. In Excel terms, features are similar to cells. The term feature has other definitions in different contexts.

.. _glossary_feature_selection:

Feature Selection
  Feature selection is the process of selecting relevant features from a data-set for creating a Machine Learning model.

.. _glossary_feature_vector:

Feature Vector
  A list of features describing an observation with multiple attributes. In Excel we call this a row.

.. _glossary_gradient_accumulation:

Gradient Accumulation
  A mechanism to split the batch of samples—used for training a neural network—into several mini-batches of samples that will be run sequentially. This is used to enable using large batch sizes that require more GPU memory than available.

.. _glossary_hyperparameters:

Hyperparameters
  Hyperparameters are higher-level properties of a model such as how fast it can learn (learning rate) or complexity of a model. The depth of trees in a Decision Tree or number of hidden layers in a Neural Networks are examples of hyper parameters.

.. _glossary_induction:

Induction
  A bottoms-up approach to answering questions or solving problems. A logic technique that goes from observations to theory. E.g. We keep observing X, so we infer that Y must be True.

.. _glossary_instance:

Instance
  A data point, row, or sample in a dataset. Another term for :ref:`observation <glossary_observation>`.

.. _glossary_label:

Label
  The "answer" portion of an :ref:`observation <glossary_observation>` in :ref:`supervised learning <glossary_supervised_learning>`. For example, in a dataset used to classify flowers into different species, the features might include the petal length and petal width, while the label would be the flower's species.

.. _glossary_learning_rate:

Learning Rate
  The size of the update steps to take during optimization loops like :doc:`gradient_descent`. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

.. _glossary_loss:

Loss
  Loss = true_value(from data-set)- predicted value(from ML-model)  The lower the loss, the better a model (unless the model has over-fitted to the training data). The loss is calculated on training and validation and its interpretation is how well the model is doing for these two sets. Unlike accuracy, loss is not a percentage. It is a summation of the errors made for each example in training or validation sets.

.. _glossary_machine_learning:

Machine Learning
   Mitchell (1997) provides a succinct definition: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." In simple language machine learning is a field in which human made algorithms have an ability learn by itself or predict future for unseen data.

.. _glossary_model:

Model
  A data structure that stores a representation of a dataset (weights and biases). Models are created/learned when you train an algorithm on a dataset.

.. _glossary_neural_networks:

Neural Networks
  Neural Networks are mathematical algorithms modeled after the brain's architecture, designed to recognize patterns and relationships in data. 

.. _glossary_normalization:

Normalization
  Restriction of the values of weights in regression to avoid overfitting and improving computation speed.

.. _glossary_noise:

Noise
  Any irrelevant information or randomness in a dataset which obscures the underlying pattern.

.. _glossary_null_accuracy:

Null Accuracy
  Baseline accuracy that can be achieved by always predicting the most frequent class ("B has the highest frequency, so lets guess B every time").

.. _glossary_observation:

Observation
  A data point, row, or sample in a dataset. Another term for :ref:`instance <glossary_instance>`.

.. _glossary_outlier:

Outlier
  An observation that deviates significantly from other observations in the dataset.

.. _glossary_overfitting:

Overfitting
  Overfitting occurs when your model learns the training data too well and incorporates details and noise specific to your dataset. You can tell a model is overfitting when it performs great on your training/validation set, but poorly on your test set (or new real-world data).

.. _glossary_parameters:

Parameters
  Parameters are properties of training data learned by training a machine learning model or classifier. They are adjusted using optimization algorithms and unique to each experiment. 

  Examples of parameters include:

  - weights in an artificial neural network
  - support vectors in a support vector machine
  - coefficients in a linear or logistic regression
  

.. _glossary_precision:

Precision
  In the context of binary classification (Yes/No), precision measures the model's performance at classifying positive observations (i.e. "Yes"). In other words, when a positive value is predicted, how often is the prediction correct? We could game this metric by only returning positive for the single observation we are most confident in.

  .. math::

    P = \frac{True Positives}{True Positives + False Positives}

.. _glossary_recall:

Recall
  Also called sensitivity. In the context of binary classification (Yes/No), recall measures how "sensitive" the classifier is at detecting positive instances. In other words, for all the true observations in our sample, how many did we "catch." We could game this metric by always classifying observations as positive.

  .. math::

    R = \frac{True Positives}{True Positives + False Negatives}

.. _glossary_recall_vs_precision:

Recall vs Precision
  Say we are analyzing Brain scans and trying to predict whether a person has a tumor (True) or not (False). We feed it into our model and our model starts guessing.

  - **Precision** is the % of True guesses that were actually correct! If we guess 1 image is True out of 100 images and that image is actually True, then our precision is 100%! Our results aren't helpful however because we missed 10 brain tumors! We were super precise when we tried, but we didn’t try hard enough.

  - **Recall**, or Sensitivity, provides another lens which with to view how good our model is. Again let’s say there are 100 images, 10 with brain tumors, and we correctly guessed 1 had a brain tumor. Precision is 100%, but recall is 10%. Perfect recall requires that we catch all 10 tumors!

.. _glossary_regression:

Regression
  Predicting a continuous output (e.g. price, sales).

.. _glossary_regularization:

Regularization
  Regularization is a technique utilized to combat the overfitting problem. This is achieved by adding a complexity term to the loss function that gives a bigger loss for more complex models 

.. _glossary_reinforcement_learning:

Reinforcement Learning
  Training a model to maximize a reward via iterative trial and error.

.. _glossary_roc_curve:

ROC (Receiver Operating Characteristic) Curve
  A plot of the :ref:`true positive rate <glossary_true_positive_rate>` against the :ref:`false positive rate <glossary_false_positive_rate>` at all :ref:`classification thresholds <glossary_classification_threshold>`. This is used to evaluate the performance of a classification model at different classification thresholds. The area under the ROC curve can be interpreted as the probability that the model correctly distinguishes between a randomly chosen positive observation (e.g. "spam") and a randomly chosen negative observation (e.g. "not spam").

.. _glossary_segmentation:

Segmentation
  It is the process of partitioning a data set into multiple distinct sets. This separation is done such that the members of the same set are similar to each otherand different from the members of other sets.

.. _glossary_specificity:

Specificity
  In the context of binary classification (Yes/No), specificity measures the model's performance at classifying negative observations (i.e. "No"). In other words, when the correct label is negative, how often is the prediction correct? We could game this metric if we predict everything as negative.

  .. math::

    S = \frac{True Negatives}{True Negatives + False Positives}

.. _glossary_supervised_learning:

Supervised Learning
  Training a model using a labeled dataset.

.. _glossary_test_set:

Test Set
  A set of observations used at the end of model training and validation to assess the predictive power of your model. How generalizable is your model to unseen data?

.. _glossary_training_set:

Training Set
  A set of observations used to generate machine learning models.

.. _glossary_transfer_learning:

Transfer Learning
  A machine learning method where a model developed for a task is reused as the starting point for a model on a second task. In transfer learning, we take the pre-trained weights of an already trained model (one that has been trained on millions of images belonging to 1000’s of classes, on several high power GPU’s for several days) and use these already learned features to predict new classes.

.. _glossary_true_positive_rate:

True Positive Rate
  Another term for :ref:`recall <glossary_recall>`, i.e.

  .. math::

    TPR = \frac{True Positives}{True Positives + False Negatives}

  The True Positive Rate forms the y-axis of the :ref:`ROC curve <glossary_roc_curve>`.

.. _glossary_type_1_error:

Type 1 Error
  False Positives. Consider a company optimizing hiring practices to reduce false positives in job offers. A type 1 error occurs when candidate seems good and they hire him, but he is actually bad.

.. _glossary_type_2_error:

Type 2 Error
  False Negatives. The candidate was great but the company passed on him.

.. _glossary_underfitting:

Underfitting
  Underfitting occurs when your model over-generalizes and fails to incorporate relevant variations in your data that would give your model more predictive power. You can tell a model is underfitting when it performs poorly on both training and test sets.

.. _glossary_uat:

Universal Approximation Theorem
  A neural network with one hidden layer can approximate any continuous function but only for inputs in a specific range. If you train a network on inputs between -2 and 2, then it will work well for inputs in the same range, but you can’t expect it to generalize to other inputs without retraining the model or adding more hidden neurons.

.. _glossary_unsupervised_learning:

Unsupervised Learning
  Training a model to find patterns in an unlabeled dataset (e.g. clustering).

.. _glossary_validation_set:

Validation Set
  A set of observations used during model training to provide feedback on how well the current parameters generalize beyond the training set. If training error decreases but validation error increases, your model is likely overfitting and you should pause training.

.. _glossary_variance:

Variance
  How tightly packed are your predictions for a particular observation relative to each other?

  - **Low variance** suggests your model is internally consistent, with predictions varying little from each other after every iteration.

  - **High variance** (with low bias) suggests your model may be overfitting and reading too deeply into the noise found in every training set.


.. rubric:: References

.. [1] http://robotics.stanford.edu/~ronnyk/glossary.html
.. [2] https://developers.google.com/machine-learning/glossary
