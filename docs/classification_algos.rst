.. _classification_algos:

=========================
Classification Algorithms
=========================

Classification problems is when our output Y is always in categories like positive vs negative in terms of sentiment analysis, dog vs cat in terms of image classification and disease vs no disease in terms of medical diagnosis.

Bayesian
=======

Overlaps..

Boosting
========

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Decision Trees
==============

ID3 decision tree: `code example <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/id3_decision_tree_simple.py>`__

K-Nearest Neighbor
==================
.. rubric:: Introduction

K-Nearest Neighbor is a supervised learning algorithm both for classification and regression. The principle is to find the predefined number of training samples closest to the new point, and predict the label from these training samples[1].

For example, when a new point comes, the algorithm will follow these steps:

1. Calculate the Euclidean distance between the new point and all training data
2. Pick the top-K closest training data
3. For regression problem, take the average of the labels as the result; for classification problem, take the most common label of these labels as the result.

.. rubric:: Code

Below is the Numpy implementation of K-Nearest Neighbor function. Refer to `code example <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/knn.py>`__ for details.

.. code-block:: python

    def KNN(training_data, target, k, func):
        """
        training_data: all training data point
        target: new point
        k: user-defined constant, number of closest training data
        func: functions used to get the the target label
        """
        # Step one: calculate the Euclidean distance between the new point and all training data
        neighbors= []
        for index, data in enumerate(training_data):
            # distance between the target data and the current example from the data.
            distance = euclidean_distance(data[:-1], target)
            neighbors.append((distance, index))

        # Step two: pick the top-K closest training data
        sorted_neighbors = sorted(neighbors)
        k_nearest = sorted_neighbors[:k]
        k_nearest_labels = [training_data[i][1] for distance, i in k_nearest]

        # Step three: For regression problem, take the average of the labels as the result;
        #             for classification problem, take the most common label of these labels as the result.
        return k_nearest, func(k_nearest_labels)
..


Logistic Regression
===================

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

Random Forests
==============

Random Forest Classifier using ID3 Tree: `code example <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/random_forest_classifier.py>`__

Support Vector Machines
=======================

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__



.. rubric:: References

.. [1] https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification



