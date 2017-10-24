.. _deep_learning:

=============
Deep Learning
=============

.. contents:: :local:

Autoencoder
===========

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

- `Deep Learning Book <http://www.deeplearningbook.org/contents/autoencoders.html>`_


CNN
===

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__

- `Deep Learning Book <http://www.deeplearningbook.org/contents/convnets.html>`_


GAN
===

- `Deep Learning Book <http://www.deeplearningbook.org/contents/generative_models.html>`_


RNN
===

Description of RNN use case and basic architecture.

.. image:: images/rnn.png
      :align: center

.. rubric:: Model

.. literalinclude:: ../code/rnn.py   
      :pyobject: RNN
      
.. rubric:: Training

In this example, our input is a list of last names, where each name is 
a variable length array of one-hot encoded characters. Our target is is a list of
indices representing the class (language) of the name.

1. For each input name..
2. Initialize the hidden vector
3. Loop through the characters and predict the class
4. Pass the final character's prediction to the loss function
5. Backprop and update the weights

.. literalinclude:: ../code/rnn.py
      :pyobject: train

.. rubric:: Further reading

- `Jupyter notebook <https://github.com/bfortuner/ml-cheatsheet/notebooks/rnn.ipynb>`_
- `Deep Learning Book <http://www.deeplearningbook.org/contents/rnn.html>`_


VAE
===

.. rubric:: References

.. [1] Example
