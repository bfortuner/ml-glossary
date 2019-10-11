.. _layers:

======
Layers
======

.. contents:: :local:


BatchNorm
---------

BatchNorm accelerates convergence by reducing internal covariate shift inside each batch.
If the individual observations in the batch are widely different, the gradient
updates will be choppy and take longer to converge.

The batch norm layer normalizes the incoming activations and outputs a new batch
where the mean equals 0 and standard deviation equals 1. It subtracts the mean
and divides by the standard deviation of the batch.

.. rubric:: Code

Code example from `Agustinus Kristiadi <https://wiseodd.github.io/techblog/2016/07/04/batchnorm/>`_

.. literalinclude:: ../code/layers.py
      :pyobject: BatchNorm

.. rubric:: Further reading

- `Original Paper <https://arxiv.org/abs/1502.03167>`_
- `Implementing BatchNorm in Neural Net <https://wiseodd.github.io/techblog/2016/07/04/batchnorm/>`_
- `Understanding the backward pass through Batch Norm <https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html>`_


Convolution
-----------

In CNN, a convolution is a linear operation that involves multiplication of weight (kernel/filter) with the input and it does most of the heavy lifting job.

Convolution layer consists of 2 major component 1. Kernel(Filter) 2. Stride 

1. Kernel (Filter): A convolution layer can have more than one filter. The size of the filter should be smaller than the size of input dimension. It is intentional as it allows filter to be applied multiple times at difference point (position) on the input.Filters are helpful in understanding and identifying important features from given input. By applying different filters (more than one filter) on the same input helps in extracting different features from given input. Output from multiplying filter with the input gives Two dimensional array. As such, the output array from this operation is called "Feature Map".  

2. Stride: This property controls the movement of filter over input. when the value is set to 1, then filter moves 1 column at a time over input. When the value is set to 2 then the filer jump 2 columns at a time as filter moves over the input. 


.. rubric:: Code

.. code-block:: python

      # this code demonstate on how Convolution works
      # Assume we have a image of 4 X 4 and a filter fo 2 X 2 and Stride = 1 
      
      def conv_filter_ouput(input_img_section,filter_value):
            # this method perfromas the multiplication of input and filter 
            # returns singular value

            value = 0 
            for i in range(len(filter_value)):
                  for j in range(len(filter_value[0])):
                        value = value + (input_img_section[i][j]*filter_value[i][j])
            return value

      img_input = [[260.745, 261.332, 112.27 , 262.351],
       [260.302, 208.802, 139.05 , 230.709],
       [261.775,  93.73 , 166.118, 122.847],
       [259.56 , 232.038, 262.351, 228.937]]   

      filter = [[1,0],
         [0,1]]
      
      filterX,filterY = len(filter),len(filter[0])
      filtered_result = [] 
      for i in range(0,len(img_mx)-filterX+1):
      clm = []
      for j in range(0,len(img_mx[0])-filterY+1):
            clm.append(conv_filter_ouput(img_mx[i:i+filterX,j:j+filterY],filter))
      filtered_result.append(clm)
      
      print(filtered_result)

.. image:: images/cnn_filter_output.png
      :align: center

.. rubric:: Further reading
- `cs231n reference  <http://cs231n.github.io/convolutional-networks/>`_

Dropout
-------

A dropout layer takes the output of the previous layer's activations and randomly sets a certain fraction (dropout rate) of the activatons to 0, cancelling or 'dropping' them out.

It is a common regularization technique used to prevent overfitting in Neural Networks.

.. image:: images/dropout_net.png
      :align: center

The dropout rate is the tunable hyperparameter that is adjusted to measure performance with different values. It is typically set between 0.2 and 0.5 (but may be arbitrarily set).

Dropout is only used during training; At test time, no activations are dropped, but scaled down by a factor of dropout rate. This is to account for more units being active during test time than training time.

For example:

 - A layer in a neural net outputs a tensor (matrix) A of shape (batch_size, num_features).
 - The dropout rate of the layer is set to 0.5 (50%).
 - A random 50% of the values in A will be set to 0.
 - These will then be multiplied with the weight matrix to form the inputs to the next layer.

The premise behind dropout is to introduce noise into a layer in order to disrupt any interdependent learning or coincidental patterns that may occur between units in the layer, that aren't significant.

.. rubric:: Code

.. code-block:: python

      # layer_output is a 2D numpy matrix of activations

      layer_output *= np.random.randint(0, high=2, size=layer_output.shape) # dropping out values

      # scaling up by dropout rate during TRAINING time, so no scaling needs to be done at test time
      layer_output /= 0.5 
      # OR
      layer_output *= 0.5 # Scaling down during TEST time.

.. [2]

This results in the following operation.

.. image:: images/dropout.png
      :align: center

All reference, images and code examples, unless mentioned otherwise, are from section 4.4.3 of `Deep Learning for Python <https://www.manning.com/books/deep-learning-with-python>`_ by François Chollet. 

.. [2]


Linear
------

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


LSTM
----

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


Pooling
-------

Max and average pooling layers.

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


RNN
---

Be the first to `contribute! <https://github.com/bfortuner/ml-cheatsheet>`__


.. rubric:: References

.. [1] http://www.deeplearningbook.org/contents/convnets.html
.. [2] “4.4.3, Fundamentals of Machine Learning: Adding Dropout.” `Deep Learning for Python <https://www.manning.com/books/deep-learning-with-python>`_, by Chollet, François. Manning Publications Co., 2018, pp. 109–110.
