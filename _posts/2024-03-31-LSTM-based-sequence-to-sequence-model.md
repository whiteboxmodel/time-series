---
title: "LSTM-based sequence-to-sequence model for time series with PyTorch"
categories: [Stress-testing, time-series, LSTM, sequence-to-sequence, PyTorch]
date: 2024-03-31
---

In the <a href="2024-03-21-benchmark-linear-regression-for-stress-testing.md">previous post</a>, we developed a linear model which didn't perform well. Now let's build an LSTM-based model to see if we can achieve a better performance.

The main advantage of LSTM is that it has a state where it stores relevant information from past periods. Then it uses the state and new inputs to predict the current period. We will not cover the inner structure of the LSTM cell (there are plenty of materials available on the web) but rather will use it as a building block in the model.

Along with LSTM, we will need a few other building blocks for the model which are described below:

1. Batch normalization: LSTM (and in general neural networks) works best when the data is normalized. We will normalize the data dynamically using a `torch.nn.BatchNorm1d` layer. We will normalize only the continuous variables and exclude the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables from normalization.

   The `torch.nn.BatchNorm1d` layer normalizes the second dimension. On the other hand, our data is structured into `(batch_size, sequence_size, x_size)` tensors, so we need to normalize the third dimension. To achieve this, we will swap the order of the second and the third dimensions using the `torch.permute` function, and then after batch normalization, we will change them back.

3.  Concatenate: After normalizing the continuous variables, we will concatenate them with the one-hot encoded variables to form one feature vector.

4.  Run through an LSTM layer: The data prepared in the previous step will be fed into a two-layer LSTM module. We will set the size of the LSTM state vector to 10. The output size of the LSTM layer is the same as its state's vector size, i.e. 10. We will apply dropout to the first LSTM layer to reduce overfitting.

5.  Dropout: We will apply dropout to the output of the second LSM layer to further regularize the network.

6.  Output: We will apply a linear layer to transform the output of the LSTM layer into a one-dimensional sequence.
