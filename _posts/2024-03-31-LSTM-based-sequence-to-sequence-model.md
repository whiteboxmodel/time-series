---
title: "LSTM-based sequence-to-sequence model for time series with PyTorch"
categories: [Stress-testing, time-series, LSTM, sequence-to-sequence, PyTorch]
date: 2024-03-31
---

In the previous post, we developed a linear model which didn't perform well. Now let's build an LSTM-based model to see if we can achieve a better performance.

We will not cover the inner structure of the LSTM cell (there are plenty of materials available on the web) but rather will use it as a building block in the model.

We will build the model using the following steps:

1. Normalize the data: LSTM (and in general neural networks) works best when the data is normalized. We will normalize the data dynamically using a `torch.nn.BatchNorm1d` layer. We will normalize only the continuous variables and exclude the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables from normalization.

2.  Concatenate: After normalizing the continuous variables, we will concatenate them with the one-hot encoded variables to form one feature vector for each observation.

3.  Run through an LSTM layer: The data prepared in the previous step will be fed into a two-layer LSTM module.

4.  Dropout: We will apply dropout the the output of the LSM layer to further regularize the network.

5.  Output: We will apply a linear layer to transform the output of the LSTM layer into a one-dimensional sequence.
