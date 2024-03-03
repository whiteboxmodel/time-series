---
title: "Data preparation for stress testing model, part 2"
date: 2024-03-02
---

We will use preprocessed time series data from <a href="2024-03-01-data-preparation-for-stress-testing-model-part-1.md">part 1</a> to create the training and testing data for deep learning sequence-to-sequence models.

In the context of the stress testing, we need to predict $y_1, y_2, ..., y_T$ given $x_1, x_2, ..., x_T$. To make things more specific, here $y_t$ is a scalar representing the disposable income growth at time $t$, and $x_t$ is a vector of all other variables (GDP, Treasury rates, etc.) at time $t$. We will denote $N$ to be the size of the vector $x_t$.

We need to structure the historical data to reflect the setting needed for stress-testing models. We will slice the historical data into sequences using a fixed-length sliding window. For example, with a window size of 4, we will create the following sequences:
$$(x_0, x_1, x_2, x_3), (y_0, y_1, y_2, y_3)$$
$$(x_1, x_2, x_3, x_4), (y_1, y_2, y_3, y_4)$$
$$(x_2, x_3, x_4, x_5), (y_2, y_3, y_4, y_5)$$
$$...$$
Here, each sequence for $x$ variable is essentially a $4\times N$ matrix, where the first index (or dimension) represents the time steps, and the second index is one of the variables at a time step. Although each $y$ sequence is a vector, we will represent it as a one-column $4\times 1$ matrix as well to better align with the data format expected in deep learning frameworks.

Next, given these fixed-length sequences of $x$ and $y$ variables, we need to further group them into batches (or mini-batches) since the Pytorch (and TensorFlow too) require three-dimensional batched inputs for sequence-to-sequence models. Pytorch provides Tensor data type to conveniently handle these three-dimensional data structures (Tensors can handle higher dimensions too.)
