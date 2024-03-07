---
title: "Data preparation for stress testing model, part 2"
date: 2024-03-02
---

We will use preprocessed time series data from <a href="2024-03-01-data-preparation-for-stress-testing-model-part-1.md">part 1</a> to create the training and testing data for deep learning sequence-to-sequence models.

In the context of the stress testing, we need to predict $y_1, y_2, ..., y_T$ given a hypothetical scenario $x_1, x_2, ..., x_T$. To make things more specific, here $y_t$ is a scalar representing the disposable income growth at time $t$, and $x_t$ is a vector of all other variables (GDP, Treasury rates, etc.) at time $t$. We will denote $N$ to be the size of the vector $x_t$.

We need to structure the historical data to reflect the setting needed for predicting with stress-testing models. We will slice the historical data into sequences using a fixed-length sliding window. For example, with a window size of 4, we will create the following sequences:
$$(x_0, x_1, x_2, x_3), (y_0, y_1, y_2, y_3)$$
$$(x_1, x_2, x_3, x_4), (y_1, y_2, y_3, y_4)$$
$$(x_2, x_3, x_4, x_5), (y_2, y_3, y_4, y_5)$$
$$...$$
Here, each sequence for $x$ variable is essentially a $4\times N$ matrix, where the rows (or first index) represent the time steps, and each column (the second index) is one of the features. Although each $y$ sequence is a vector, we will represent it as a one-column $4\times 1$ matrix to better align with the data structure needed in deep learning frameworks.

Next, given these fixed-length sequences of $x$ and $y$ variables, we need to further group them into batches (or mini-batches) since the Pytorch (and TensorFlow) require three-dimensional batched inputs for sequence-to-sequence models. Pytorch provides Tensor data type to conveniently handle these three-dimensional data structures (in general, Tensors can handle any number of dimensions.)

Let's start with a function that takes a time series (either $x$ or $y$) as an input and produces a list of fixed-length sequences. We will use a data frame to represent the series and will use tensors to represent the sequences. For now, each tensor represents a batch that contains only one sequence and has a shape (1, sequence_length, data_size). Later, we will group these tensors into batches of multiple sequences.

```Python3
import pandas as pd
import numpy as np
import os
import random
import torch


def create_fixed_length_sequences(d, sequence_length):
    list_of_sequences = []
    for t in range(d.shape[0] - sequence_length + 1):
        sequence = d.iloc[t:(t + sequence_length)].copy().to_numpy() # numpy array with shape (seq_len, data_dim)
        sequence = torch.from_numpy(sequence) # tensor with shape (seq_len, data_dim)
        sequence = torch.unsqueeze(sequence, 0) # tensor with shape (1, seq_len, data_dim)
        list_of_sequences.append(sequence)
    return list_of_sequences
```
The function implements the sliding window approach described earlier.

Next, we will group the sequence batches into larger batches. The function below takes as an input a list of single-sequence batches and combines them into batches of a given size. It also creates one smaller batch in the end in case the number of remaining sequences is smaller than the batch size.

```Python3
def batch_sequences(list_of_sequences, batch_size):
    list_of_batched_sequences = []
    n = len(list_of_sequences)
    for i in range(0, n, batch_size):
        # Convert list of tensors with shape (1, seq_len, data_dim) to a tensor of shape (batch_size, seq_len, data_dim)
        batched_sequence = torch.cat(list_of_sequences[i:(i + batch_size)], dim = 0)
        list_of_batched_sequences.append(batched_sequence)
    n_tail = n % batch_size
    if n_tail > 0:
        batched_sequence = torch.cat(list_of_sequences[-n_tail:], dim = 0)
        list_of_batched_sequences.append(batched_sequence)
    return list_of_batched_sequences
```
We will need many sequences to train a deep learning model. The historical quarterly time series data covers 1990-2023 period (33 years), which is essentially $33 \times 4 = 132$ time steps (and we still need to allocate some part of it to the test set.) By slicing it into sequences of length 4, we will end up with 129 sequences which is quite a small training set for deep learning models.

We will produce more sequences to alleviate this deficiency by varying the sequence length. For example, after producing 4-length sequences, we will produce 5-length sequences, then 6, etc. This will allow us to generate a relatively large training set. Note that, when it comes to batching, we can batch together only the sequences of the same length (we can't batch a 4-length sequence with a 6-length sequence.)

The function below implements the idea of generating multiple-length sequences and their batching given a time series (data frame), a list of sequence lengths, and a batch size.

```Python3
def create_batched_sequences(d, sequence_lengths, batch_size):
    list_of_all_sequences = []
    for sequence_length in sequence_lengths:
        list_of_sequences = create_fixed_length_sequences(d, sequence_length)
        random.shuffle(list_of_sequences)
        list_of_batched_sequences = batch_sequences(list_of_sequences, batch_size)
        list_of_all_sequences += list_of_batched_sequences
    return list_of_all_sequences
```

Noticed that the function randomly shuffles the list of same-length sequences before batching them. To make the result reproducible, we need to set the seed before calling this function. Since we will be using PyTorch to train models, let's implement a function that sets the seed for all packages we may use.

```Python3
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # In case running on the CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
```
