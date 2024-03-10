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
        # Convert data frame slice into a numpy array of shape (sequence_length, data_dim)
        sequence = d.iloc[t:(t + sequence_length)].copy().to_numpy(dtype = np.float32)
        # Convert numpy array into a torch tensor of shape (sequence_length, data_dim)
        sequence = torch.from_numpy(sequence) 
        # Add batch dimension to the tensor
        sequence = torch.unsqueeze(sequence, 0) # the new shape is (1, sequence_length, data_dim)
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
        # Combine list of tensors with shape (1, seq_len, data_dim) to a tensor of shape (batch_size, seq_len, data_dim)
        batched_sequence = torch.cat(list_of_sequences[i:(i + batch_size)], dim = 0)
        list_of_batched_sequences.append(batched_sequence)
    # Combine the remainder of the list into a smaller batch
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
def create_batched_sequences(x, y, sequence_lengths, batch_size):
    list_of_xy_sequences = []
    for sequence_length in sequence_lengths:
        # Create x and y sequences with a given length
        list_of_x_sequences = create_fixed_length_sequences(x, sequence_length)
        list_of_y_sequences = create_fixed_length_sequences(y, sequence_length)
        # Shuffle x and y sequences in the same random order
        random_order = list(range(len(list_of_x_sequences)))
        random.shuffle(random_order)
        list_of_x_sequences = [list_of_x_sequences[i] for i in random_order]
        list_of_y_sequences = [list_of_y_sequences[i] for i in random_order]
        # Batch sequences with a given batch size
        list_of_batched_x_sequences = batch_sequences(list_of_x_sequences, batch_size)
        list_of_batched_y_sequences = batch_sequences(list_of_y_sequences, batch_size)
        # Add tuples of (x, y) batched sequences into the data list
        list_of_xy_sequences += list(zip(list_of_batched_x_sequences, list_of_batched_y_sequences))
    # Shuffle the final data to have a random order of sequence lengths
    random.shuffle(list_of_xy_sequences)
    return list_of_xy_sequences
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

Now let's put everything together. we will use the preprocessed file from the <a href="2024-03-01-data-preparation-for-stress-testing-model-part-1.md">part 1</a> to demonstrate how the data sequencing works. For training data, we will create sequences of 4, 5, and 6 lengths (we will use longer sequences later.)

```Python3
import pandas as pd
import torch
from make_sequences import *


d = pd.read_csv('../Data/historical_data_processed_2024.csv')

x_columns = ['real_gdp_growth', 'unemployment_rate', 'cpi_inflation_rate',
             'treasury_3m_rate_diff', 'treasury_5y_rate_diff', 'treasury_10y_rate_diff',
             'bbb_rate_diff', 'mortgage_rate_diff', 'prime_rate_diff', 'vix_diff',
             'dwcf_growth', 'hpi_growth', 'crei_growth',
             'q1', 'q2', 'q3', 'q4']
y_columns = ['real_disp_inc_growth']

d_train = d.iloc[:-4]
d_test = d.iloc[-4:] # last 4 quarters are for testing
xy_train = create_batched_sequences(d_train[x_columns], d_train[y_columns],
                                    sequence_lengths = [4, 5, 6], batch_size = 2)
xy_test = create_batched_sequences(d_test[x_columns], d_test[y_columns],
                                   sequence_lengths = [4], batch_size = 1)

print(f'Mini-batches in the training set: {len(xy_train)}')
# Each item of the list is a tuple of x and y mini-batches
print(f'Item (tuple) size in the training set: {len(xy_train[0])}')
print(f'Shape of the first x mini-batch (tensor): {xy_train[0][0].shape}')
print(f'Shape of the first y mini-batch (tensor): {xy_train[0][1].shape}')
print(f'Shape of the last x mini-batch (tensor): {xy_train[-1][0].shape}')
print(f'Shape of the last y mini-batch (tensor): {xy_train[-1][1].shape}')

print(f'Mini-batches in test set: {len(xy_test)}')
```

Output:

```
Mini-batches in the training set: 186
Item (tuple) size in the training set: 2
Shape of the first x mini-batch (tensor): torch.Size([2, 4, 17])
Shape of the first y mini-batch (tensor): torch.Size([2, 4, 1])
Shape of the last x mini-batch (tensor): torch.Size([2, 12, 17])
Shape of the last y mini-batch (tensor): torch.Size([2, 12, 1])
Mini-batches in test set: 1
```

In the next post, we will train a simple LSTM-based model to forecast real disposable income growth.
