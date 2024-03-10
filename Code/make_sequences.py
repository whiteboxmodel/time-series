import pandas as pd
import numpy as np
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
