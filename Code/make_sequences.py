import pandas as pd
import numpy as np
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


def create_batched_sequences(X, y, sequence_lengths, batch_size):
    list_of_xy_sequences = []
    for sequence_length in sequence_lengths:
        list_of_x_sequences = create_fixed_length_sequences(X, sequence_length)
        list_of_y_sequences = create_fixed_length_sequences(y, sequence_length)
        random_order = list(range(len(list_of_x_sequences)))
        random.shuffle(random_order)
        list_of_x_sequences = [list_of_x_sequences[i] for i in random_order]
        list_of_y_sequences = [list_of_y_sequences[i] for i in random_order]
        list_of_batched_x_sequences = batch_sequences(list_of_x_sequences, batch_size)
        list_of_batched_y_sequences = batch_sequences(list_of_y_sequences, batch_size)
        list_of_xy_sequences += list(zip(list_of_batched_x_sequences, list_of_batched_y_sequences))
    return list_of_xy_sequences
