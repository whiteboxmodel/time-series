---
title: "Attention-based sequence-to-sequence model for time series with PyTorch"
categories: [Stress-testing, time-series, attention, sequence-to-sequence, time-series, PyTorch]
date: 2024-04-12
---

In the previous post, we developed an <a href="2024-03-31-LSTM-based-sequence-to-sequence-model.md">LSTM-based model</a> which performed remarkably well compared to the benchmark <a href="2024-03-21-benchmark-linear-regression-for-stress-testing.md">linear model</a>. Now let's build an attention-based time-series model to see if we can further improve the performance. As before, we will predict the unemployment rate given all other variables.

The idea of using the attention mechanism for this problem is inspired by language translation models. At a very high level, a translation model (such as from English to Chinese) converts words both in source and target languages into high-dimensional vectors called embeddings, builds a context vector by comparing the projections of the target embeddings with the projections of the source embeddings, then predicts the next word in the target language using the context vector.

The context vector represents the relevant information to predict the next word. This idea is conceptually similar to the hidden state of the LSTM (or GRU) cell. However, the context vector is more flexible since it can directly consume information from a much wider set of data points, while LSTM can use only the current input data point and the previous hidden state.

In our problem, we are given a time series of $x_t$ vectors representing macroeconomic variables at each time step $t$ and need to predict a one-dimensional time series $y_t$ of the unemployment rate. Since $y_t$ is one-dimensional, it doesn't make sense to take projections of it (and compare them to $x_t$ projections.) Instead, we will take $x_t$ projections and compare them with projections of past periods $x_{t - 1}$, $x_{t - 2}$, etc. to build the context vector. The latter represents the information from the current and past periods that can be helpful to predict $y_t$.

When we compare $x_t$ projections with its past projections, this type of attention is called self-attention. Note that in language translation models it is usual to compare against both past and future values to build a better context. However, we can only look back in time series to avoid data leaks.

Not let's go over the steps to build the model:

1. Batch normalization: The attention mechanism works better when the data is normalized. We will normalize the continuous variables using a `torch.nn.BatchNorm1d` layer. We will exclude the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables from normalization.

2. Positional encoding: The context vector built by the attention mechanism is based on a weighted average of past input projections. As such, attention does not account for the order of the past inputs. The general approach in large language models (LLMs) is to add sine and cosine-based encoding vectors to the input vectors (embeddings) to address this deficiency.

   In our problem, since the length and dimensionality of the input sequences are much smaller, we will append a feature to the sequences to encode the position. Note that the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables are not so efficient for positional encoding since the look-back period of the attention mechanism can be much longer than 4.

3. Reduce the input dimensionality: This optional step aims to reduce the dimensionality of the input vectors to a number with many multipliers. This provides more flexibility in selecting the number of attention heads since PyTorch requires the embedding size to be divisible by the number of attention heads.

4. Multihead attention: This layer implements the attention mechanism. It takes three sequences as inputs: queries, keys, and values, then returns a sequence of context vectors. We will set all three inputs to the same sequence obtained from the previous step.

   At each time step the attention layer can only attend to past observations as we forecast a time series. To enforce this, we need to pass an attention mask. The latter is a boolean matrix of `sequence_length x sequence_length` where for each `i`-th row the above diagonal elements are `True` indicating that they are masked (should not be attended).

5. Residual connection: At this step, we add the input of the attention layer (from step 3) to the output of the attention layer. This operation is called residual connection. It ensures that the information in the input flows through the rest of the network in addition to the context vectors created by the attention layer. Note that the input and the output of the attention layer have the same dimensionality.

6. Layer normalization: We apply layer normalization after the residual connection to normalize the data before processing further. Unlike batch normalization which normalizes each feature separately, the layer normalization normalizes the feature vector at each time step.

7. Feed-forward: Finally, we use a feed-forward network to process the sequences from the previous step and output the final prediction. This block consists of a dropout step, a linear layer, non-linear activation (ReLU), and another linear layer which reduces the output dimensionality to one.

Let's start the coding by importing the required modules:

```Python3
import pandas as pd
import torch
import matplotlib.pyplot as plt
# All custom libraries can be found in Code directory
from make_sequences import *
from torch_seed import *
from model_training import *
from predict_scenarios import *
```

Now let's define the class for the self-attention model:

```Python3
class SelfAttentionSequence(torch.nn.Module):
    def __init__(self, x_size, embed_size, num_heads, dropout_rate, max_sequence_size = 50):
        super(SelfAttentionSequence, self).__init__()
        # batch normalization of inputs excluding q1, q2, q3, and q4
        self.batch_norm = torch.nn.BatchNorm1d(x_size - 4)
        # a tensor for positional encoding
        self.pe = torch.linspace(0.0, 1.0, max_sequence_size, dtype = torch.float).reshape(max_sequence_size, 1) # (max_sequence_size, 1)
        # a linear layer to reduce input dimensionality (+1 is for positional encoding)
        self.initial_linear = torch.nn.Linear(in_features = x_size + 1, out_features = embed_size)
        # a dictionary to store and reuse the attention masks for each sequence length
        self.attn_masks = {}
        # self-attention module
        self.self_attn = torch.nn.MultiheadAttention(embed_dim = embed_size, num_heads = num_heads,
                                                     dropout = dropout_rate, batch_first = True,
                                                     add_bias_kv = False)
        # dropout to apply to the residual connection
        self.dropout = torch.nn.Dropout(p = dropout_rate)
        # layer normalization
        self.layer_norm = torch.nn.LayerNorm(embed_size)
        # feed-forward module
        self.dropout2 = torch.nn.Dropout(p = dropout_rate)
        self.linear2 = torch.nn.Linear(in_features = embed_size, out_features = embed_size)
        self.linear2 = torch.nn.utils.parametrizations.weight_norm(self.linear2) # keeps weights normalized
        self.non_linear = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(in_features = embed_size, out_features = 1)
```

The positional encoding is a linear feature that starts from `0` and increases by `1/50` at each time step. We will add it to the output of the batch normalization.

We use a dictionary to store the attention masks since the input sequences can have different lengths. When a new sequence length is encountered, we will create the attention mask and store it in the dictionary to reuse next time.

We apply weight normalization to one of the linear layers in the final feed-forward network. It normalizes the parameters of the layers which helps to regularize the model. Note that, unlike batch and layer normalization which work on the input sequence, parameter normalization works on the layer weights.
