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

1. Batch normalization: The attention mechanism works better when the data is normalized or standardized. We will normalize the continuous variables using a `torch.nn.BatchNorm1d` layer. We will exclude the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables from normalization.

2. Positional encoding: The context vector built by the attention mechanism is based on a weighted average of past input projections. As such, it does not account for the order of the past inputs. The general approach in large language models (LLMs) is to add sine and cosine-based encoding vectors to the input vectors (embeddings) to address this deficiency.

In our problem, since the length and dimensionality of the input sequences are much smaller, we will append a feature to the sequences to encode the position. Note that the one-hot encoded `q1`, `q2`, `q3`, and `q4` variables are not so efficient for positional encoding since the look-back period of the attention mechanism can be much longer than 4.

3. Reduce the input dimensionality: The purpose of this optional step is to shrink down the dimensionality of the input vectors to a number with many multipliers. This helps select the number of attention heads since PyTorch requires the embedding size to be divisible by the number of attention heads.
