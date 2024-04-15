---
title: "Attention-based sequence-to-sequence model for time series with PyTorch"
categories: [Stress-testing, time-series, attention, sequence-to-sequence, time-series, PyTorch]
date: 2024-04-12
---

In the previous post, we developed an <a href="2024-03-31-LSTM-based-sequence-to-sequence-model.md">LSTM-based model</a> which performed remarkably well compared to the benchmark <a href="2024-03-21-benchmark-linear-regression-for-stress-testing.md">linear model</a>. Now let's build an attention-based time-series model to see if we can further improve the performance. As before, we will predict the unemployment rate given all other variables.

The idea of using the attention mechanism for this problem is inspired by language translation models. At a very high level, a translation model (such as from English to Chinese) converts words both in source and target languages into high-dimensional vectors called embeddings, builds a context vector by comparing the projections of the target embeddings with the projections of the source embeddings, then predicts the next word using the context vector.

In our problem, we are given a time series of $x_t$ vectors representing macroeconomic variables at each time step $t$ and need to predict a one-dimensional time series $y_t$ of the unemployment rate. Since $y_t$ is one-dimensional, it doesn't make sense to take projections of it (and compare them to $x_t$ projections.) Instead, we will take $x_t$ projections and compare them with projections of past periods $x_{t - 1}$, $x_{t - 2}$, etc. to build the context vector. The latter represents the information from the current and past periods that can be helpful to predict $y_t$.

The type of attention, when we compare $x_t$ projections with its past projections, is called self-attention. Note that in language translation models it is usual to compare against both past and future values to build a better context. However, we can only look back in time series to avoid data leaks.
