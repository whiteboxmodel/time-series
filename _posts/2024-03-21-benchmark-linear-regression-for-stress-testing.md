---
title: "Benchmark linear regression for stress testing"
categories: [Stress-testing, time-series, linear-regression, PyTorch]
date: 2024-03-21
---

Let's develop a benchmark linear regression model using the data prepared in <a href="2024-03-01-data-preparation-for-stress-testing-model-part-1.md">part 1</a> and sequence preparation functions created in <a href="2024-03-01-data-preparation-for-stress-testing-model-part-2.md">part 2</a>. Note that we don't really need to structure the data in a three-dimensional `(batch_size, sequence_size, data_size)` shape for linear regression. We could have only used the `(batch_size, data_size)` shape. However, to establish a framework for sequence-to-sequence models, we will use the three-dimensional shape with `sequence_size = 1`.

We will start by loading the libraries:

```Python3
import pandas as pd
import torch
import matplotlib.pyplot as plt
from make_sequences import *
from torch_seed import *
```

The libraries `make_sequences` and `torch_seed` can be found in the `Code` directory of this repository.

Next, we read the data, define the x and y variables, and allocate data into the training and test sets:

```Python3
x_columns = ['real_disp_inc_growth', 'real_gdp_growth', 'cpi_inflation_rate',
             'spread_treasury_10y_over_3m', 'spread_treasury_5y_over_3m',
             'treasury_3m_rate_diff', 'treasury_5y_rate_diff', 'treasury_10y_rate_diff',
             'bbb_rate_diff', 'mortgage_rate_diff', 'vix_diff',
             'dwcf_growth', 'hpi_growth', 'crei_growth',
             'q1', 'q2', 'q3', 'q4']
y_columns = ['unemployment_rate']
x_size = len(x_columns)
d_train = d.iloc[:-4]
d_test = d.iloc[-4:] # last 4 quarters are for testing
```

In theory, we need a separate validation set to control overfitting as well as to perform hyperparameter tuning, however, since the data is short, we will reuse the test set as a validation set.
