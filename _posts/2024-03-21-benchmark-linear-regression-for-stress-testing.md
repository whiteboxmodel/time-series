---
title: "Benchmark linear regression for stress testing"
categories: [Stress-testing, time-series, linear-regression, PyTorch]
date: 2024-03-21
---

Let's develop a benchmark linear regression model using the data prepared in <a href="2024-03-01-data-preparation-for-stress-testing-model-part-1.md">part 1</a> and sequence preparation functions created in <a href="2024-03-01-data-preparation-for-stress-testing-model-part-2.md">part 2</a>. Note that we don't really need to structure the data into sequences with a three-dimensional `(batch_size, sequence_size, data_size)` shape for linear regression. We could have only used the `(batch_size, data_size)` shape. However, we will use the three-dimensional shape with `sequence_size = 1` to establish a framework for sequence-to-sequence models.

We will start by loading the libraries:

```Python3
import pandas as pd
import torch
import matplotlib.pyplot as plt
from make_sequences import *
from torch_seed import *
```

The libraries `make_sequences` and `torch_seed` can be found in the <a href="../Code">`Code`</a> directory of this repository.

Next, we read the data, define the x and y variables, and split the data into the training and test sets:

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

In theory, we need a separate validation set to control overfitting and perform hyperparameter tuning, however, since the data is short, we will reuse the test set as a validation set.

Now, let's structure the training and testing data to use for model training:

```Python3
set_all_seeds(1)
xy_train = create_batched_sequences(d_train[x_columns], d_train[y_columns],
                                    sequence_lengths = [1], batch_size = 4)
xy_test = create_batched_sequences(d_test[x_columns], d_test[y_columns],
                                   sequence_lengths = [1], batch_size = 4)
```

The x variables in the training/testing data are formatted into a `(4, 1, x_size)` shape, where 4 is the batch size, 1 is the sequence length (essentially each observation is considered independently of its previous ones), and `x_size` is the number of x variables (features). Similarly, the y variable is formatted into a `(4, 1, 1)` shape (the last dimension is 1 since we are predicting only the unemployment rate).

We will use the PyTorch framework to create a shallow neural network for a linear regression model:

```Python3
class LinearModel(torch.nn.Module):
    def __init__(self, x_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features = x_size, out_features = 1)
    
    def forward(self, x):
        out = self.linear(x)
        return out
```

We could have created a linear regression model directly using  `torch.nn.Sequential`, however, again, to build up the framework for sequence-to-sequence models, we use a slightly longer subclassing approach. The `LinearModel` class defined above is derived from `torch.nn.Module` which is a base class for all PyTorch models. In the constructor of the `LinearModel` class, we create a linear layer with `x_size` inputs and 1 output.

The `forward` function of the class performs forward pass. It only calls the linear layer to perform `W*x+b` operation where `W` and `b` are learnable (trainable) parameters.

To train the model, we need to create a loss function, and an optimizer:

```Python3
model = LinearModel(x_size)
loss_fn = torch.nn.MSELoss() # Mean Squared Error (MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
n_epochs = 600
```

Next, we define two functions to train and test the model.

```Python3
def train_epoch(xy_train, model, loss_fn):
    model.train(mode = True) # switch to the training mode
    total_loss = 0.0
    for x, y in xy_train:
        model.zero_grad()
        y_hat = model(x) # preadit with the current model parameters
        loss = loss_fn(y_hat, y) # calculate the loss (error) for the prediction
        loss.backward() # backpropagation - calculates gradients
        optimizer.step() # updates gradients to reduce the loss
        total_loss += loss.item()
    total_loss /= len(xy_train)
    return total_loss

def test_epoch(xy_test, model, loss_fn):
    model.train(mode = False) # switch to evaluation mode, do not calculate gradients
    total_loss = 0.0
    for x, y in xy_test:
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
    total_loss /= len(xy_test)
    return total_loss
```

These functions will be called at each epoch of the training procedure defined below:

```Python3
def train_model(xy_train, xy_test, model, loss_fn, n_epochs):
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        train_loss = train_epoch(xy_train, model, loss_fn)
        test_loss = test_epoch(xy_test, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_loss, test_loss = round(train_loss, 3), round(test_loss, 3)
        print(f'Epoch {epoch}, train loss - {train_loss}, test loss - {test_loss}')
    return train_losses, test_losses
```

The model training function returns training and testing losses for each epoch. They are useful to assess how well the model is trained and if there is any evidence of overfitting or underfitting. We define a function to print the training and testing losses:

```Python3
def plot_loss_history(train_losses, val_losses):
    plt.plot(train_losses, label = 'train_loss', color = 'blue')
    plt.plot(val_losses, label = 'val_loss', color = 'green')
    plt.legend()
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
```

Now let's train the model and plot the losses:

```Python3
train_losses, test_losses = train_model(xy_train, xy_test, model, loss_fn, n_epochs)
plot_loss_history(train_losses, test_losses)
```
