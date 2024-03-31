import pandas as pd
import torch
import matplotlib.pyplot as plt
from make_sequences import *
from torch_seed import *
from model_training import *


d = pd.read_csv('../Data/historical_data_processed_2024.csv')

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

set_all_seeds(1) # ensure the prepared data and overall results are reproducible
xy_train = create_batched_sequences(d_train[x_columns], d_train[y_columns],
                                    sequence_lengths = [1], batch_size = 4)
xy_test = create_batched_sequences(d_test[x_columns], d_test[y_columns],
                                   sequence_lengths = [1], batch_size = 4)

print(f'Mini-batches in the training set: {len(xy_train)}')
# Each item of the list is a tuple of x and y mini-batches
print(f'Item (tuple) size in the training set: {len(xy_train[0])}')
print(f'Shape of the first x mini-batch (tensor): {xy_train[0][0].shape}')
print(f'Shape of the first y mini-batch (tensor): {xy_train[0][1].shape}')
print(f'Shape of the last x mini-batch (tensor): {xy_train[-1][0].shape}')
print(f'Shape of the last y mini-batch (tensor): {xy_train[-1][1].shape}')

print(f'Mini-batches in test set: {len(xy_test)}')


class LinearModel(torch.nn.Module):
    def __init__(self, x_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features = x_size, out_features = 1)
    
    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearModel(x_size)
#loss_fn = torch.nn.L1Loss() # Mean Absolute Error (MAE)
loss_fn = torch.nn.MSELoss() # Mean Squared Error (MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
n_epochs = 600

train_losses, test_losses = train_model(xy_train, xy_test, model, loss_fn, optimizer, n_epochs)
plot_loss_history(train_losses, test_losses)

# Create one long array (train + test) and predict with it
model.train(mode = False)
x_all = create_fixed_length_sequences(d[x_columns], sequence_length = d.shape[0])[0]  # take the first and only tensor from the returned list
y_all = model(x_all)
y_all = y_all[0, :, 0].detach().numpy()

# Plot the actual and prediction
d_all = d[['date'] + y_columns].copy()
d_all['pred'] = y_all
d_all.plot(x = 'date', y = ['unemployment_rate', 'pred'], grid = True, rot = 45, xlabel = '', title = 'Model fit')
plt.tight_layout()
plt.show()

# Scenario prediction
from predict_scenarios import *

# These are preprocessed scenario files (see part 1 blog)
scenario_files = {'Base': '../Data/Base_data_processed_2024.csv',
                  'SA': '../Data/SA_data_processed_2024.csv'}
# Load scenario files into a dictionary
d_scenarios = load_scenarios(scenario_files, start_date = '2024 Q1')
# Predict with scenarios
d_forecast = predict_scenarios(d_scenarios, x_columns, y_columns, model)
# Plot each scenario
plot_scenario_forecasts(d_forecast, y_label = 'Unemployment rate')
