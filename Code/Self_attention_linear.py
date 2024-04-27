import pandas as pd
import torch
import matplotlib.pyplot as plt
# All custom libraries can be found in Code directory
from make_sequences import *
from torch_seed import *
from model_training import *
from predict_scenarios import *


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
        # self-attnetion module
        self.self_attn = torch.nn.MultiheadAttention(embed_dim = embed_size, num_heads = num_heads,
                                                     dropout = dropout_rate, batch_first = True,
                                                     add_bias_kv = False)
        # dropout to apply to the residual connection
        self.dropout = torch.nn.Dropout(p = dropout_rate)
        # # layer normalization
        # self.layer_norm = torch.nn.LayerNorm(embed_size)
        # feed-forward module
        self.dropout2 = torch.nn.Dropout(p = dropout_rate)
        self.linear2 = torch.nn.Linear(in_features = embed_size, out_features = embed_size)
        self.linear2 = torch.nn.utils.parametrizations.weight_norm(self.linear2) # keeps weights normalized
        self.non_linear = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(in_features = embed_size, out_features = 1)
    
    def normalize_input(self, sequence):
        # sequence size is (batch_size, sequence_size, x_size)
        x1 = sequence[:, :, :-4] # continuous variables
        x2 = sequence[:, :, -4:] # one-hot encoded variables: q1, q2, q3, and q4
        x1 = torch.permute(x1, (0, 2, 1)) # change to (batch_size, x1_size, sequence_size)
        x1 = self.batch_norm(x1) # normalize the continuous variables
        x1 = torch.permute(x1, (0, 2, 1)) # change back to (batch_size, sequence_size, x1_size)
        x = torch.cat([x1, x2], dim = 2) # concatenate continuous and one-hot encoded variables
        return x
    
    def add_positional_encoding(self, sequence):
        # truncate the encoding to the given sequence length: (max_sequence_size, 1) -> (sequence_size, 1)
        pe = self.pe[:sequence.shape[1], :]
        # repeat for every batch: (sequence_size, 1) -> (batch_size, sequence_size, 1)
        pe = pe.repeat(sequence.shape[0], 1, 1)
        # concatenate the feature vector with the positional encoding: (batch_size, sequence_size, x_size) -> (batch_size, sequence_size, x_size + 1)
        sequence_with_pe = torch.cat([sequence, pe], dim = 2)
        return sequence_with_pe
    
    def get_attention_mask(self, sequence_length):
        if sequence_length in self.attn_masks.keys():
            return self.attn_masks[sequence_length] # the mask for sequence_length already created
        attn_mask = torch.ones(sequence_length, sequence_length) # sequence_length x sequence_length matrix of ones
        attn_mask = torch.tril(attn_mask) # zero out above the diagonal
        attn_mask = attn_mask == 0.0 # convert to True/False, mask above diagonal which are future values
        self.attn_masks[sequence_length] = attn_mask # store the mask for repetative use
        return attn_mask
    
    def forward(self, sequence):
        # sequence size is (batch_size, sequence_size, x_size)
        x = self.normalize_input(sequence)
        x = self.add_positional_encoding(x)
        x = self.initial_linear(x) # map (batch_size, sequence_size, x_size + 1) to (batch_size, sequence_size, embed_size)
        x_prev = torch.clone(x) # save a copy for residual connection
        # self attention
        attn_mask = self.get_attention_mask(sequence_length = x.shape[1])
        x, _ = self.self_attn(query = x, key = x, value = x,
                              need_weights = False, attn_mask = attn_mask)
        # residual connection
        x = torch.add(x, self.dropout(x_prev))
        # # layer normalization
        # x = self.layer_norm(x)
        # feed-forward
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.non_linear(x)
        out = self.linear3(x) # final layer to create output with a shape (batch_size, sequence_size, 1)
        return out


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
                                    sequence_lengths = [4, 6, 8, 12, 16], batch_size = 8) # [4, 6, 8]
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

model = SelfAttentionSequence(x_size, embed_size = 12, num_heads = 4, dropout_rate = 0.1) # embed_size = 12, num_heads = 4
#loss_fn = torch.nn.L1Loss() # Mean Absolute Error (MAE)
loss_fn = torch.nn.MSELoss() # Mean Squared Error (MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
n_epochs = 400 #120 #

train_losses, test_losses = train_model(xy_train, xy_test, model, loss_fn, optimizer, n_epochs)
plot_loss_history(train_losses, test_losses)
plot_loss_history(train_losses, test_losses, start_epoch = 50)

# Create one long array (train + test) and predict with it
model.train(mode = False)
#x_all = create_fixed_length_sequences(d[x_columns], sequence_length = d.shape[0])[0]  # take the first and only tensor from the returned list
#y_all = model(x_all)
#y_all = y_all[0, :, 0] # Convert (batch_size, sequence_size, x_size) tensor to an array of sequence_size

y_all = predict_with_sliding_window(model, d, x_columns, sequence_length = 4) # in predict_scenarios module

d_all = d[['date'] + y_columns].copy()
d_all['pred'] = y_all.detach().numpy()

# Plot the actual and prediction
d_all.plot(x = 'date', y = ['unemployment_rate', 'pred'], grid = True, rot = 45, xlabel = '', title = 'Model fit')
plt.tight_layout()
plt.show()

# Scenario prediction
# These are preprocessed scenario files (see part 1 blog)
scenario_files = {'Base': '../Data/Base_data_processed_2024.csv',
                  'SA': '../Data/SA_data_processed_2024.csv'}
# Load scenario files into a dictionary
d_scenarios = load_scenarios(scenario_files, start_date = '2023 Q1') # starting from an earlier date for look back
# Predict with scenarios
d_forecast = predict_scenarios_with_sliding_window(model, d_scenarios, x_columns, y_columns, sequence_length = 16)
# Plot each scenario
plot_scenario_forecasts(d_forecast, y_label = 'Unemployment rate')
