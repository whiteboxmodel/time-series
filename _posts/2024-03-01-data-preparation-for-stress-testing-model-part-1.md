---
title: "Data preparation for stress testing model, part 1"
date: 2024-03-01
---

Fed released <a href="https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2024.htm">stress testing</a> scenarios, and I thought it was an interesting opportunity to try deep learning models to forecast with these scenarios. Particularly, sequence-to-sequence models are good candidates to use in this setting since both historical and scenario data are time series.

The historical data is a quarterly time series of important macroeconomic variables, such as Treasury rates, GDP, etc. Fed provides two scenarios, base and severely adverse, which have the same time step and macroeconomic variables.

We will train our deep learning models on historical data and forecast using the two scenarios. What exactly are we forecasting? Well, in real life, we would forecast mortgage prepayment, credit card default, or some other risk-bearing quantity, but in this exercise, we will keep things simple by forecasting disposable income growth given all other variables.

Let's start with the data preparation.

First, VIX is available starting from 1990, so we will truncate the data to start from 1990.

Second, the data contains multiple pairs of similar series, such as nominal and real GDP (the real series are inflation-adjusted.) We will drop the nominal version of the series and use only the real ones. Note that the data still contains the inflation (CPI) series, so there shouldn't be any loss of information by keeping only the real series.

Third, we will convert the home price, commercial real estate, and Dow Jones indexes into growth to eliminate the trend. For all other non-growth variables, such as Treasury rates and VIX, we will take the first-order difference.

Finally, we will introduce quarter indicators, by using one-hot encoding (4 additional variables.) Even though all relevant series are seasonality adjusted, the quarter indicators still can be useful since they provide a relative positional encoding. This means that within a year (not necessarily a calendar year) the model can figure out how far apart any two observations (quarters) are.

```Python3
import pandas as pd

def prepare_data(d: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    column_names = {'Scenario Name': 'scenario',
                    'Date': 'date',
                    'Real GDP growth': 'real_gdp_growth',
                    'Nominal GDP growth': 'nominal_gdp_growth',
                    'Real disposable income growth': 'real_disp_inc_growth',
                    'Nominal disposable income growth': 'nominal_disp_inc_growth',
                    'Unemployment rate': 'unemployment_rate',
                    'CPI inflation rate': 'cpi_inflation_rate',
                    '3-month Treasury rate': 'treasury_3m_rate',
                    '5-year Treasury yield': 'treasury_5y_rate',
                    '10-year Treasury yield': 'treasury_10y_rate',
                    'BBB corporate yield': 'bbb_rate',
                    'Mortgage rate': 'mortgage_rate',
                    'Prime rate': 'prime_rate',
                    'Dow Jones Total Stock Market Index (Level)': 'dwcf',
                    'House Price Index (Level)': 'hpi',
                    'Commercial Real Estate Price Index (Level)': 'crei',
                    'Market Volatility Index (Level)': 'vix'}
    d.rename(columns = column_names, inplace = True)
    # Keep only the columns we need
    keep_columns = ['date', 'real_disp_inc_growth', 'real_gdp_growth', 'unemployment_rate',
                    'cpi_inflation_rate', 'treasury_3m_rate', 'treasury_5y_rate',
                    'treasury_10y_rate', 'bbb_rate', 'mortgage_rate',
                    'prime_rate', 'dwcf', 'hpi', 'crei', 'vix']
    d = d[keep_columns]
    # First-order difference variables
    columns_to_diff = ['treasury_3m_rate', 'treasury_5y_rate',
                       'treasury_10y_rate', 'bbb_rate', 'mortgage_rate',
                       'prime_rate', 'vix']
    diff_column_names = [c + '_diff' for c in columns_to_diff]
    d_diff = d[columns_to_diff].diff(1).rename(columns = dict(zip(columns_to_diff, diff_column_names)))
    d = pd.concat([d, d_diff.round(2)], axis = 1)
    d.drop(columns = columns_to_diff, inplace = True)
    # Growth variables
    columns_to_growth = ['dwcf', 'hpi', 'crei']
    growth_column_names = [c + '_growth' for c in columns_to_growth]
    d_growth = 100.0 * d[columns_to_growth].diff(1) / d[columns_to_growth].shift(1)
    d_growth.rename(columns = dict(zip(column_names, diff_column_names)), inplace = True)
    d = pd.concat([d, d_growth.round(2)], axis = 1)
    d.drop(columns = columns_to_growth, inplace = True)
    d.dropna(inplace = True)
    # One-hot encoding for quarters
    d['quarter'] = d['date'].str[-2:].str.lower()
    d = pd.get_dummies(d, columns = ['quarter'], prefix = [''], prefix_sep = '', dtype = float)
    
    return d
```

Let's call this function

```Python3
d = pd.read_csv('../Data/2024-Table_2A_Historic_Domestic.csv')
d = prepare_data(d)

print(d.columns.values)
print(d.shape)
print('N/A count: {}'.format(d.isna().sum().sum()))
print(d.describe())

d.to_csv('../Data/historical_data_processed_2024.csv', index = False)
```

Later, we will use the same function for scenario preprocessing.
