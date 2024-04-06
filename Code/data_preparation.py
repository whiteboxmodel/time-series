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
    d = d.rename(columns = column_names)
    # Keep only the columns we need
    keep_columns = ['date', 'real_disp_inc_growth', 'real_gdp_growth', 'unemployment_rate',
                    'cpi_inflation_rate', 'treasury_3m_rate', 'treasury_5y_rate',
                    'treasury_10y_rate', 'bbb_rate', 'mortgage_rate',
                    'dwcf', 'hpi', 'crei', 'vix']
    d = d[keep_columns]
    # Variables for steepness of yield curve
    d['spread_treasury_10y_over_3m'] = d['treasury_10y_rate'] - d['treasury_3m_rate']
    d['spread_treasury_5y_over_3m'] = d['treasury_5y_rate'] - d['treasury_3m_rate']
    # First-order difference variables
    columns_to_diff = ['treasury_3m_rate', 'treasury_5y_rate',
                       'treasury_10y_rate', 'bbb_rate', 'mortgage_rate',
                       'vix']
    diff_column_names = [c + '_diff' for c in columns_to_diff]
    d_diff = d[columns_to_diff].diff(1)
    d_diff.rename(columns = dict(zip(columns_to_diff, diff_column_names)), inplace = True)
    d = pd.concat([d, d_diff.round(2)], axis = 1)
    d.drop(columns = columns_to_diff, inplace = True)
    # Growth variables
    columns_to_growth = ['dwcf', 'hpi', 'crei']
    growth_column_names = [c + '_growth' for c in columns_to_growth]
    d_growth = 100.0 * d[columns_to_growth].diff(1) / d[columns_to_growth].shift(1)
    d_growth.rename(columns = dict(zip(columns_to_growth, growth_column_names)), inplace = True)
    d = pd.concat([d, d_growth.round(2)], axis = 1)
    d.drop(columns = columns_to_growth, inplace = True)
    d.dropna(inplace = True)
    # One-hot encoding for quarters
    d['quarter'] = d['date'].str[-2:].str.lower()
    d = pd.get_dummies(d, columns = ['quarter'], prefix = [''], prefix_sep = '', dtype = float)
    return d
