---
title: "Data preparation for stress testing model, part 1"
date: 2024-03-01
---

Fed released stress testing scenarios, and I thought it was an interesting opportunity to try deep learning models to forecast with these scenarios. Particularly, sequence-to-sequence models are good candidates to use in this setting since both historical and scenario data are time series.

The historical data is a quarterly time series of important macroeconomic variables, such as Treasury rates, GDP, etc. Fed provides two scenarios, base and severely adverse, which have the same time step and macroeconomic variables.

We will train our deep learning models on historical data and forecast using the two scenarios. What exactly are we forecasting? Well, in real life, we would forecast mortgage prepayment, credit card default, or some other risk-bearing quantity, but in this exercise, we will keep things simple by forecasting disposable income growth given all other variables.

Let's start with the data preparation.

First, VIX is available starting from 1990, so we will truncate the data to start from 1990.

Second, the data contains multiple pairs of similar series, such as nominal and real GDP (the real series are inflation-adjusted.) We will drop the nominal version of the series and use only the real ones. Note that the data still contains the inflation (CPI) series, so there shouldn't be any loss of information by keeping only the real series.

Third, we will convert the home price, commercial real estate, and Dow Jones indexes into growth to eliminate the trend. For all other non-growth variables, such as Treasury rates and VIX, we will take the first-order difference.

Finally, we will introduce quarter indicators, by using one-hot encoding (4 additional variables.) Even though all relevant series are seasonality adjusted, the quarter indicators still can be useful since they provide a relative positional encoding. This means that within a year (not necessarily a calendar year) the model can know how far apart any two observations (quarters) are.
