# multi-step-ahead-predictions
This repository holds code for the following multi-step-ahead prediction methods: autoregressive integrated moving average (ARIMA) model, stacked RNN, stacked LSTM network, and stacked GRU. Each method's performance is measured in terms of prediction accuracy using MBE, MAPE, and RMSE, as well as average computational run-time. The methods are trained on the univariate time series of global horizontal irradiance (GHI) from satellite-measured data across varying locations throughout the United States.  We predict GHI for five years using both short and long-term prediction intervals.

The data contains satellite-measured irradiances as well as other meteorological parameters for the following locations:
  - ARM: ARM Southern Great Plains Facility, Oklahoma
  - BON: Bondville, Illinois
  - DRA: Desert Rock, Nevada
  - FPK: Fort Peck, Montana
  - GWN: Goodwin Creek, Mississippi
  - NREL: National Renewable Energy Laboratory, Golden, CO
  - PSU: Pennsylvania State University, Pennsylvania
  - SXF: Sioux Falls, South Dakota
  - TBL: Table Mountain, Boulder, Colorado

data.py includes data download and pre-processing steps.

The short-term and lon-term predictions files are labeled as 'One' and 'Multi', respectively. Furthermore, the files are distinguished by the GHI series used: 'monhtly', 'weekly', and 'daily'. Each series represents an averaged GHI series, as the original data is in thirty-minute intervals and computational intensive to train/test. 
