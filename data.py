import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from timezonefinder import TimezoneFinder
from datetime import datetime
from dateutil import tz
from calendar import isleap
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler

NREL_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/NREL_39.74-105.18_all.csv', header=0, parse_dates=[0], index_col=0)

ARM_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/ARM_36.60-97.49_all.csv', header=0, parse_dates=[0], index_col=0)

BON_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/BON_40.05-88.37_all.csv', header=0, parse_dates=[0], index_col=0)

DRA_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/DRA _36.62-116.02_all.csv', header=0, parse_dates=[0], index_col=0)

FPK_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/FPK_48.31-105.10_all.csv', header=0, parse_dates=[0], index_col=0)

GWN_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/GWN_34.25-89.87_all.csv', header=0, parse_dates=[0], index_col=0)

PSU_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/PSU_40.72-77.93_all.csv', header=0, parse_dates=[0], index_col=0)

SXF_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/SXF_43.73-96.62_all.csv', header=0, parse_dates=[0], index_col=0)

TBL_all_years = pd.read_csv('/wendianHome/bins/adenhard/nrel/TBL_40.12-105.24_all.csv', header=0, parse_dates=[0], index_col=0)

dataframes = [NREL_all_years, ARM_all_years, BON_all_years, DRA_all_years, FPK_all_years, GWN_all_years, PSU_all_years, SXF_all_years, TBL_all_years]

latitude = [39.742, 
            36.60406, 
            40.05192, 
            36.62373, 
            48.31,
            34.2547, 
            40.72012, 
            43.73403, 
            40.12498]
            
longitude = [-105.18, 
             -97.48525, 
             -88.37309, 
             -116.01947,
             -105.10,
             -89.8729, 
             -77.93085, 
             -96.62328, 
             -105.23680]

ts = []
for i in range(len(dataframes)):
  tf = TimezoneFinder()
  zone = tf.timezone_at(lng=longitude[i], lat=latitude[i])

  # convert UTC to local time
  data = dataframes[i]
  data = data.tz_localize('UTC')
  data = data.tz_convert(zone)
  data.index
  
  # remove night-time observations
  data = data.drop(data[data.solar_zenith_angle > 89.5].index)
  
  # Set the time series as the ghi variable
  ts.append(data['ghi'])

daily_totals = []
annual_averages = [] 
monthly_averages = []
weekly_averages = []

for i in range(len(ts)):
  daily_totals.append(((ts[i].resample('D').sum())/1000)/2)
  
  annual_averages_temp = daily_totals[i].resample('Y').mean()
  annual_averages.append(annual_averages_temp[1:])
  
  monthly_averages_temp = daily_totals[i].resample('M').mean()
  monthly_averages.append(monthly_averages_temp[1:])
  
  weekly_averages_temp = daily_totals[i].resample('W').mean()
  weekly_averages.append(weekly_averages_temp[1:])
  
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 # create a log transformed series 
def log_transform(dataset):
  return Series(np.log(dataset))
  
# undo log transform
def inverse_log(dataset):
  vals = dataset.values.astype(float)
  ind = dataset.index
  return Series(np.exp(vals), index=ind)
  
# remove trend in monthly means of daily totals data 
def remove_trend_monthly(dataset):
  # account for dataset monthly averages not starting at January
  k = dataset.index[0].month
  k = k -1
  for i in range(k):
    obs = pd.Series(np.nan)
    dataset = pd.concat([obs, dataset])
  # account for missing observations and fill with NaNs for correct shape
  while (len(dataset) % 12 != 0):
    obs = pd.Series(np.nan)
    dataset = dataset.append(obs)
  nrow = int(len(dataset)/12)
  temp = dataset.values.reshape(nrow,12)
  monthly_mean = np.nanmean(temp, 0)
  monthly_detrended = (temp - monthly_mean).flatten()
  return Series(monthly_detrended), monthly_mean
  
# add trend back in 
def inverse_trend_monthly(dataset, monthly_mean):
  nrow = int(len(dataset)/12)
  predicted = array(dataset).reshape(nrow,12)
  predicted = predicted + monthly_mean
  predicted = predicted.flatten()
  return Series(predicted)
  
# remove trend in weekly means of daily totals data 
def remove_trend_weekly(dataset):
  # account for dataset monthly averages not starting at January
  jan_1st = datetime(dataset.index[0].year, 1, 1)
  first_datapoint = datetime(dataset.index[0].year, dataset.index[0].month, dataset.index[0].day)
  days_offset = (jan_1st - first_datapoint).days
  num_missing_weeks = int(days_offset / 7)
  for i in range(num_missing_weeks):
    obs = pd.Series(np.nan)
    dataset = pd.concat([obs, dataset])
  # account for missing observations and fill with NaNs for correct shape
  while (len(dataset) % 52 != 0):
    obs = pd.Series(np.nan)
    dataset = dataset.append(obs)
  nrow = int(len(dataset)/52)
  temp = dataset.values.reshape(nrow,52)
  weekly_mean = np.nanmean(temp, 0)
  weekly_detrended = (temp - weekly_mean).flatten()
  return Series(weekly_detrended), weekly_mean
  
# add trend back in 
def inverse_trend_weekly(dataset, weekly_mean):
  nrow = int(len(dataset)/52)
  predicted = array(dataset).reshape(nrow,52)
  predicted = predicted + weekly_mean
  predicted = predicted.flatten()
  return Series(predicted)
  
# daily along same lines but issue is to account for leap year
def remove_trend_daily(dataset):
  # remove leap days
  for i in range(len(dataset.index)):
    if dataset.index[i].month == 2 and dataset.index[i].day == 29:
      dataset.drop(index=dataset.index[i])
  # account for dataset monthly averages not starting at January
  jan_1st = datetime(dataset.index[0].year, 1, 1)
  first_datapoint = datetime(dataset.index[0].year, dataset.index[0].month, dataset.index[0].day)
  days_offset = (jan_1st - first_datapoint).days
  for i in range(days_offset):
    obs = pd.Series(np.nan)
    dataset = pd.concat([obs, dataset])
  # account for missing observations and fill with NaNs for correct shape
  while (len(dataset) % 365 != 0):
    obs = pd.Series(np.nan)
    dataset = dataset.append(obs)
  nrow = int(len(dataset)/365)
  temp = dataset.values.reshape(nrow,365)
  daily_mean = np.nanmean(temp, 0)
  daily_detrended = (temp - daily_mean).flatten()
  return Series(daily_detrended), daily_mean
  
# add trend back in 
def inverse_trend_daily(dataset, daily_mean):
  nrow = int(len(dataset)/365)
  predicted = array(dataset).reshape(nrow,365)
  predicted = predicted + daily_mean
  predicted = predicted.flatten()
  return Series(predicted)
  
def inverse_transforms(forecasts, scalar, monthly_mean):
  inverted = list()
  for i in range(len(forecasts)):
    # create array from forecast
    forecast = array(forecasts[i])
    forecast = forecast.reshape(1, forecast.size)
    # invert scaling
    inv_scale = scalar.inverse_transform(forecast)
    inv_scale = inv_scale[0, :]
    # store
    inverted.append(inv_scale)
  # invert trend
  inverted = Series(inverted)
  inv_trend = inverse_trend(inverted, monthly_mean)
  # invert log
  inv_log = inverse_log(inv_trend)  
  return inv_log
  
# transform series into train and test sets for supervised learning 
def prepare_networks_monthly(series, n_test, val_size, n_lag, n_seq):
  # extract raw values
  raw_values = series
  # run log transform
  transform_series = log_transform(raw_values)
  # remove trend from the data
  detrended_series, monthly_mean = remove_trend_monthly(transform_series)
  detrended_values = detrended_series.values
  detrended_values = detrended_values.reshape(len(detrended_values), 1)
  # rescale values to -1, 1
  scalar = MinMaxScaler(feature_range=(-1,1))
  scaled_values = scalar.fit_transform(detrended_values)
  scaled_values = scaled_values.reshape(len(scaled_values), 1)
  # transform into supervised learning problem X, y
  supervised = series_to_supervised(scaled_values, n_lag, n_seq)
  supervised_values = supervised.values
  # split into train and test sets 
  train_full, test = supervised_values[0:-n_test], supervised_values[-n_test:]
  train, val = train_full[0:-val_size], train_full[-val_size:] 
  return scalar, train, train_full, val, test, monthly_mean
  
def prepare_networks_weekly(series, n_test, val_size, n_lag, n_seq):
  # extract raw values
  raw_values = series
  # run log transform
  transform_series = log_transform(raw_values)
  # remove trend from the data
  detrended_series, weekly_mean = remove_trend_weekly(transform_series)
  detrended_values = detrended_series.values
  detrended_values = detrended_values.reshape(len(detrended_values), 1)
  # rescale values to -1, 1
  scalar = MinMaxScaler(feature_range=(-1,1))
  scaled_values = scalar.fit_transform(detrended_values)
  scaled_values = scaled_values.reshape(len(scaled_values), 1)
  # transform into supervised learning problem X, y
  supervised = series_to_supervised(scaled_values, n_lag, n_seq)
  supervised_values = supervised.values
  # split into train and test sets 
  train_full, test = supervised_values[0:-n_test], supervised_values[-n_test:]
  train, val = train_full[0:-val_size], train_full[-val_size:] 
  return scalar, train, train_full, val, test, weekly_mean
  
def prepare_networks_daily(series, n_test, val_size, n_lag, n_seq):
  # extract raw values
  raw_values = series
  # run log transform
  transform_series = log_transform(raw_values)
  # remove trend from the data
  detrended_series, daily_mean = remove_trend_daily(transform_series)
  detrended_values = detrended_series.values
  detrended_values = detrended_values.reshape(len(detrended_values), 1)
  # rescale values to -1, 1
  scalar = MinMaxScaler(feature_range=(-1,1))
  scaled_values = scalar.fit_transform(detrended_values)
  scaled_values = scaled_values.reshape(len(scaled_values), 1)
  # transform into supervised learning problem X, y
  supervised = series_to_supervised(scaled_values, n_lag, n_seq)
  supervised_values = supervised.values
  # split into train and test sets 
  train_full, test = supervised_values[0:-n_test], supervised_values[-n_test:]
  train, val = train_full[0:-val_size], train_full[-val_size:] 
  return scalar, train, train_full, val, test, daily_mean

def prepare_arima_monthly(dataset, n_test, val_size):
  monthly_log = log_transform(dataset)
  monthly_detrended, monthly_mean = remove_trend_monthly(monthly_log)
  monthly_detrended = monthly_detrended[~pd.isnull(monthly_detrended)]

  train_full = monthly_detrended[:(len(monthly_detrended)-n_test+1)] 
  test = monthly_detrended[-(n_test):]
  train = train_full[:(len(train_full)-val_size+1)] 
  val = train_full[-(val_size): ]
  return train, train_full, val, test, monthly_mean

def prepare_arima_weekly(dataset, n_test, val_size):
  weekly_log = log_transform(dataset)
  weekly_detrended, weekly_mean = remove_trend_weekly(weekly_log)
  
  train_full = weekly_detrended[:(len(weekly_detrended)-n_test+1)] 
  test = weekly_detrended[-(n_test):]
  train = train_full[:(len(train_full)-val_size+1)] 
  val = train_full[-(val_size): ]
  return train, train_full, val, test, weekly_mean


def prepare_arima_daily(dataset, n_test, val_size):
  daily_log = log_transform(dataset)
  daily_detrended, daily_mean = remove_trend_daily(daily_log)
  
  train_full = daily_detrended[:(len(daily_detrended)-n_test+1)] 
  test = daily_detrended[-(n_test):]
  train = train_full[:(len(train_full)-val_size+1)] 
  val = train_full[-(val_size): ]
  return train, train_full, val, test, daily_mean

  
# configure
n_lag = 1
n_seq_monthly = 60 #changed this to do multistep
n_seq_weekly = 1
n_seq_daily = 1825 #1825

n_test_monthly = 60
n_test_weekly = 260
n_test_daily = 1825

val_size_monthly = 40
val_size_weekly = 173
val_size_daily = 1206

n_epochs = 500
n_batch = 1
n_neurons = 1

# prepare data
scalar_monthly = []
train_monthly = []
train_full_monthly = []
val_monthly = []
test_monthly = []
monthly_mean = []

scalar_weekly = []
train_weekly = []
train_full_weekly = []
val_weekly = []
test_weekly = []
weekly_mean = []

scalar_daily = []
train_daily = []
train_full_daily = []
val_daily = []
test_daily = []
daily_mean = []

train_monthly_arima = []
train_full_monthly_arima = []
val_monthly_arima = []
test_monthly_arima = []
monthly_mean_arima = []

train_weekly_arima = []
train_full_weekly_arima = []
val_weekly_arima = []
test_weekly_arima = []
weekly_mean_arima = []

train_daily_arima = []
train_full_daily_arima = []
val_daily_arima = []
test_daily_arima = []
daily_mean_arima = []

for i in range(len(monthly_averages)):
  scalar, train, train_full, val, test, mean = prepare_networks_monthly(monthly_averages[i], n_test_monthly, val_size_monthly, n_lag, n_seq_monthly)
  
  train_arima, train_full_arima, val_arima, test_arima, mean_arima = prepare_arima_monthly(monthly_averages[i], n_test_monthly, val_size_monthly)
  
  scalar_monthly.append(scalar)
  train_monthly.append(train)
  train_full_monthly.append(train_full)
  val_monthly.append(val)
  test_monthly.append(test)
  monthly_mean.append(mean)
  
  train_monthly_arima.append(train_arima)
  train_full_monthly_arima.append(train_full_arima)
  val_monthly_arima.append(val_arima)
  test_monthly_arima.append(test_arima)
  monthly_mean_arima.append(mean_arima)

for i in range(len(weekly_averages)):
  scalar, train, train_full, val, test, mean = prepare_networks_weekly(weekly_averages[i],
                                                      n_test_weekly, val_size_weekly, n_lag, n_seq_weekly)
                                                      
  train_arima, train_full_arima, val_arima, test_arima, mean_arima = prepare_arima_weekly(weekly_averages[i], n_test_weekly, val_size_weekly)
  
  scalar_weekly.append(scalar)
  train_weekly.append(train)
  train_full_weekly.append(train_full)
  val_weekly.append(val)
  test_weekly.append(test)
  weekly_mean.append(mean)
  
  train_weekly_arima.append(train_arima)
  train_full_weekly_arima.append(train_full_arima)
  val_weekly_arima.append(val_arima)
  test_weekly_arima.append(test_arima)
  weekly_mean_arima.append(mean_arima)
  
for i in range(len(daily_totals)):
  scalar, train, train_full, val, test, mean = prepare_networks_daily(daily_totals[i], 
                                                      n_test_daily, val_size_daily, n_lag, n_seq_daily)
                                                      
  train_arima, train_full_arima, val_arima, test_arima, mean_arima = prepare_arima_daily(daily_totals[i], n_test_daily, val_size_daily)
  
  scalar_daily.append(scalar)
  train_daily.append(train)
  train_full_daily.append(train_full)
  val_daily.append(val)
  test_daily.append(test)
  daily_mean.append(mean)
  
  train_daily_arima.append(train_arima)
  train_full_daily_arima.append(train_full_arima)
  val_daily_arima.append(val_arima)
  test_daily_arima.append(test_arima)
  daily_mean_arima.append(mean_arima)
