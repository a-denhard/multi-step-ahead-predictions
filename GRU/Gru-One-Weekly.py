# importing required libraries

import data

import statsmodels as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.api import SARIMAX

import pmdarima as pm
from pmdarima.arima import auto_arima

import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

import numpy as np
from numpy import array
from numpy import asarray
from numpy import savetxt

import matplotlib.pyplot as plt

import tensorflow as tf

import os

import statistics as st

import math 
from math import sqrt

from timezonefinder import TimezoneFinder
from datetime import datetime
from dateutil import tz

from keras.models import Sequential

from keras.layers import concatenate
from keras.models import Model

from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.layers import SimpleRNN
from keras.layers import TimeDistributed, RNN, GRU

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from pprint import pprint

from keras.wrappers.scikit_learn import KerasRegressor

from ttictoc import tic,toc

import calendar
from calendar import isleap

# make one forecast with an LSTM,
def forecast_network(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts_network(model, n_batch, test, n_lag):
  forecasts = list()
  for i in range(len(test)):
    X, y = test[i, 0:n_lag], test[i, n_lag:]
    # make forecast
    forecast = forecast_network(model, X, n_batch)
    # store the forecast
    forecasts.append(forecast)
  return forecasts
  
# fit an GRU network to training data
def fit_gru(train, n_lag, n_batch, nb_epoch, n_step):
	# reshape training into [samples, timesteps, features]
	#X, y = trainX, trainy
  X, y = train[:, 0:n_lag], train[:, n_lag:]
  X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
  model = Sequential()
  # two-layer stacked of GRU
  model.add(GRU(units = 500, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), return_sequences=True, 
                 activation = 'tanh', dropout = 0.20))
  model.add(GRU(units = 500, return_sequences=True, 
                 activation = 'tanh', dropout = 0.20))
  # wrapper function using TimeDistributed 
  model.add(TimeDistributed(Dense(units = n_step, activation = 'sigmoid')))
  model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
  for i in range(nb_epoch):   
    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()
  return model
  
# undo log transform
def inverse_log(dataset):
  vals = dataset.values.astype(float)
  ind = dataset.index
  return Series(np.exp(vals), index=ind)
  
# add trend back in 
def inverse_trend_monthly(dataset, monthly_mean):
  nrow = int(len(dataset)/12)
  predicted = array(dataset).reshape(nrow,12)
  predicted = predicted + monthly_mean
  predicted = predicted.flatten()
  return Series(predicted)

# add trend back in 
def inverse_trend_weekly(dataset, weekly_mean):
  nrow = int(len(dataset)/52)
  predicted = array(dataset).reshape(nrow,52)
  predicted = predicted + weekly_mean
  predicted = predicted.flatten()
  return Series(predicted)
  
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
  inv_trend = inverse_trend_monthly(inverted, monthly_mean)
  # invert log
  inv_log = inverse_log(inv_trend)  
  return inv_log
  
def inverse_transforms_weekly(forecasts, scalar, weekly_mean):
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
  inv_trend = inverse_trend_weekly(inverted, weekly_mean)
  # invert log
  inv_log = inverse_log(inv_trend)  
  return inv_log
  
def inverse_transforms_daily(forecasts, scalar, daily_mean):
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
  inv_trend = inverse_trend_daily(inverted, daily_mean)
  # invert log
  inv_log = inverse_log(inv_trend)  
  return inv_log
    
weekly_averages = data.weekly_averages

train = data.train_weekly
train_full = data.train_full_weekly
test = data.test_weekly
val = data.val_weekly
scalar = data.scalar_weekly
weekly_mean = data.weekly_mean

n_lag = data.n_lag
n_seq = 1

n_test = data.n_test_weekly
val_size = data.val_size_weekly

n_epochs = data.n_epochs
n_batch = data.n_batch
n_neurons = data.n_neurons

n_step = 1

for i in range(3, len(weekly_averages)):
  
  # GRU
  tic()
  model_gru = fit_gru(train_full[i], n_lag, n_batch, n_epochs, n_step)
  seconds = toc()
  m, s = divmod(seconds, 60)
  print('Runtime (GRU) is', m, ' minutes and ', s, ' seconds\n')
  
  forecasts_gru = make_forecasts_network(model_gru, n_batch, test[i], n_lag)
  forecasts_gru = np.array(forecasts_gru).flatten()
  
  predicted_GRU = inverse_transforms_weekly(forecasts_gru, scalar[i], weekly_mean[i])
  temp = weekly_averages[i]
  actual = temp[-n_test:]
  q = predicted_GRU
  x = actual

  MBE = st.mean(q.values - x.values)
  RMSE = sqrt(mean_squared_error(x, q)) # Mean squared error regression loss.
  MAE = mean_absolute_error(x, q) # Mean absolute error regression loss.
  MSE = mean_squared_error(x, q) # Mean squared error regression loss.
  maxE = max_error(x,q) # max_error metric calculates the maximum residual error.
  MSlogE = mean_squared_log_error(x,q) # Mean squared logarithmic error regression loss.
  MedAE = median_absolute_error(x,q) # Median absolute error regression loss.
  r_squared  = r2_score(x,q) # R_squared (coefficient of determination) regression score function.
  
  MBE_p = (MBE/st.mean(x.values))*100
  RMSE_p = sqrt((mean_squared_error(x, q))/(st.mean(x.values)**2))*100
  MAE_p = (MAE/st.mean(x.values))*100

  MBE = float('{:f}'.format(MBE))
  RMSE = float('{:f}'.format(RMSE))
  MAE = float('{:f}'.format(MAE))
  MBE_p = float('{:f}'.format(MBE_p))
  RMSE_p = float('{:f}'.format(RMSE_p))
  MAE_p = float('{:f}'.format(MAE_p))
  r_squared = float('{:f}'.format(r_squared))
  MSE = float('{:f}'.format(MSE))
  maxE = float('{:f}'.format(maxE))
  MSlogE = float('{:f}'.format(MSlogE))
  MedAE = float('{:f}'.format(MedAE))
  
  #metrics = np.array(['MBE', 'RMSE', 'MAE', 'MBE%', 'RMSE%', 'MAE%', 'R-squared', 'MSE', 'Max Error', 'MS log error', 'Median AE'])
  metrics = np.array([MBE, RMSE, MAE, MBE_p, RMSE_p, MAE_p, r_squared, MSE, maxE, MSlogE, MedAE])
  
  if (i==0):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-NREL-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-NREL-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-NREL-weekly.txt', metrics,  delimiter = ' ')
  elif(i==1):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-ARM-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-ARM-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-ARM-weekly.txt', metrics,  delimiter = ' ')
  elif(i==2):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-BON-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-BON-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-BON-weekly.txt', metrics,  delimiter = ' ')
  elif(i==3):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-DRA-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-DRA-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-DRA-weekly.txt', metrics,  delimiter = ' ')
  elif(i==4):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-FPK-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-FPK-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-FPK-weekly.txt', metrics,  delimiter = ' ')
  elif(i==5):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-GWN-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-GWN-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-GWN-weekly.txt', metrics,  delimiter = ' ')
  elif(i==6):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-PSU-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-PSU-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-PSU-weekly.txt', metrics,  delimiter = ' ')
  elif(i==7):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-SXF-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-SXF-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-SXF-weekly.txt', metrics,  delimiter = ' ')
  elif(i==8):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-TBL-weekly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-TBL-weekly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-TBL-weekly.txt', metrics,  delimiter = ' ')