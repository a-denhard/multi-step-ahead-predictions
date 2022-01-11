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
  
monthly_averages = data.monthly_averages

train = data.train_monthly
train_full = data.train_full_monthly
test = data.test_monthly
val = data.val_monthly
scalar = data.scalar_monthly
monthly_mean = data.monthly_mean

n_lag = data.n_lag
n_seq = 1

n_test = data.n_test_monthly
val_size = data.val_size_monthly

n_epochs = data.n_epochs
n_batch = data.n_batch
n_neurons = data.n_neurons

n_step = 1

for i in range(len(monthly_averages)):  
  # GRU
  tic()
  model_gru = fit_gru(train_full[i], n_lag, n_batch, n_epochs, n_step)
  seconds = toc()
  m, s = divmod(seconds, 60)
  print('Runtime (RNN) is ', m, ' minutes and ', s, ' seconds\n')
  
  
  forecasts_gru = make_forecasts_network(model_gru, n_batch, test[i], n_lag)
  forecasts_gru = np.array(forecasts_gru).flatten()
  
  predicted_GRU = inverse_transforms(forecasts_gru, scalar[i], monthly_mean[i])
  q = predicted_GRU
  temp = monthly_averages[i]
  actual = temp[-n_test:]
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
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-NREL-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-NREL-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-NREL-monthly.txt', metrics,  delimiter = ' ')
  elif(i==1):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-ARM-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-ARM-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-ARM-monthly.txt', metrics,  delimiter = ' ')
  elif(i==2):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-BON-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-BON-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-BON-monthly.txt', metrics,  delimiter = ' ')
  elif(i==3):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-DRA-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-DRA-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-DRA-monthly.txt', metrics,  delimiter = ' ')
  elif(i==4):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-FPK-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-FPK-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-FPK-monthly.txt', metrics,  delimiter = ' ')
  elif(i==5):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-GWN-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-GWN-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-GWN-monthly.txt', metrics,  delimiter = ' ')
  elif(i==6):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-PSU-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-PSU-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-PSU-monthly.txt', metrics,  delimiter = ' ')
  elif(i==7):
    model_gru.save('/wendianHome/bins/adenhard/one-step/models/GRU-SXF-monthly.h5')
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/GRU-SXF-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-SXF-monthly.txt', metrics,  delimiter = ' ')
  elif(i==8):
    model_gru.save('/u/st/am/adenhard/bins/fittedGRU-TBL-monthly.h5')
    np.savetxt('/u/st/am/adenhard/bins/forecastsGRU-TBL-monthly.txt', predicted_GRU, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/GRU-TBL-monthly.txt', metrics,  delimiter = ' ')