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

daily_totals = data.daily_totals

train_full_daily  = data.train_full_daily_arima
test_daily = data.test_daily_arima

pred_daily = []

  
for i in range(len(daily_totals)):
  tic()
  arima_model = auto_arima(train_full_daily[i], start_p=0, start_q=0, max_p=5, max_d=12, max_q=5, 
                         start_P=0, start_Q=0, max_P=5, max_D=12, max_Q=5, test='adf',information_criterion='aic')
  seconds = toc()
  m, s = divmod(seconds, 60)
  print('Runtime is ', m, ' minutes and ', s, ' seconds\n')
  
  print('Location', i, arima_model.summary()) 
  
  forecast, interval = arima_model.predict(n_periods = data.n_test_daily, return_conf_int = True)
  pred_daily.append(forecast)
  
  pred = data.inverse_trend_daily(pred_daily[i], data.daily_mean[i])
  predicted_ARIMA = data.inverse_log(pred)
  temp = daily_totals[i]
  actual = temp[-data.n_test_daily:]
  
  x = actual[~pd.isnull(actual)]
  y = predicted_ARIMA[~pd.isnull(predicted_ARIMA)]

  MBE = st.mean(y.values - x.values)
  RMSE = sqrt(mean_squared_error(x, y)) # Mean squared error regression loss.
  MAE = mean_absolute_error(x, y) # Mean absolute error regression loss.
  MSE = mean_squared_error(x, y) # Mean squared error regression loss.
  maxE = max_error(x,y) # max_error metric calculates the maximum residual error.
  MSlogE = mean_squared_log_error(x,y) # Mean squared logarithmic error regression loss.
  MedAE = median_absolute_error(x,y) # Median absolute error regression loss.
  r_squared = r2_score(x,y) # R_squared (coefficient of determination) regression score function.
  
  MBE_p = (MBE/st.mean(x))*100
  RMSE_p = sqrt((mean_squared_error(x, y))/(st.mean(x)**2))*100
  MAE_p = (MAE/st.mean(x))*100

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
  
  if(i==0):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-NREL-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-NREL-daily.txt', metrics,  delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-NREL-daily.txt', interval, delimiter = ' ')
  elif(i==1):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-ARM-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-ARM-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-ARM-daily.txt', interval, delimiter = ' ')
  elif(i==2):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-BON-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-BON-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-BON-daily.txt', interval, delimiter = ' ')
  elif(i==3):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-DRA-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-DRA-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-DRA-daily.txt', interval, delimiter = ' ')
  elif(i==4):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-FPK-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-FPK-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-FPK-daily.txt', interval, delimiter = ' ')
  elif(i==5):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-GWN-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-GWN-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-GWN-daily.txt', interval, delimiter = ' ')
  elif(i==6):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-PSU-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-PSU-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-PSU-daily.txt', interval, delimiter = ' ')
  elif(i==7):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-SXF-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-SXF-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-SXF-daily.txt', interval, delimiter = ' ')
  elif(i==8):
    np.savetxt('/wendianHome/bins/adenhard/multi-step/forecasts/ARIMA-TBL-daily.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/metrics/ARIMA-TBL-daily.txt', metrics, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/multi-step/intervals/ARIMA-TBL-daily.txt', interval, delimiter = ' ')