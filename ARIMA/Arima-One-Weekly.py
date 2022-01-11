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

weekly_averages = data.weekly_averages

train_full_weekly  = data.train_full_weekly_arima
test_weekly = data.test_weekly_arima

pred_weekly = []

for i in range(0, len(weekly_averages)):
  tic()
  new_train = train_full_weekly[i]
  pred_weekly = []
  arima_model = auto_arima(new_train, start_p=0, start_q=0, max_p=5, max_d=12, max_q=5, start_P=0,
    start_Q=0, max_P=5, max_D=12, max_Q=5, test='adf', information_criterion='aic')
  for j in range(1,261):
    
    arima_model.fit(new_train)
    pred_weekly.append(arima_model.predict(n_periods = 1))
    new_train = new_train.append(test_weekly[j-1:j])
  
  pred_weekly = np.concatenate(pred_weekly, axis=0)
  print(pred_weekly)
    

  seconds = toc()             
  m, s = divmod(seconds, 60)
  print('Runtime is ', m, ' minutes and ', s, ' seconds\n')
  
  print('Location', i, arima_model.summary()) 
  
  pred = data.inverse_trend_weekly(pred_weekly, data.weekly_mean[i])
  predicted_ARIMA = data.inverse_log(pred)
  temp = weekly_averages[i]
  actual = temp[-data.n_test_weekly:]
  
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
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-NREL-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-NREL-weekly.txt', metrics,  delimiter = ' ')
  elif(i==1):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-ARM-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-ARM-weekly.txt', metrics, delimiter = ' ')
  elif(i==2):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-BON-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-BON-weekly.txt', metrics, delimiter = ' ')
  elif(i==3):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-DRA-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-DRA-weekly.txt', metrics, delimiter = ' ')
  elif(i==4):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-FPK-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-FPK-weekly.txt', metrics, delimiter = ' ')
  elif(i==5):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-GWN-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-GWN-weekly.txt', metrics, delimiter = ' ')
  elif(i==6):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-PSU-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-PSU-weekly.txt', metrics, delimiter = ' ')
  elif(i==7):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-SXF-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-SXF-weekly.txt', metrics, delimiter = ' ')
  elif(i==8):
    np.savetxt('/wendianHome/bins/adenhard/one-step/forecasts/ARIMA-TBL-weekly.txt', predicted_ARIMA, delimiter = ' ')
    np.savetxt('/wendianHome/bins/adenhard/one-step/metrics/ARIMA-TBL-weekly.txt', metrics, delimiter = ' ')
