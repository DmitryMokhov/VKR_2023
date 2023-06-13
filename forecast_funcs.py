import pandas as pd
import numpy as np
import time
from prophet import Prophet
from catboost import CatBoostRegressor as cbr
from catboost import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from itertools import product
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import plotly
import plotly.express as px
import plotly.graph_objects as go

from data_prep import *


def prophet_forecast(train: pd.DataFrame, 
                     test: pd.DataFrame, 
                     best_params: dict):
    """
    Функция для построения прогноза с Prophet
    Args:
        train: данные для обучения модели,
        test: тестовые данные,
        best_params: словарь с гиперпараметрами модели,
        error_func: функционал ошибки, который нужно рассчитать для построенного прогноза

    Returns:
        errors: float - значение ошибки прогноза,
        forecast - прогноз Prophet для тестового периода
    """

    prop = Prophet(**best_params)
    prop.add_seasonality(name = 'monthly', period = 30.4, 
                            fourier_order = 5, prior_scale = 10, mode = 'multiplicative')
    prop.fit(train)
    forecast = prop.predict(test).yhat.values
    cur_mape = mape(test.y, forecast)
    cur_rmse = mse(test.y, forecast, squared=False)
    cur_mae = mae(test.y, forecast)

    metrics = pd.DataFrame(data = {'mape': cur_mape,
                                  'rmse': cur_rmse,
                                  'mae': cur_mae})

    return metrics, forecast


def catboost_loop(train: сatboost.Pool, 
                  valid: catboost.Pool, 
                  test: catboost.Pool, 
                  best_params: dict, 
                  error_func):
    """
    Функция для построения прогноза с помощью Catboost
    параметры:
        train: данные для обучения модели,
        valid: данные для валидации в ходе обучения,
        test: тестовые данные,
        best_params: словарь с гиперпараметрами модели
        error_func: функция ошибки

    Возвращает:
        error: float - ошибка прогноза по выбранному функционалу
    """

    cat = cbr(**best_params)
    cat.fit(train, 
            eval_set = valid,
            early_stopping_rounds = 25,
            verbose = False)
    forecast = cat.predict(test)
    error = error_func(test.get_label(), forecast)

    return error, forecast


def get_lstm_net(lookback):
    """
    Функция, возвращающая объект LSTM-нейросети
    """
    lstm_model = Sequential()
    lstm_model.add(LSTM(256, activation='relu', input_shape=(1, lookback), return_sequences = True))
    lstm_model.add(LSTM(128, activation='relu'))
    lstm_model.add(Dropout(0.25))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.25))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mae')
    return lstm_model



def prediction_loop(model, start, n_rounds):
    """
    Функция для цикличного прогнозирования с помощью LSTM:
    строится прогноз на одну точку вперед, затем происходит обновление окна оглядки - 
    оно сдвигается на одну точку вперед, захватывая только что выполненный прогноз
    Args:
        model: объект модели LSTM 
        start: np.attay - стартовое окно оглядки, по которому будет спрогнозирована первая точка тестового периода
        n_rounds: int - количество шагов прогноза

    Returns:
        preds: list - прогноз LSTM-модели на следующие n_rounds шагов
    """
    preds = []
    for i in range(n_rounds):
        cur_pred = model.predict(start)
        preds.append(cur_pred[0])
        start = start[:, :, 1:]
        start = np.append(start, cur_pred)
        start = np.reshape(start, (1, 1, len(start)))
    return preds



def lstm_loop(train: pd.DataFrame, 
              valid: pd.DataFrame, 
              test: pd.DataFrame, 
              lookback: int):
    """
    Петля прогнозов LSTM
    Args:
        train: данные для обучения,
        valid: данные для валидации,
        test: данные для проверки точности прогноза,
        lookback: длина окна оглядки

    Returns:
        error: float - ошибка прогноза,
        preds: np.array - прогноз модели на тестовый период
    """
    trainX, trainY, validX, validY, cur_scaler = prep_data_lstm(train, valid, lookback)

    lstm_model = get_lstm_net()
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5)

    lstm_model.fit(trainX, trainY, 
                    validation_data = (validX, validY),
                    epochs = 25, batch_size = 16, callbacks = [early_stop])
    start = validX[-1][:, 1:]
    start = np.append(start, validY[-1])
    start = np.reshape(start, (1, 1, len(start)))
    preds = prediction_loop(lstm_model, start, len(test))
    preds = np.array(preds)
    preds = cur_scaler.inverse_transform(preds).flatten()

    error = mae(test.y, preds)

    return error, preds