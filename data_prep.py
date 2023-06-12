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
from sklearn.metrics import mean_absolute_percentage_error as mape

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import plotly
import plotly.express as px
import plotly.graph_objects as go



def create_lstm_dataset(data: pd.DataFrame, lookback: int):
    """
    Функция для создания датасета для обучения LSTM-сети
    Функция возвращает 2 массива: массив массивов длиной lookback, 
    которые являются лагами прогнозируемого значения и непосредственно массив
    прогнозируемых значений
    Args:
        data: данные, из которых нужно создать датасет для LSTM
        lookback: длина окна оглядки, по которому LSTM будет делать прогноз

    Returns:
        dataX: np.array - лаги прогнозируемых значений,
        dataY: массив значений целевой переменной;
    """
    dataX, dataY = [], []
    for i in range(len(data)-lookback):
        a = data[i:(i+lookback), 0]
        dataX.append(a)
        dataY.append(data[i + lookback, 0])
    return np.array(dataX), np.array(dataY)


def prep_data_lstm(train: pd.DataFrame, valid: pd.DataFrame, lookback: int):
    """
    Подготовка данныз для запуска петли прогнозов LSTM
    Args:
        train: данные для обучения
        valid: данные для валидации
        lookback: длина окна оглядки
    """

    cur_train = train['y']
    cur_valid = valid['y']

    cur_scaler = MinMaxScaler()
    cur_scaler.fit(cur_train.values.reshape(-1,1))

    scaled_train = cur_scaler.transform(cur_train.values.reshape(-1, 1))
    scaled_valid = cur_scaler.transform(cur_valid.values.reshape(-1, 1))

    trainX, trainY = create_lstm_dataset(scaled_train, lookback)
    validX, validY = create_lstm_dataset(scaled_valid, lookback)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

    return trainX, trainY, validX, validY, cur_scaler


def gmdh_split(data: list, n_lags: int):
    """
    Создание датасета для обучения gmdh-модели
    Для каждой точки прогноз строится по предыдущим точкам, количество которых равно n_lags
    следовательно, для прогнозирования нужен массив массивов длиной n_lags
    Параметры:
        data: список значений временного ряда, который нужно прогнозировать,
        n_lags (_type_): количество предыдущих точек, по которому 
            будет построен прогноз для текущей точки

    Возвращает:
        gmdh_lags: list - признаки для выполнения прогноза,
        gmdh_y: list - список значений целевой переменной
    """
    gmdh_lags = []
    gmdh_y = []

    for i in range(n_lags, len(data)):
        gmdh_lags.append(data[i - n_lags:i])
    gmdh_y = data[n_lags:]
    return gmdh_lags, gmdh_y


def get_cat_pools(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    """
    Функция для формирования объектов типа catboost.Pool, требуемых для обучения Catboost
    Параметры:
        train: данные для обучения модели,
        valid: данные для валидации в ходе обучения,
        test: данные выполнения прогноза
    Возвращает:

    """
    train_cat = train.copy()
    val_cat = valid.copy()
    test_cat = test.copy()

    y_train = train_cat.y
    y_val = val_cat.y
    y_test = test_cat.y
    train_cat.drop(columns = ['ds', 'y'], inplace = True)
    val_cat.drop(columns = ['ds', 'y'], inplace = True)
    test_cat.drop(columns = ['ds', 'y'], inplace = True)
    train_pool = Pool(train_cat, label = y_train)
    val_pool = Pool(val_cat, label = y_val)
    test_pool = Pool(test_cat, label = y_test)

    return train_pool, val_pool, test_pool
