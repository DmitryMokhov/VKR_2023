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



holidays = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
            '2019-01-05', '2019-01-06', '2019-01-07', '2019-01-08',
            '2019-02-23', '2019-03-08', '2019-05-01', '2019-05-02',
            '2019-05-03', '2019-05-09', '2019-05-10', '2019-06-12',
            '2019-11-04',

            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
            '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
            '2020-02-23', '2020-02-24', '2020-03-08', '2020-03-09', 
            '2020-05-01', '2020-05-04', '2020-05-05', 
            '2020-05-09', '2020-05-11', 
            '2020-06-12', '2020-11-04',
            
            '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',
            '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',
            '2021-02-22', '2021-02-23', '2021-03-08',  
            '2021-05-01', '2021-05-03', 
            '2021-05-09', '2021-05-10', 
            '2021-06-12', '2021-06-14', '2021-11-04', '2021-11-05',

            '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
            '2022-01-05', '2022-01-06', '2022-01-07', '2022-02-23', 
            '2022-03-07', '2022-03-08'
            '2022-05-01', '2022-05-02', '2022-05-03', 
            '2022-05-09', '2022-05-10', 
            '2022-06-12', '2022-06-13', '2022-11-04', 

            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
            '2023-02-23', '2023-02-24', '2023-03-08',
            '2023-05-01', 
            '2023-05-08', '2023-05-09', 
            '2023-06-12', '2023-11-04', '2023-11-06' 
            ]



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


def split_data_without_val(id: int,
                           df: pd.DataFrame):
    """
    Функция для разделения данный на обучение и тест для методов, которым не нужен дополнительный
    валидационный набор
    Параметры:
        id: номер ряда из исследования - 1 или 2,
        df - датафрейм, содержащий значения временного ряда и колонку дат ds
    Возвращает:
        train: pd.DataFrame с данными для обучения,
        test: pd.DataFrame c данными для проверки точности прогноза
    """
    if id == 1:
        train = df[df.ds < '2023-03-22']
        test = df[df.ds >= '2023-03-22']
        return train, test
    elif id == 2:
        train = df[df.ds < '2023-02-01']
        test = df[df.ds >= '2023-02-01']
        return train, test