import pandas as pd
import numpy as np
from typing import Union
from prophet import Prophet
from catboost import CatBoostRegressor as cbr
from catboost import Pool
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

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
    #metrics = calculate_metrics(test, forecast)

    return forecast


def catboost_loop(train: Pool, 
                  valid: Pool, 
                  test: Pool, 
                  best_params: dict):
    """
    Функция для построения прогноза с помощью Catboost
    параметры:
        train: данные для обучения модели,
        valid: данные для валидации в ходе обучения,
        test: тестовые данные,
        best_params: словарь с гиперпараметрами модели

    Возвращает:
        error: float - ошибка прогноза по выбранному функционалу
    """

    cat = cbr(**best_params)
    cat.fit(train, 
            eval_set = valid,
            early_stopping_rounds = 25,
            verbose = False)
    forecast = cat.predict(test)
    test_y = test.get_label()
    test_y = pd.DataFrame(test_y)
    test_y.columns = ['y']
    #metrics = calculate_metrics(test_y, forecast)

    return forecast


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

    lstm_model = get_lstm_net(lookback=7)
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

    #metrics = calculate_metrics(test, preds)

    return preds


def calculate_metrics(fact: pd.DataFrame, forecast: Union[np.array, list], id: int):
    """
    Функция для вычисления метрик точности прогноза
    Параметры:
        fact (pd.DataFrame): датафрейм с фактическими данными, обязательно должен содержать колонку 'y'
            с настоящими значениями временного ряда за прогнозный период, 
        forecast (Union[np.array, list]): прогноз модели.
        id: номер ряда в исследовании - 1 или 2; от этого зависит, какое скалирование применять для 
            расчета метрик
    """
    temp_fact = fact[['ds', 'y']]
    temp_fact.reset_index(drop = True, inplace = True)
    temp_fact['y_forc'] = forecast
    temp_fact = temp_fact[~temp_fact.ds.isin(['2023-02-23', '2023-02-24', '2023-03-08',
                                                '2023-05-01', '2023-05-08', '2023-05-09'])]
    
    if id == 1:
        temp_fact['y'] = temp_fact['y'] * 644.66 + 1062.1
        temp_fact['y_forc'] = temp_fact['y_forc'] * 644.66 + 1062.1
    elif id == 2:
        temp_fact['y'] = temp_fact['y'] * 169.26 + 253.07
        temp_fact['y_forc'] = temp_fact['y_forc'] * 169.26 + 253.07

    #cur_mape = mape(fact.y, forecast)
    #cur_rmse = mse(fact.y, forecast, squared=False)
    #cur_mae = mae(fact.y, forecast)
    cur_mape = mape(temp_fact.y, temp_fact.y_forc)
    cur_rmse = mse(temp_fact.y, temp_fact.y_forc, squared=False)
    cur_mae = mae(temp_fact.y, temp_fact.y_forc)

    metrics = pd.DataFrame(data = {'mape': [cur_mape],
                                  'rmse': [cur_rmse],
                                  'mae': [cur_mae]})
    return metrics