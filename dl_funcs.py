import pandas as pd
import numpy as np

import torch
from lightning import Trainer
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet as tsd
from pytorch_forecasting.data.encoders import GroupNormalizer as gn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.models import TemporalFusionTransformer as tft
from pytorch_forecasting.models import NBeats
from pytorch_forecasting.metrics.quantile import QuantileLoss

from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

from common_utils import *


def create_tsds(train_data: pd.DataFrame, 
                val_data: pd.DataFrame, 
                test_data: pd.DataFrame, 
                forecast_length: int, 
                looback_length: int,
                model_type: str):
    """
    Функция для создания TimeSeriesDatseat, 
    на которых обучаются модели pytorch_forecasting

    Параметры
        train_data - датафрейм для создания тренировочного tsd,
        val_data - датафрейм для создания tsd для валидации,
        test_data - датафрейм для создания tsd для теста,
        lookback_length - длина окна оглядки,
        forecast_length - длина прогноза,
        model_type - тип модели, должен быть либо 'tft' для создания tsd для TFT,
            либо 'nbeats' для создания tsd для N-BEATS
    Возвращает
        train: TimeSeriesDataSet - tsd для обучения модели,
        valid: TimeSeriesDataSet - tsd для валидации,
        test: TimeSeriesDataSet - tsd для тестирования модели
    """
    train = ''

    if model_type == 'tft':
        train = tsd(
        train_data,
        time_idx = 'time_idx',
        target = 'y',
        group_ids = ['series_id'],
        max_encoder_length = looback_length,
        max_prediction_length = forecast_length,
        static_categoricals = ['series_id'],
        time_varying_known_reals = ['time_idx', 'dayofweek', 'quarter', 'month', 'year', 
                                'dayofyear', 'week', 'month_start', 'month_end', 
                                'quarter_end', 'year_start', 'year_end', 'days_in_month', 'is_holiday'],
        time_varying_unknown_reals = ['y'],
        )

    elif model_type == 'nbeats':
        train = tsd(
        train_data,
        time_idx = 'time_idx',
        target = 'y',
        group_ids = ['series_id'],
        max_encoder_length = looback_length,
        max_prediction_length = forecast_length,
        time_varying_unknown_reals = ['y'],
        target_normalizer=gn(
            groups=["series_id"]
            )
        )

    valid = tsd.from_dataset(train, val_data, predict = True)
    test = tsd.from_dataset(train, test_data, predict = True)

    return train, valid, test


def slice_dfs(data: pd.DataFrame, 
              forecast_length: int, 
              lookback_length: int, 
              update_test_only: bool = False):
    """
    Функция для разделения данных на обучение, валидацию и тест
    Параметры:
        data: датафрейм с исходными данными
        forecast_length: длина прогноза 
        lookback_length: длина окна оглядки
        update_test_only: флаг того, что вернуть нужно только датафрейм для теста
    Возвращает:
        train_data: pd.DataFrame - данные для получения tsd для обучения,
        valid_data: pd.DataFrame - данные для получения валидационного tsd,
        test_data: pd.DataFrame - данные для получения тестового tsd

    Важное примечание: т.к. основной сценарий применения TFT и N-BEATS - это авторегрессионный
        метод прогнозирования, когда итоговый прогноз состоит из нескольких шагов, на каждом шаге
        выполняется прогноз только по небольшой части всего прогнозного периода, длина которой
        определяется в forecast_length, а полученные на предыдущем шаге значения используются для
        следующего шага, в переменной test_data данная функция возвращает данные для только для одного шага
        прогноза, а не для всего прогнозного периода
    """

    if update_test_only:
        test_data = data[lambda x: x.time_idx > x.time_idx.max() - forecast_length - lookback_length]
        return test_data
    else:
        train_data = data[lambda x: x.time_idx < x.time_idx.max() - forecast_length * 2]
        valid_data = data[lambda x: x.time_idx < x.time_idx.max() - forecast_length]
        test_data = data[lambda x: x.time_idx > x.time_idx.max() - forecast_length - lookback_length]
        return train_data, valid_data, test_data
    

    
def prediction_loop(model, 
                    known_data: pd.DataFrame, 
                    tsd_from: pytorch_forecasting.data.timeseries.TimeSeriesDataSet, 
                    first_test: torch.utils.data.dataloader.DataLoader, 
                    future_dates : np.array, 
                    forecast_length: int,
                    lookback_length: int,
                    forecast_periods: int,
                    model_type: 'str',
                    batch_size: int,
                    holidays: list,
                    forecast_start_date: str):
    """
    Функция для выполнения цикла прогнозов. Таким образом реализуется авторегрессионный метод:
        итоговый прогноз состоит из нескольких шагов, на каждом шаге выполняется прогноз только 
        по небольшой части всего прогнозного периода, длина которой определяется в forecast_length, 
        а полученные на предыдущем шаге значения используются для следующего шага
    Параметры:
        model: обученная модель из pytorch_forecasting,
        known_data: датафрейм, который был использован как аргумент data в 
            в функции slice_dfs, в которой данные разделялись на обучение, валидацию и первый шаг
            теста,
        tsd_from: tsd, с которого будут скопированы параметры; в идеале подавать
            в этот параметр тренировочный tsd, на котором обучалась модель
        first_test: даталоадер для первого шага прогноза,
        future_dates: список дат, для которых нуджно сделать прогноз; сюда не входят даты первого шага прогноза:
            допустим, требуется построить прогноз с 1 февраля по 31 марта при длине прогноза модели, равной 7: тогда 
            первый шаг будет включать в себя прогноз с 1 по 7 февраля, соответственно, первой датой в future_dates
            должно быть 8 февраля,
        forecast_length: длина шага прогноза (длина прогноза для модели),
        lookback_length: длина периода оглядки модели,
        forecast_periods: количество шагов прогноза - 1 (не считаем первый шаг) для покрытия 
        всего прогнозного периода; Например, если весь прогнозный период - это интервал с 1 февраля до 31 марта, 
            а длина прогноза модели 7 дней, то forecast_periods = 8: весь прогнозный период состоит из 59 точек,
            за вычетом первого шага остается 52 точки, чтобы покрыть из, с длиной прогноза модели, равной 7, нужно сделать
            минимум 8 шагов прогноза;
        model_type: тип модели pytorch_forecasting, должен быть либо 'tft', либо 'nbeats',
        batch_size: значение batch_size, даталоадеров,
        holidays: список выходных; требуется для заполнения признака is_holiday в функции create_features,
            если model_type = 'tft'

    Возвращает:
        pd.DataFrame c тремя колонками - датой, прогнозным значением целевой переменной,
        и индексом времени
    """

    df_known = known_data.copy()
    cur_time_idx = df_known.time_idx.max()
    #строим прогноз для первого шага
    first_forecast = model.predict(first_test).to("cpu").numpy().flatten()
    #последние n (n = forecast_lenght) значений целевой переменной 
    #заменяем на спрогнозированные на первом шагу значения
    for i in range(forecast_length):
        df_known.loc[df_known.time_idx == cur_time_idx-i, 'y'] = first_forecast[-i-1]

    for j in range(forecast_periods):
        cur_time_idx += 1
        #отбираем даты, которые будут спрогнозированы на текущем шаге
        if j != forecast_periods-1:
            cur_dates = future_dates[forecast_length*j:forecast_length*(j+1)]
        else:
            cur_dates = future_dates[forecast_length*j:]
        cur_forecast = pd.DataFrame(data = {'ds': cur_dates,
                                            'y': 0})
        
        #только tft использует ковариаты, nbeats нужны только прошлые 
        # значения ряда для построения прогноза
        if model_type == 'tft':
            cur_forecast = create_features(cur_forecast, holidays = holidays)
        cur_forecast['series_id'] = '0'
        cur_forecast['time_idx'] = np.arange(cur_time_idx, cur_time_idx + len(cur_forecast))
        cur_forecast = cur_forecast[list(df_known.columns)]

        # известные данные соединяем с датафреймом признаков текущего шага прогноза;
        # это нужно для того, чтобы сдвинуть окно прогноза вперед на длину forecast_length
        df_known = pd.concat([df_known, cur_forecast])
        cur_test = slice_dfs(df_known, lookback_length, forecast_length, update_test_only = True)
        cur_test.reset_index(inplace = True, drop = True)
        cur_test = tsd.from_dataset(tsd_from, cur_test, predict = True)
        cur_test_loader = cur_test.to_dataloader(train = False, batch_size = batch_size)

        #прогноз на текущий шаг (длина прогноза = forecast_length)
        cur_preds = model.predict(cur_test_loader).to("cpu").numpy().flatten()
        cur_time_idx += forecast_length - 1
        #снова заменяем последние n значений целевой переменной на спрогнозированные значения,
        #чтобы использовать их в внутри окна оглядки на следующем шаге
        for k in range(forecast_length):
            df_known.loc[df_known.time_idx == cur_time_idx-k, 'y'] = cur_preds[-k-1]

    df_known = df_known[['ds', 'y', 'time_idx']]
    df_known = df_known[df_known.ds >= forecast_start_date]
    df_known.reset_index(inplace = True, drop = True)

    return df_known


def split_train_forecast(df: pd.DataFrame, id: int):
    """
    Функция для разделения данных на прогнозный период и обучение + валидацию + первый шаг прогноза
    Args:
        df (pd.DataFrame): датафрейм с временным рядом
        id (int): id ряда в исследовании - 1 или 2

    Returns:
        train: датафрейм, содержащий данные для обучения DL-модели, ее валидации и первого шага прогноза
        forecast: датафрейм с прогнозным периодом
    """
    if id == 1:
        train = df[df.ds < '2023-03-29']
        forecast = df[df.ds >= '2023-03-22']
        return train, forecast
    elif id == 2:
        train = df[df.ds < '2023-02-08']
        forecast = df[df.ds >= '2023-02-01']
        return train, forecast
    


def fit_dl_model(train_tsd, 
                 train_loader, 
                 val_loader,
                 model_type,
                 model_params,
                 forecast_length = 7):
    """
    Обучение DL-модели из pytorch_forecasting
    Параметры:
        train_tsd (TimeSeriesDataset): tsd, из которого создается модель с помощью метода "from_dataset",
        train_loader (Dataloader): лоадер тренировочных данных,
        val_loader (Dataloader): лоадер валидационных данных,
        model_type (str): тип модели - tft или nbeats,
        model_params (dict): словарь параметров модели,
        forecast_length - длина одного шага прогноза
    """

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, 
                                        patience=5, verbose=True, mode="min")
    
    trainer = Trainer(
    max_epochs=50,
    accelerator = 'gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback]
    )

    if model_type == 'tft':
        model = tft.from_dataset(
        train_tsd,
        learning_rate = model_params['learning_rate'],
        hidden_size = model_params['hidden_size'],
        attention_head_size = model_params['attention_head_size'],
        dropout = model_params['dropout'],
        hidden_continuous_size = model_params['hidden_continuous_size'],
        output_size = 7, 
        loss = QuantileLoss(),
        log_interval = 10, 
        reduce_on_plateau_patience = 2)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
        
        return model
    

    elif model_type == 'nbeats':
        model = NBeats.from_dataset(
        train_tsd, 
        learning_rate = model_params['learning_rate'], 
        stack_types = ['generic'],
        num_blocks = model_params['num_blocks'],
        num_block_layers = [4],
        widths = [512],
        sharing = False,
        prediction_length = forecast_length,
        expansion_coefficient_lengths = [32],
        context_length = forecast_length * model_params['lookback_coef'],
        weight_decay=1e-2, 
        backcast_loss_ratio=0.1,
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
        
        return model

