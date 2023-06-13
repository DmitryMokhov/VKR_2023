import pandas as pd
import numpy as np
import json
import torch
from lightning import Trainer
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet as tsd
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.models import TemporalFusionTransformer as tft
from pytorch_forecasting.metrics.quantile import QuantileLoss

from common_utils import *
from dl_funcs import *
from data_prep import holidays
FORECAST_LENGTH = 7
pl.seed_everything(42)

def run_tft_eval(id: int, plot_forecast = True):
    """
    Функция для оценки точности TFT на рядах, представленных в ВКР
    Args:
        series (int): номер ряда - 1 или 2
    """

    df = load_data(f'data/series{id}.csv')
    config = json.load(open(f'configs/tft_series{id}.json'))

    batch_size = config['batch_size']
    lookback_coef = config['lookback_coef']

    ds, future_dates = retrive_ds(id)
    df['ds'] = ds
    df = create_features(df, holidays)
    df['time_idx'] = np.arange(len(df))
    df['series_id'] = '0'

    df, future = split_train_forecast(df, id)

    train, valid, test = slice_dfs(df, FORECAST_LENGTH, FORECAST_LENGTH * lookback_coef)
    train_tsd, valid_tsd, test_tsd = create_tsds(train, valid, test, FORECAST_LENGTH, 
                                                 lookback_coef, model_type='tft')
    train_loader = train_tsd.to_dataloader(train = True, batch_size = batch_size)
    valid_loader = valid_tsd.to_dataloader(train = False, batch_size = batch_size)
    test_loader = test_tsd.to_dataloader(train = False, batch_size = batch_size)


    model = fit_dl_model(train_tsd, train_loader, valid_loader,
                         'tft', config)

    forecast = prediction_loop(model, 
                               df, 
                               train_tsd, 
                               test_loader,
                               future_dates, 
                               FORECAST_LENGTH, 
                               FORECAST_LENGTH * lookback_coef,
                               8,
                               'tft',
                               batch_size,
                               holidays=holidays,
                               forecast_start_date = future.ds.min())
    
    mape = mape(future.y, forecast.y)
    rmse = mse(future.y, forecast.y, squared=True)
    mae = mae(future.y, forecast.y)
    metrics = {'mape': mape, 'rmse': rmse, 'mae': mae}
    metrics = pd.DataFrame(metrics)

    if plot_forecast:
        plot_forecasts(future, [forecast.y], ['TFT'])

    return metrics, forecast

tft_metrics, tft_forecast = run_tft_eval(1)
print(tft_metrics)

