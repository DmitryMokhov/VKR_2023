import pandas as pd
import numpy as np
import json

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

from common_utils import *
from dl_funcs import *
from data_prep import holidays
from forecast_funcs import calculate_metrics
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

    future_dates = get_future_dates(id)
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
    
    #cur_mape = mape(future.y, forecast.y)
    #cur_rmse = mse(future.y, forecast.y, squared=False)
    #cur_mae = mae(future.y, forecast.y)
    #metrics = {'mape': [cur_mape], 'rmse': [cur_rmse], 'mae': [cur_mae]}
    #metrics = pd.DataFrame(metrics)
    forecast = forecast.y.values
    metrics = calculate_metrics(future, forecast, id = id)

    if plot_forecast:
        plot_forecasts(future, [forecast], ['TFT'])

    return metrics, forecast

tft_metrics, tft_forecast = run_tft_eval(1)
print(tft_metrics)

