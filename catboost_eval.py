import pandas as pd
import numpy as np

from tuning_funcs import tune_catboost
from forecast_funcs import catboost_loop
from common_utils import *
from data_prep import *

from sklearn.metrics import mean_absolute_error as mae

def run_catboost_eval(id: int, plot_forecast = True):
    """
    Функция для оценки точности прогноза Catboost
    Параметры:
        id (int): номер ряда, 1 или 2
        plot_forecast (bool, optional): Показывать ли график прогноза
    """
    df = load_data(f'data/series{id}.csv')
    df = create_features(df, holidays=holidays)
    train, val, test = split_data_with_val(id, df)
    train_pool, val_pool, test_pool = get_cat_pools(train, val, test)

    cat_params = tune_catboost(train_pool, val_pool, mae)
    cat_metrics, cat_forecast = catboost_loop(train_pool, 
                                              val_pool, 
                                              test_pool,
                                              cat_params)
    
    if plot_forecast:
        plot_forecasts(test, [cat_forecast], ['Catboost'])

    return cat_metrics, cat_forecast


cat_metrics, cat_forecast = run_catboost_eval(1)
print(cat_metrics)