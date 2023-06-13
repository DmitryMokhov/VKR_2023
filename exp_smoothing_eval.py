import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

import plotly
import plotly.express as px
import plotly.graph_objects as go

from common_utils import*
from data_prep import *


def run_exp_smoothing_eval(id: int, plot_forecast = True):
    """
    Функция для оценки точности экспоненциального сглаживания на рядах, представленных в ВКР
    Args:
        id (int): номер ряда, 1 или 2
        plot_forecast (bool, optional): Выводить или не выводить график прогноза
    """
    df = load_data(f'data/series{id}.csv')
    train, test = split_data_without_val(id, df)
    exp_train = train[['ds', 'y']]
    exp_test = test[['ds', 'y']]
    
    exp_sm = ExponentialSmoothing(exp_train,
                                  trend = 'add',
                                  seasonal_periods = 7,
                                  seasonal = 'add',
                                  freq = 'D')
    exp_sm = exp_sm.fit()
    exp_sm_forecast = exp_sm.forecast(len(test))
    metrics = calculate_metrics(exp_test, exp_sm_forecast)

    if plot_forecast:
        plot_forecasts(exp_test, [exp_sm_forecast], ['exp smoothing'])

    return metrics, exp_sm_forecast


exp_metrics, exp_forecast = run_exp_smoothing_eval(1)
print(metrics)