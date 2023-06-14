import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

from common_utils import *
from data_prep import split_data_without_val
from forecast_funcs import calculate_metrics


def run_exp_smoothing_eval(id: int, plot_forecast = True):
    """
    Функция для оценки точности экспоненциального сглаживания на рядах, представленных в ВКР
    Args:
        id (int): номер ряда, 1 или 2
        plot_forecast (bool, optional): Выводить или не выводить график прогноза
    """
    df = load_data(f'data/series{id}.csv')
    train, test = split_data_without_val(id, df)
    exp_train = train[['ds', 'y']].set_index(['ds'])
    #exp_test = test[['ds', 'y']].set_index(['ds'])
    exp_test = test[['ds', 'y']]

    exp_sm = ExponentialSmoothing(exp_train,
                                  trend = 'add',
                                  seasonal_periods = 7,
                                  seasonal = 'add',
                                  freq = 'D')
    exp_sm = exp_sm.fit()
    exp_sm_forecast = exp_sm.forecast(len(test))
    metrics = calculate_metrics(exp_test, exp_sm_forecast, id = id)

    exp_test.reset_index(inplace = True)
    if plot_forecast:
        plot_forecasts(exp_test, [exp_sm_forecast], ['exp smoothing'])

    return metrics, exp_sm_forecast


exp_metrics, exp_forecast = run_exp_smoothing_eval(1)
print(exp_metrics)