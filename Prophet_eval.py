import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
from itertools import product
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

import plotly
import plotly.express as px
import plotly.graph_objects as go

from common_utils import*
from forecast_funcs import *
from data_prep import *
from tuning_funcs import *


def run_prophet_eval(id: int, plot_forecast = True):
    """
    Функция для оценки точности Prophet на рядах, представленных в ВКР
    Args:
        series (int): номер ряда - 1 или 2
    """
    df = load_data(f'data/series{id}.csv')
    df = create_features(df, holidays=holidays)
    train, test = split_data_without_val(id, df)
    
    prop_best_params = tune_prophet(train, test, mae)
    metrics, forecast = prophet_forecast(train, test, prop_best_params)

    if plot_forecast:
        plot_forecasts(test, [forecast.y], ['Prophet'])

    return metrics, forecast

prop_metrics, prop_forecast = run_prophet_eval(1)
print(prop_metrics)
