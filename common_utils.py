import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go


def create_features(df: pd.DataFrame, holidays: list):
    """
    Функция для создания временных признаков
    Параметры:
        df: датафрейм с временным рядов; обязательно должен содержать колонку 'ds', 
            в которой указаны даты,
        holidays: список дат, которые считаются выходными

    Возвращает:
        pd.DataFrame с временными рядом, дополненным временными признаками
    """
    
    d = df.copy()
    d['ds'] = pd.to_datetime(d['ds'])
    d['dayofweek'] = d['ds'].dt.dayofweek
    d['quarter'] = d['ds'].dt.quarter
    d['month'] = d['ds'].dt.month
    d['year'] = d['ds'].dt.year
    d['dayofyear'] = d['ds'].dt.dayofyear
    d['dayofmonth'] = d['ds'].dt.day
    d['week'] = d['ds'].dt.week
    d['month_start'] = d['ds'].dt.is_month_start.apply(lambda x: 0 if x == False else 1)
    d['month_end'] = d['ds'].dt.is_month_end.apply(lambda x: 0 if x == False else 1)
    d['quarter_start'] = d['ds'].dt.is_quarter_start.apply(lambda x: 0 if x == False else 1)
    d['quarter_end'] = d['ds'].dt.is_quarter_end.apply(lambda x: 0 if x == False else 1)
    d['year_start'] = d['ds'].dt.is_year_start.apply(lambda x: 0 if x == False else 1)
    d['year_end'] = d['ds'].dt.is_year_end.apply(lambda x: 0 if x == False else 1)
    d['days_in_month'] = d['ds'].dt.days_in_month
    d['is_holiday'] = d.ds.apply(lambda x: 1 if str(x) in holidays else 0)
    return d


def plot_forecasts(fact: pd.DataFrame, forecasts: list, labels: list):
    """
    Отрисовка графиков прогнозов
    Args:
        fact (pd.DataFrame): датафрейм, содержищий фактические значения ряда и колонку дат ds,
        forecasts (list): список прогнозов, сами прогнозы могут быть либо списком, либо np.array,
        labels (list): список названий для отображения на графике
    """
    fig = go.Figure()
    fig.add_trace(go.Line(x = fact.ds, y = fact.y, name = 'fact'))
    for forc, label in zip(forecasts, labels):
        fig.add_trace(go.Line(x = fact.ds, y = forc, name = 'label'))
    fig.show()


def load_data(path: str):
    data = pd.read_csv(path, sep = ';')
    return data


def retrive_ds(series: int):
    if series == 1:
        ds = pd.date_range('2021-07-01', '2023-05-22')
        future_dates = pd.date_range('2023-03-29', '2023-05-22')
        return ds, future_dates
    elif series == 2:
        ds = pd.date_range('2019-01-01', '2023-03-31')
        future_dates = pd.date_range('2023-02-08', '2023-03-31')
        return ds, future_dates
    else:
        raise ValueError('No such series')