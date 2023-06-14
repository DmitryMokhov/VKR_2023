import pandas as pd
import numpy as np
from typing import Union

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
    d['week'] = d['ds'].dt.isocalendar().week
    d['month_start'] = d['ds'].dt.is_month_start.apply(lambda x: 0 if x == False else 1)
    d['month_end'] = d['ds'].dt.is_month_end.apply(lambda x: 0 if x == False else 1)
    d['quarter_start'] = d['ds'].dt.is_quarter_start.apply(lambda x: 0 if x == False else 1)
    d['quarter_end'] = d['ds'].dt.is_quarter_end.apply(lambda x: 0 if x == False else 1)
    d['year_start'] = d['ds'].dt.is_year_start.apply(lambda x: 0 if x == False else 1)
    d['year_end'] = d['ds'].dt.is_year_end.apply(lambda x: 0 if x == False else 1)
    d['days_in_month'] = d['ds'].dt.days_in_month
    d['is_holiday'] = d.ds.apply(lambda x: 1 if str(x) in holidays else 0)
    return d


def plot_forecasts(fact: pd.DataFrame, forecasts: Union[list, np.array], labels: list):
    """
    Отрисовка графиков прогнозов
    Args:
        fact (pd.DataFrame): датафрейм, содержищий фактические значения ряда и колонку дат ds,
        forecasts (list | np.array): список прогнозов,
        labels (list): список названий для отображения на графике
    """
    fig = go.Figure()
    fig.add_trace(go.Line(x = fact.ds, y = fact.y, name = 'fact'))
    for forc, label in zip(forecasts, labels):
        fig.add_trace(go.Line(x = fact.ds, y = forc, name = 'label'))
    fig.show()


def load_data(path: str):
    data = pd.read_csv(path, sep = ';')
    data['ds'] = pd.to_datetime(data['ds'])
    return data


def get_future_dates(series: int):
    if series == 1:
        future_dates = pd.date_range('2023-03-29', '2023-05-19')
        return future_dates
    elif series == 2:
        future_dates = pd.date_range('2023-02-08', '2023-03-31')
        return future_dates
    else:
        raise ValueError('No such series')