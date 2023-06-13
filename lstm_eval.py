from common_utils import *
from forecast_funcs import *
from data_prep import *


def run_lstm_eval(id: int, plot_forecast = True):
    """
    Проверка точности прогноза lstm-сети
    Параметры:
        id (int): номер ряда, 1 или 2
        plot_forecast (bool, optional): нужно ли выводить график прогноза
    """
    df = load_data(f'data/series{id}.csv')
    train, val, test = split_data_with_val(id, df)
    metrics, forecast = lstm_loop(train, val, test, 7)

    if plot_forecast:
        plot_forecasts(test, [forecast], ['LSTM'])

    return metrics, forecast

lstm_metrics, lstm_forecast = run_lstm_eval(1)
print(lstm_metrics)