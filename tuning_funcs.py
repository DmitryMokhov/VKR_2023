import pandas as pd
import numpy as np
import time
from prophet import Prophet
from catboost import CatBoostRegressor as cbr
from itertools import product
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape


def tune_prophet(tune_train, tune_valid, error_func):
    """
    Функция для подбора гиперпараметров Prophet
    Параметры:
        tune_train: данные для обучения,
        tune_valid: данные для оценки качества
        error_func: функционал ошибок, по которому ранжируются комбинации параметров

    Returns:
        best_params: dict - словарь с лучшими подобранными параметрами
    """

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 1, 10],
        'seasonality_prior_scale': [0.01, 0.1, 1, 10],
        'holidays_prior_scale': [0.01, 0.1, 1, 10],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.85, 0.9, 0.95, 0.99]
    }

    errors = []
    params_combs = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    start_time = time.time()
    for comb in params_combs:
        model = Prophet(**comb,
                        yearly_seasonality = 'auto',
                        weekly_seasonality = True,
                        daily_seasonality = False)
        model.add_seasonality(name = 'monthly', period = 30.4, 
                            fourier_order = 5, prior_scale = 10, mode = 'multiplicative')
        model.fit(tune_train)
        pred = model.predict(tune_valid)
        cur_error = error_func(tune_valid.reset_index().y, pred.reset_index().yhat)
        errors.append(cur_error)

    results = pd.DataFrame(params_combs)
    results['error'] = errors
    results.sort_values(by = ['error'], ascending = True, inplace = True)
    best_params = params_combs[np.argmin(errors)]
    print(f'tuning time: {round(time.time() - start_time)}')
    print(results.head(5))
    return best_params


def tune_catboost(tune_train, tune_valid, error_func):
    """
    Функция для подбора гиперпараметров Catboost
    Args:
        tune_train: catboost.Pool - данные для обучения,
        tune_valid: catboost.Pool - данные для обучения,
        error_func: catboost.Pool -  ошибки для ранжирования комбинаций параметров.

    Returns:
        best_params: dict - лучшая комбинация гиперпараметров
    """

    cat_params_grid = {
        'eta': [0.01, 0.05, 0.1, 0.5, 0.1],
        'iterations': [100, 250, 500, 1000],
        'depth': [5, 6, 7, 8, 9],
        'l2_leaf_reg': [0, 1, 5, 10, 15],
        'subsample': [0.5, 0.66, 0.8, 0.9],
    }

    cat_combs = [dict(zip(cat_params_grid.keys(), v)) for v in product(*cat_params_grid.values())]
    errors = []
    start_time = time.time()
    iters_count = 0
    for comb in cat_combs:
        if iters_count % 50 == 0:
            print(f'iter {iters_count} started')
        cat = cbr(**comb,
                loss_function = 'MAE')
        cat.fit(tune_train,
                eval_set = tune_valid,
                early_stopping_rounds = 25,
                verbose = False)
        pred = cat.predict(tune_valid)
        cur_error = error_func(tune_valid.get_label(), pred)
        errors.append(cur_error)
        iters_count += 1

    results = pd.DataFrame(cat_combs)
    results['error'] = errors
    results.sort_values(by = 'error', ascending = True, inplace = True)
    best_params = cat_combs[np.argmin(errors)]
    print(f'tuning time: {round(time.time() - start_time)}')
    print(results.head(5))
    return best_params