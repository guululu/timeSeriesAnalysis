# from typing import Tuple
# from collections import OrderedDict
#
# from sklearn.preprocessing import MinMaxScaler

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from finta import TA
from bayes_opt import BayesianOptimization

from main.settings import bayesianOptimization
from main.utils.data import load_stock_data


def optimization_process(fn, pbounds: Dict) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions

    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn

    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    optimizer.maximize(
        **bayesianOptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters


def stock_feature(code):
    df = load_stock_data(code)
    df_macd = TA.MACD(df).rename(columns={'MACD': f"{code}_MACD", 'SIGNAL': f"{code}_SIGNAL"})
    df_vbm = TA.VBM(df).to_frame().rename(columns={'VBM': f"{code}_VBM"})
    df_ewm = TA.EVWMA(df, period=5).to_frame().rename(columns={'5 period EVWMA.': f'{code}_EVWMA'})

    return pd.concat([df_macd, df_vbm, df_ewm], axis=1).fillna(method='bfill')


def to_supervised(train, train_label, n_input: int, n_out: int, n_gap: int,
                  numerical_columns_index=None, export_future_time_info: bool = False):
    '''
    Args:
        n_input: days
        n_out: measured in weeks
        n_future: measured in weeks
    '''

    # Multivariant input
    # flatten data

    X, y, X_weekly = list(), list(), list()
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))

    if export_future_time_info:
        data_timestamp = np.delete(data, numerical_columns_index, axis=1)
        data = data[:, numerical_columns_index]
        X_timestamp = list()

    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data) - (n_out + n_gap) * 7):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_start = in_end + 7 * n_gap
        out_end = out_start + 7 * n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(np.array(np.split(train_label[out_start: out_end], n_out)).sum(axis=1))
            X_weekly.append(np.array(np.split(data[in_start:in_end, :], n_out)).sum(axis=1))
            if export_future_time_info:
                X_timestamp.append(np.array(np.split(data_timestamp[out_start:out_end, :], n_out))[:, 0])
        # add another week
        in_start += 7

    if export_future_time_info:
        return np.array(X), np.array(y), np.array(X_weekly), np.array(X_timestamp)
    else:
        return np.array(X), np.array(y), np.array(X_weekly)


def to_supervised_time(train: np.ndarray, n_input: int, n_out: int, n_gap: int):

    '''
    Args:
        n_input: days
        n_out: measured in weeks
        n_future: measured in weeks
    '''

    # Multivariant input
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X = list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data) - (n_out + n_gap) * 7):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_start = in_end + 7 * n_gap
        out_end = out_start + 7 * n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(np.array(np.split(data[out_start:out_end, :], n_out))[:, 0])  # take the day of first day
        # move along one time step
        in_start += 7
    return np.array(X)




