from typing import Dict, Tuple, List, Optional

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


def to_supervised(train, train_label, n_input: int, n_out: int, n_gap: int, day_increment: int = 7,
                  statistical_operation: Dict = None, timestamp_columns_index=None):
    '''
    Args:
        n_input: days
        n_out: measured in weeks
        n_future: measured in weeks
        statistical_operation: used only for weekly2weekly model
    '''

    # Multivariant input
    # flatten data

    X, y, X_timestamp, X_weekly, y_weekly = list(), list(), list(), list(), list()

    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))

    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data) - (n_out + n_gap) * 7):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_start = in_end + 7 * n_gap
        out_end = out_start + 7 * n_out
        # ensure we have enough data for this instance

        X = []
        feature_data_weekly = []

        if out_end <= len(data):
            target_data = train_label[out_start: out_end]
            target_data_weekly = np.array(np.split(target_data, n_out)).sum(axis=1)
            if timestamp_columns_index:
                X_timestamp.append(data[out_start:out_end, timestamp_columns_index])
            for col_index, operations in statistical_operation.items():
                for operation in operations:
                    statistical_feature = eval(f'np.nan{operation}(np.array(np.split(data[in_start:in_end, '
                                               f'col_index], n_input // 7)), axis=1, keepdims=True)')
                    if np.isnan(np.sum(statistical_feature)):
                        statistical_feature = np.nan_to_num(statistical_feature)
                    # statistical_feature = np.expand_dims(statistical_feature, 1)
                    feature_data_weekly.append(statistical_feature)
            feature_data_weekly = np.concatenate(feature_data_weekly, axis=1)
            y.append(target_data)
            X_weekly.append(feature_data_weekly)
            y_weekly.append(target_data_weekly)

        # add another week
        in_start += day_increment

    results = {'X': np.array(X), 'X_weekly': np.array(X_weekly),
               'X_timestamp': np.array(X_timestamp),
               'y': np.array(y), 'y_weekly': np.array(y_weekly)}

    return results


def time_series_data_preparation(train, train_label, n_input,
                                 n_out=4, n_gap=7, day_increment=7, statistical_operation: Optional[Dict] = None,
                                 timestamp_columns_index: Optional[List] = None):
    # prepare data
    data_dict = to_supervised(train, train_label, n_input, n_out=n_out,
                              n_gap=n_gap, day_increment=day_increment, statistical_operation=statistical_operation,
                              timestamp_columns_index=timestamp_columns_index)

    train_x = data_dict['X']
    train_y = data_dict['y']
    train_x_weekly = data_dict['X_weekly']
    train_x_time = data_dict['X_timestamp']
    train_y_weekly = data_dict['y_weekly']

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    train_y_weekly = train_y_weekly.reshape((train_y_weekly.shape[0], train_y_weekly.shape[1], 1))

    return train_x, train_x_weekly, train_y, train_x_time, train_y_weekly
