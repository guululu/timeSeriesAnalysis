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

    _, y_oof = fn(model_stacking=True, **optimized_parameters)

    return optimized_parameters, y_oof


def stock_feature(code):
    df = load_stock_data(code)
    df_macd = TA.MACD(df).rename(columns={'MACD': f"{code}_MACD", 'SIGNAL': f"{code}_SIGNAL"})
    df_vbm = TA.VBM(df).to_frame().rename(columns={'VBM': f"{code}_VBM"})
    df_ewm = TA.EVWMA(df, period=5).to_frame().rename(columns={'5 period EVWMA.': f'{code}_EVWMA'})

    return pd.concat([df_macd, df_vbm, df_ewm], axis=1).fillna(method='bfill')

# def time_series_train_test_split(df: pd.DataFrame, frac: float, lag:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
#
#     assert (frac < 1) and (frac > 0)
#
#     df_train = df[:int(frac * len(df))].fillna(0)
#     df_test = df[int(frac * len(df)) - lag:].fillna(0)
#
#     return df_train, df_test
#
#
# def data_normalization(df_train: pd.DataFrame, df_test: pd.DataFrame):
#
#     normalizer = MinMaxScaler()
#
#     df_train_normalized = normalizer.fit_transform(df_train)
#     df_test_normalized = normalizer.transform(df_test)
#
#     return df_train_normalized, df_test_normalized
#
#
# def feature_stage_one_preparation(df: pd.DataFrame, history: int, future: int, extension: int, frac: float):
#
#     self_df_train, self_df_test = time_series_train_test_split(df, frac=frac, lag=(history + future + extension))
#     self_train_normalized, self_test_normalized = data_normalization(self_df_train, self_df_test)
#
#     return self_train_normalized, self_test_normalized
#
#
# def target_stage_one_preparation(df: pd.DataFrame, history: int, future: int, extension: int, frac: float, label: str):
#
#     scaler = MinMaxScaler()
#
#     y_df_train, y_df_test = time_series_train_test_split(df[[label]], frac=frac, lag=(history + future + extension))
#
#     y_train_norm = scaler.fit_transform(y_df_train).reshape(-1, )
#     y_test_norm = scaler.transform(y_df_test).reshape(-1, )
#
#     return y_train_norm, y_test_norm
#
#
# def add_features(feature_dict: OrderedDict, set_: str, history: int, total_features: int, start_index: int):
#
#     feature_idx = 0
#     time_block = np.zeros((history, total_features))
#
#     for key in feature_dict:
#         features = feature_dict[key][set_]
#         _, n_features = features.shape
#         for period in feature_dict[key]['Period']:
#             lag_start, lag_end = period
#             period_length = lag_end - lag_start
#             assert period_length > 0
#             time_block[history - period_length: history, feature_idx: feature_idx + n_features] = \
#             features[start_index + lag_start: start_index + lag_end]
#             feature_idx += n_features
#
#     return time_block
#
#
# def label_aggregation(label_array, duration: int):
#
#     _, extension = label_array.shape
#
#     weekly_label = np.array([label_array[:, i * duration:(i + 1) * duration].sum(axis=1, keepdims=True) for
#                              i in range(extension // duration)])
#
#     four_weeks_mean = weekly_label.mean(axis=0)
#     target_week = weekly_label[0, :, :]
#
#     label = np.concatenate([four_weeks_mean, target_week], axis=1).mean(axis=1)
#
#     return np.expand_dims(label, -1)


# def feature_aggregation(df: pd.DataFrame, history: int, future: int, extension: int, duration: int):
#     X_days = np.array([df[i: i + history] for i in range(len(df) - history - future - extension)])
#     weekly_data = [X_days[:, i * duration:(i + 1) * duration, :].sum(axis=1, keepdims=True) for i in
#                    range(history // duration)]
#     X_week = np.concatenate(weekly_data, axis=1)
#
#     return X_week
#
#
# def target_aggregation(df: pd.DataFrame, history: int, future: int, extension: int, duration: int, target: str):
#     y_days = np.array([df[target].values[i: i + extension] for i in range(history + future, len(df) - extension)])
#     weekly_data = np.array(
#         [y_days[:, i * duration:(i + 1) * duration].sum(axis=1, keepdims=True) for i in range(extension // duration)])
#
#     four_weeks_mean = weekly_data.mean(axis=0)
#     target_week = weekly_data[0, :, :]
#
#     y_week = np.concatenate([four_weeks_mean, target_week], axis=1).mean(axis=1)
#
#     return y_week
#
#
# def data_transformation(df_train: pd.DataFrame, df_test: pd.DataFrame, history: int, future: int, extension: int,
#                         duration: int, target: str):
#     df_train_normalized, df_test_normalized = data_normalization(df_train, df_test)
#
#     X_train = feature_aggregation(df_train_normalized, history=history, future=future, extension=extension,
#                                   duration=duration)
#     X_test = feature_aggregation(df_test_normalized, history=history, future=future, extension=extension,
#                                  duration=duration)
#
#     assert (target in df_train.columns) and (target in df_test.columns)
#
#     y_train = target_aggregation(df_train, history=history, future=future, extension=extension, duration=duration,
#                                  target=target)
#     y_test = target_aggregation(df_test, history=history, future=future, extension=extension, duration=duration,
#                                 target=target)
#
#     return X_train, X_test, y_train, y_test




